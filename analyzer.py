# -*- coding: utf-8 -*-
"""
Generador de Tablas y Parser LR(1) en Python.

La lógica está encapsulada en la clase LR1Analyzer, permitiendo
que la gramática y la cadena de entrada sean proporcionadas por el usuario
(o un frontend), haciendo el código más modular y reutilizable.
"""
import collections
import sys
import re  # <--- MEJORA: Importar re para tokenización

# Definiciones de Símbolos Globales
EOF = "$"
EMPTY = "ε"


class ASTNode:
    """Nodo del Árbol de Sintaxis Abstracta (AST)"""

    def __init__(self, symbol, children=None, value=None):
        self.symbol = symbol
        self.children = children if children is not None else []
        self.value = value

    def __str__(self):
        return self._to_string()

    def _to_string(self, level=0):
        indent = "  " * level
        result = f"{indent}{self.symbol}"
        if self.value:
            result += f" ({self.value})"
        result += "\n"
        for child in self.children:
            if isinstance(child, ASTNode):
                result += child._to_string(level + 1)
            else:
                # Si el hijo no es un nodo, es un token terminal
                result += f"{indent}  {child.symbol}\n"
        return result


class LR1Analyzer:
    """
    Clase para manejar todo el proceso de análisis LR(1) para una gramática dada.
    """

    def __init__(self, grammar_str):
        self.grammar_str = grammar_str
        self.GRAMMAR = []  # Lista de tuplas: (NoTerminal, [Producción])
        self.NON_TERMINALS = set()
        self.TERMINALS = set()
        self.START_SYMBOL = None
        self.FIRST = collections.defaultdict(set)
        self.ast = None

        # 1. Inicializar y Aumentar la Gramática
        self._initialize_grammar()
        # 2. Calcular Conjuntos First
        self._calculate_first_sets()

    # --- Métodos de Inicialización y Gramática ---

    def _parse_grammar_string(self):
        """Convierte una cadena de gramática de usuario a la lista de tuplas GRAMMAR."""
        rules = []
        non_terminals = set()
        all_symbols = set()

        # Formato de entrada esperado: E -> T + E | T (una regla por línea)
        for line in self.grammar_str.strip().split('\n'):
            if not line.strip(): continue

            try:
                nt_part, prods_part = line.split('->', 1)
                non_terminal = nt_part.strip()
                non_terminals.add(non_terminal)

                prods = [p.strip().split() for p in prods_part.split('|')]

                for prod in prods:
                    # <--- FIX 1: Manejar explícitamente '' como símbolo de épsilon
                    if prod == ["''"]:
                        rules.append((non_terminal, [EMPTY]))
                    else:
                        rules.append((non_terminal, prod))
                        all_symbols.update(prod)
            except ValueError:
                raise ValueError(f"Formato de regla inválido: '{line}'. Use 'A -> B C | d'")

        if not rules:
            raise ValueError("La gramática no puede estar vacía.")

        # Determinar el Símbolo Inicial (el no-terminal de la primera regla)
        self.START_SYMBOL = rules[0][0]

        # 1. Aumento de la Gramática: S' -> S
        start_prime = self.START_SYMBOL + "'"
        self.GRAMMAR.append((start_prime, [self.START_SYMBOL]))
        self.NON_TERMINALS.add(start_prime)

        # Añadir el resto de reglas (manteniendo el orden para indexación)
        self.GRAMMAR.extend(rules)
        self.NON_TERMINALS.update(non_terminals)

        # 2. Determinar Terminales
        self.TERMINALS = (all_symbols - self.NON_TERMINALS)
        if EMPTY in self.TERMINALS:
            self.TERMINALS.remove(EMPTY)
        self.TERMINALS.add(EOF)

        # Ordenar para consistencia
        self.NON_TERMINALS = sorted(list(self.NON_TERMINALS), reverse=True)
        self.TERMINALS = sorted(list(self.TERMINALS))

    def _initialize_grammar(self):
        """Inicializa los sets y la gramática aumentada."""
        self._parse_grammar_string()

    def _calculate_first_sets(self):
        """Calcula el conjunto FIRST para todos los símbolos."""

        # Inicialización con terminales
        for t in self.TERMINALS:
            self.FIRST[t].add(t)

        self.FIRST[EMPTY].add(EMPTY)  # Asegurar que EMPTY tiene su propio conjunto FIRST

        changed = True
        while changed:
            changed = False
            for A, production in self.GRAMMAR:

                # 1. Si A -> ε
                if production == [EMPTY]:
                    if EMPTY not in self.FIRST[A]:
                        self.FIRST[A].add(EMPTY)
                        changed = True
                    continue

                # 2. Si A -> X1 X2 ... Xk
                for X in production:
                    if X not in self.FIRST:  # Símbolo no reconocido, podría ser un error de gramática
                        continue

                    old_size = len(self.FIRST[A])
                    self.FIRST[A].update(self.FIRST[X] - {EMPTY})
                    if len(self.FIRST[A]) != old_size:
                        changed = True

                    # Si X no deriva a ε, terminamos la secuencia
                    if EMPTY not in self.FIRST[X]:
                        break
                else:
                    # Si todos los Xi derivan a ε, añadir ε a First(A)
                    if EMPTY not in self.FIRST[A]:
                        self.FIRST[A].add(EMPTY)
                        changed = True

    # --- Clase Item (Anidada para acceso a la gramática) ---

    class LR1Item:
        """Representa un ítem LR(1). Se inicializa con el objeto Analyzer para acceder a GRAMMAR."""

        def __init__(self, rule_index, dot_position, lookahead, analyzer):
            self.rule_index = rule_index
            self.dot_position = dot_position
            self.lookahead = lookahead
            self.analyzer = analyzer
            self.non_terminal = self.analyzer.GRAMMAR[rule_index][0]
            self.production = self.analyzer.GRAMMAR[rule_index][1]

        def is_final(self):
            return self.dot_position >= len(self.production) or self.production == [EMPTY]

        def next_symbol(self):
            if self.is_final():
                return None
            return self.production[self.dot_position]

        def to_string(self):
            prod_display = self.production[:]
            if prod_display == [EMPTY]:
                prod_display = ["ε"]

            prod_display.insert(self.dot_position, "•")
            return f"[{self.non_terminal} -> {' '.join(prod_display)}, {self.lookahead}]"

        def __eq__(self, other):
            return (self.rule_index == other.rule_index and
                    self.dot_position == other.dot_position and
                    self.lookahead == other.lookahead)

        def __hash__(self):
            return hash((self.rule_index, self.dot_position, self.lookahead))

    # --- Métodos de Transición ---

    def _closure(self, I):
        """Calcula el cierre (closure) del conjunto de ítems I."""
        J = set(I)
        worklist = list(I)  # Usar una lista de trabajo en lugar de bucles anidados

        while worklist:
            item = worklist.pop(0)

            X = item.next_symbol()
            if X is None or X in self.TERMINALS:
                continue

            # Regla: [A -> alpha . X beta, a]
            beta_symbols = item.production[item.dot_position + 1:]

            # 1. Calcular First(beta a)
            first_beta_a = set()
            epsilon_in_beta = True

            for B in beta_symbols:
                first_beta_a.update(self.FIRST[B] - {EMPTY})
                if EMPTY not in self.FIRST[B]:
                    epsilon_in_beta = False
                    break

            if epsilon_in_beta:
                first_beta_a.add(item.lookahead)

            # 2. Aplicar la regla: para cada producción X -> gamma y cada b en First(beta a)
            for rule_index, (nt, production) in enumerate(self.GRAMMAR):
                if nt == X:
                    for b in first_beta_a:
                        new_item = self.LR1Item(rule_index, 0, b, self)
                        if new_item not in J:
                            J.add(new_item)
                            worklist.append(new_item)
        return J

    def _goto(self, I, X):
        """Calcula el conjunto de ítems que se alcanza al leer el símbolo X desde el conjunto I."""
        J = set()
        for item in I:
            if item.next_symbol() == X:
                # Avanza el punto
                new_item = self.LR1Item(item.rule_index, item.dot_position + 1, item.lookahead, self)
                J.add(new_item)
        return self._closure(J)

    # --- Generación de Tablas ---

    def _build_canonical_collection(self):
        """Construye la colección canónica C y la tabla de transiciones GOTO."""
        # Ítem inicial: [S' -> . S, $]
        initial_item = self.LR1Item(0, 0, EOF, self)
        I0 = self._closure({initial_item})

        C = [I0]
        # <--- MEJORA: Usar un mapa de frozenset para búsqueda O(1) en lugar de O(n) con C.index()
        states_map = {frozenset(I0): 0}

        goto_map = {}  # { (state_index, symbol): new_state_index }

        # Símbolos de transición: todos los Terminales y No-Terminales
        symbols = self.NON_TERMINALS + self.TERMINALS

        queue = collections.deque([0])  # Cola de índices de estados

        while queue:
            i = queue.popleft()
            I = C[i]

            for X in symbols:
                if X == EOF or X == EMPTY: continue
                J = self._goto(I, X)

                if not J:
                    continue

                J_frozen = frozenset(J)
                if J_frozen not in states_map:
                    j = len(C)
                    states_map[J_frozen] = j
                    C.append(J)
                    queue.append(j)
                else:
                    j = states_map[J_frozen]

                goto_map[(i, X)] = j

        return C, goto_map

    def _generate_parsing_tables(self, C, goto_map):
        """Genera las tablas ACTION y GOTO a partir de la colección canónica C."""

        action_table = collections.defaultdict(dict)
        goto_output_table = collections.defaultdict(dict)

        for i, I in enumerate(C):

            # 4. Tabla GOTO (No-Terminales)
            for nt in self.NON_TERMINALS:
                if nt != self.GRAMMAR[0][0] and (i, nt) in goto_map:
                    j = goto_map[(i, nt)]
                    goto_output_table[i][nt] = j

            # Recolectar todas las acciones potenciales para este estado i
            potential_actions = collections.defaultdict(list)  # {terminal: [acciones]}

            for item in I:
                t = item.next_symbol()

                # Caso 1: Acción de Desplazamiento (Shift)
                if t in self.TERMINALS and t != EOF:
                    if (i, t) in goto_map:
                        j = goto_map[(i, t)]
                        potential_actions[t].append(f"s{j}")

                # Caso 2: Acción de Reducción/Aceptación
                elif item.is_final():
                    a = item.lookahead

                    if item.rule_index == 0:  # S' -> S .
                        # Aceptación
                        potential_actions[EOF].append("acc")
                    else:
                        # Reducción
                        r_index = item.rule_index
                        potential_actions[a].append(f"r{r_index}")

            # Procesar acciones potenciales y detectar conflictos
            for terminal, actions in potential_actions.items():
                unique_actions = sorted(list(set(actions)))

                # <--- FIX 2: Lanzar un error en caso de conflicto en lugar de resolverlo
                if len(unique_actions) > 1:
                    conflict_type = "Shift/Reduce" if any(a.startswith('s') for a in unique_actions) and any(
                        a.startswith('r') for a in unique_actions) else "Reduce/Reduce"
                    raise ValueError(
                        f"CONFLICTO {conflict_type} detectado en el estado {i} con el terminal '{terminal}'. "
                        f"Acciones conflictivas: {', '.join(unique_actions)}. La gramática no es LR(1)."
                    )

                elif len(unique_actions) == 1:
                    # Acción única (Shift, Reduce, o Accept)
                    action_table[i][terminal] = unique_actions[0]

        return action_table, goto_output_table

    # --- Driver de Parsing ---

    def _parse_input(self, input_tokens, action_table, goto_table):
        """Simula el proceso de parsing con las tablas generadas y construye el AST."""
        input_tokens.append(EOF)
        stack = [0]
        value_stack = []  # Pila para construir el AST
        pointer = 0
        trace = []

        while True:
            current_state = stack[-1]
            current_token = input_tokens[pointer]
            action = action_table.get(current_state, {}).get(current_token)

            stack_str = ' '.join(str(s) for s in stack)
            input_str = ' '.join(input_tokens[pointer:])

            trace.append({
                "stack": stack_str,
                "input": input_str,
                "action": action if action else "ERROR"
            })

            if action is None:
                trace[-1][
                    "action"] = f"ERROR: No hay acción definida en estado {current_state} con token '{current_token}'."
                break

            if action.startswith('s'):
                # Shift (Desplazamiento)
                next_state = int(action[1:])
                stack.append(current_token)
                stack.append(next_state)
                value_stack.append(ASTNode(current_token))
                pointer += 1

            elif action.startswith('r'):
                # Reduce (Reducción)
                rule_index = int(action[1:])
                nt, prod = self.GRAMMAR[rule_index]
                prod_len = len(prod) if prod != [EMPTY] else 0

                children = []
                for _ in range(prod_len):
                    stack.pop()  # estado
                    stack.pop()  # símbolo
                    if value_stack:
                        children.insert(0, value_stack.pop())

                new_node = ASTNode(nt, children)
                value_stack.append(new_node)

                prev_state = stack[-1]
                next_state = goto_table.get(prev_state, {}).get(nt)

                if next_state is None:
                    trace.append({
                        "stack": ' '.join(str(s) for s in stack),
                        "input": input_str,
                        "action": f"ERROR: No hay GOTO definido en estado {prev_state} con no-terminal '{nt}'."
                    })
                    break

                stack.append(nt)
                stack.append(next_state)
                trace[-1]["action"] += f" ({nt} -> {' '.join(prod)})"

            elif action == 'acc':
                trace[-1]["action"] = "ACEPTADO"
                if value_stack:
                    self.ast = value_stack[0]
                break

            else:
                break  # Error desconocido

        return trace

    # Modifica el método analyze para incluir el AST en los resultados
    def analyze(self, input_string):
        """Ejecuta el análisis LR(1) completo y retorna un diccionario de resultados."""

        # <--- FIX 3: Tokenizer mejorado
        # Separa operadores, paréntesis, llaves, etc., pero mantiene literales de texto.
        tokens = re.findall(r'\"[^"]*\"|\b\w+\b|[(){};,=\+\-\*\/]', input_string)

        # 1. Construir la Colección Canónica
        C, GOTO_MAP = self._build_canonical_collection()

        # 2. Generar Tablas de Análisis
        ACTION_TABLE, GOTO_TABLE = self._generate_parsing_tables(C, GOTO_MAP)

        # 3. Demostración de Parsing
        parse_trace = self._parse_input(tokens, ACTION_TABLE, GOTO_TABLE)

        # 4. Formatear la Salida para impresión/frontend

        # Formato de GRAMMAR: NT -> Prod (Indice)
        formatted_grammar = [f"({i}) {nt} -> {' '.join(prod)}" for i, (nt, prod) in enumerate(self.GRAMMAR)]

        # Formato de estados: Ii: [Item1], [Item2], ...
        formatted_states = {}
        for i, I in enumerate(C):
            formatted_states[f"I{i}"] = [item.to_string() for item in
                                         sorted(list(I), key=lambda x: (x.rule_index, x.dot_position))]

        return {
            "grammar": formatted_grammar,
            "terminals": self.TERMINALS,
            "non_terminals": self.NON_TERMINALS,
            "states": formatted_states,
            "action_table": ACTION_TABLE,
            "goto_table": GOTO_TABLE,
            "trace": parse_trace,
            "num_states": len(C),
            "ast": self.ast
        }


# --- 8. Ejecución Principal con Input de Usuario ---
if __name__ == "__main__":
    # --- PARTE 1: DEFINICIÓN DE LA GRAMÁTICA Y CADENA DE ENTRADA ---
    GRAMMAR_INPUT = """
    Programa -> Items
    Items -> Item Items
    Items -> ''
    Item -> Declaracion
    Item -> Funcion
    Declaracion -> Tipo Identificador DeclaracionTail ;
    DeclaracionTail -> = Expresion
    DeclaracionTail -> ''
    Funcion -> Tipo Identificador ( Parametros ) Bloque
    Parametros -> Tipo Identificador MasParametros
    Parametros -> ''
    MasParametros -> , Tipo Identificador MasParametros
    MasParametros -> ''
    Bloque -> { Sentencias }
    Sentencias -> Sentencia Sentencias
    Sentencias -> ''
    Sentencia -> SentenciaM
    Sentencia -> SentenciaU
    SentenciaM -> Declaracion
    SentenciaM -> Asignacion ;
    SentenciaM -> While
    SentenciaM -> Return ;
    SentenciaM -> Bloque
    SentenciaM -> if ( Expresion ) SentenciaM else SentenciaM
    SentenciaU -> if ( Expresion ) Sentencia
    SentenciaU -> if ( Expresion ) SentenciaM else SentenciaU
    Asignacion -> Identificador = Expresion
    While -> while ( Expresion ) Sentencia
    Return -> return Expresion
    Expresion -> Expresion + Termino
    Expresion -> Expresion - Termino
    Expresion -> Termino
    Termino -> Termino * Factor
    Termino -> Termino / Factor
    Termino -> Factor
    Factor -> ( Expresion )
    Factor -> Identificador
    Factor -> Numero
    Factor -> Literal
    Tipo -> int
    Tipo -> float
    Tipo -> double
    Tipo -> char
    Tipo -> bool
    Tipo -> string
    Tipo -> void
    Identificador -> id
    Numero -> num
    Literal -> "texto"
    """

    INPUT_STRING = "int id ( ) { int id = num ; if ( id - num ) id = id + num ; else { id = id * num ; } return id ; }"

    print("--- PARSER LR(1) CON INPUT DINÁMICO ---")
    print(f"Gramática a analizar:\n{GRAMMAR_INPUT.strip()}")
    print(f"Cadena de entrada: '{INPUT_STRING}'\n")

    try:
        # Inicializar y ejecutar el análisis
        analyzer = LR1Analyzer(GRAMMAR_INPUT)
        results = analyzer.analyze(INPUT_STRING)

        # --- PARTE 2: IMPRIMIR RESULTADOS ---

        print("\n--- 3. TABLA ACTION ---")
        term_cols = sorted([t for t in results["terminals"] if t != EOF and t != EMPTY]) + [EOF]
        header = ["Estado"] + term_cols
        print(f"| {' | '.join(f'{h:^8}' for h in header)} |")
        print(f"|{'---------|' * len(header)}")

        for i in range(results["num_states"]):
            row = [f"I{i:<7}"]
            for t in term_cols:
                action = results["action_table"].get(i, {}).get(t, "")
                row.append(f"{action:^8}")
            print(f"| {' | '.join(row)} |")

        print("\n--- 4. TABLA GOTO ---")
        nt_cols = sorted([nt for nt in results["non_terminals"] if nt != analyzer.GRAMMAR[0][0]])
        header = ["Estado"] + nt_cols
        print(f"| {' | '.join(f'{h:^15}' for h in header)} |")
        print(f"|{'-----------------|' * len(header)}")

        for i in range(results["num_states"]):
            row = [f"I{i:<14}"]
            for nt in nt_cols:
                goto_state = results["goto_table"].get(i, {}).get(nt, "")
                row.append(f"{str(goto_state):^15}")
            print(f"| {' | '.join(row)} |")

        print("\n--- 5. TRAZA DEL PARSING ---")
        # Find max widths for alignment
        max_stack = max(len(s['stack']) for s in results['trace'])
        max_input = max(len(s['input']) for s in results['trace'])

        print(f"{'Step':<4} | {'Pila':<{max_stack}} | {'Entrada':<{max_input}} | Acción")
        print(f"{'-' * 4}-+-{'-' * max_stack}-+-{'-' * max_input}-+-{'-' * 20}")
        for i, step in enumerate(results["trace"]):
            print(f"{i + 1:<4} | {step['stack']:<{max_stack}} | {step['input']:<{max_input}} | {step['action']}")

        # Resultado final
        final_action = results["trace"][-1]['action']
        if "ACEPTADO" in final_action:
            print("\n--- 6. ÁRBOL DE SINTÁXIS ABSTRACTA (AST) ---")
            if results["ast"]:
                print(results["ast"])
            else:
                print("No se pudo construir el árbol.")
            print("\n¡Análisis sintáctico finalizado con éxito! ")
        else:
            print("\n¡Error de análisis sintáctico!")

    except ValueError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nERROR Inesperado durante el análisis: {e}")