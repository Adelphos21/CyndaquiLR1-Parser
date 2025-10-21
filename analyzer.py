# -*- coding: utf-8 -*-
"""
Generador de Tablas y Parser LR(1) en Python.

La lógica está encapsulada en la clase LR1Analyzer, permitiendo 
que la gramática y la cadena de entrada sean proporcionadas por el usuario 
(o un frontend), haciendo el código más modular y reutilizable.
"""
import collections
import sys

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
                result += f"{indent}  {child}\n"
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
                    if prod == [EMPTY]:
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
        self.NON_TERMINALS = sorted(list(self.NON_TERMINALS), reverse=True) # E' primero
        self.TERMINALS = sorted(list(self.TERMINALS))

    def _initialize_grammar(self):
        """Inicializa los sets y la gramática aumentada."""
        self._parse_grammar_string()
        
    def _calculate_first_sets(self):
        """Calcula el conjunto FIRST para todos los símbolos."""
        
        # Inicialización con terminales
        for t in self.TERMINALS:
            self.FIRST[t].add(t)

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
                    if X not in self.FIRST: # Símbolo no reconocido, podría ser un error de gramática
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
            self.non_terminal = analyzer.GRAMMAR[rule_index][0]
            self.production = analyzer.GRAMMAR[rule_index][1]

        def is_final(self):
            return self.dot_position >= len(self.production) or self.production == [EMPTY]

        def next_symbol(self):
            if self.is_final():
                return None
            return self.production[self.dot_position]

        def to_string(self):
            prod = self.production[:]
            if prod != [EMPTY]:
                prod.insert(self.dot_position, "•")
            return f"[{self.non_terminal} -> {' '.join(prod)}, {self.lookahead}]"

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
        changed = True
        while changed:
            changed = False
            new_items = set()

            for item in J:
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
                                new_items.add(new_item)
                                changed = True
            
            J.update(new_items)
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
        goto_map = {}  # { (state_index, symbol): new_state_index }
        
        # Símbolos de transición: todos los Terminales (excepto $) y No-Terminales
        symbols = [s for s in self.NON_TERMINALS if s != self.GRAMMAR[0][0]] + [s for s in self.TERMINALS if s != EOF]
        
        queue = collections.deque([I0])
        
        while queue:
            I = queue.popleft()
            i = C.index(I)

            for X in symbols:
                J = self._goto(I, X)
                
                if not J:
                    continue

                if J not in C:
                    C.append(J)
                    queue.append(J)
                
                j = C.index(J)
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
            potential_actions = collections.defaultdict(list) # {terminal: [acciones]}

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
                    
                    if item.rule_index == 0:
                        # Aceptación
                        potential_actions[EOF].append("acc")
                    else:
                        # Reducción
                        r_index = item.rule_index
                        potential_actions[a].append(f"r{r_index}")

            # Procesar acciones potenciales y detectar conflictos
            for terminal, actions in potential_actions.items():
                # Eliminar acciones duplicadas (varios ítems causan el mismo Shift)
                unique_actions = sorted(list(set(actions))) 

                if len(unique_actions) > 1:
                    # CONFLICTO VERDADERO: S/R o R/R
                    conflict_type = "S/R" if any(a.startswith('s') for a in unique_actions) and any(a.startswith('r') for a in unique_actions) else "R/R"
                    
                    print(f"¡ADVERTENCIA DE CONFLICTO! Estado {i}, Terminal '{terminal}'. Tipo: {conflict_type}. Acciones: {', '.join(unique_actions)}")
                    print(f"   --> Seleccionando la primera acción: {unique_actions[0]}")
                    action_table[i][terminal] = unique_actions[0]
                    
                elif len(unique_actions) == 1:
                    # Acción única (Shift, Reduce, o Accept)
                    action_table[i][terminal] = unique_actions[0]
                
                    
        return action_table, goto_output_table

    # --- Driver de Parsing ---

    def _parse_input(self, input_string, action_table, goto_table):
        """Simula el proceso de parsing con las tablas generadas y construye el AST."""
        input_tokens = input_string.split() + [EOF]
        stack = [0]
        value_stack = []  # Pila para construir el AST
        pointer = 0
        trace = []

        while True:
            current_state = stack[-1]
            current_token = input_tokens[pointer]
            action = action_table.get(current_state, {}).get(current_token)

            stack_str = " ".join(map(str, stack))
            input_str = " ".join(input_tokens[pointer:])

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
                # Shift (Desplazamiento) - Crear nodo hoja para terminal
                next_state = int(action[1:])
                stack.append(current_token)
                stack.append(next_state)

                # Apilar nodo hoja en value_stack
                value_stack.append(ASTNode(current_token))
                pointer += 1

            elif action.startswith('r'):
                # Reduce (Reducción) - Construir nodo del AST
                rule_index = int(action[1:])
                nt, prod = self.GRAMMAR[rule_index]
                prod_len = len(prod)

                # Ajuste para producciones epsilon
                if prod_len == 1 and prod[0] == EMPTY:
                    prod_len = 0
                    # Para epsilon, crear nodo con hijos vacíos
                    new_node = ASTNode(nt)
                else:
                    # Extraer hijos de la pila de valores
                    children = []
                    for _ in range(prod_len):
                        # Sacar 2 elementos de la pila principal por cada símbolo
                        stack.pop()  # estado
                        stack.pop()  # símbolo
                        # Sacar nodo de value_stack
                        if value_stack:
                            children.insert(0, value_stack.pop())

                    # Crear nuevo nodo con los hijos
                    new_node = ASTNode(nt, children)

                # Apilar el nuevo nodo
                value_stack.append(new_node)

                # Actualizar pila principal con el no terminal
                prev_state = stack[-1]
                next_state = goto_table.get(prev_state, {}).get(nt)
                if next_state is None:
                    trace.append({
                        "stack": " ".join(map(str, stack)),
                        "input": input_str,
                        "action": f"ERROR: No hay GOTO definido en estado {prev_state} con no-terminal '{nt}'."
                    })
                    break

                stack.append(nt)
                stack.append(next_state)
                trace[-1]["action"] += f" ({nt} -> {' '.join(prod)})"

            elif action == 'acc':
                trace[-1]["action"] = "ACEPTADO"
                # El AST final está en value_stack[0]
                if value_stack:
                    self.ast = value_stack[0]
                break

            else:
                break  # Error desconocido

        return trace

    # Modifica el método analyze para incluir el AST en los resultados
    def analyze(self, input_string):
        """Ejecuta el análisis LR(1) completo y retorna un diccionario de resultados."""

        # 1. Construir la Colección Canónica
        C, GOTO_MAP = self._build_canonical_collection()

        # 2. Generar Tablas de Análisis
        ACTION_TABLE, GOTO_TABLE = self._generate_parsing_tables(C, GOTO_MAP)

        # 3. Demostración de Parsing
        parse_trace = self._parse_input(input_string, ACTION_TABLE, GOTO_TABLE)

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
            "ast": getattr(self, 'ast', None)  # Incluir el AST en los resultados
        }

# --- 8. Ejecución Principal con Input de Usuario ---
    
    # --- PARTE 1: DEFINICIÓN DE LA GRAMÁTICA Y CADENA DE ENTRADA ---
    
    # Define la gramática usando el formato: NT -> Prod1 | Prod2
    # Cada regla principal debe ir en una nueva línea.
    
    # Gramática de ejemplo (para expresiones aritméticas)
    GRAMMAR_INPUT = """
S -> C C
C -> c C
C -> d
    """

    # Cadena de tokens a analizar (separados por espacio)
    INPUT_STRING = "c d d"

    print("--- PARSER LR(1) CON INPUT DINÁMICO ---")
    print(f"Gramática a analizar:\n{GRAMMAR_INPUT.strip()}")
    print(f"Cadena de entrada: '{INPUT_STRING}'\n")

    try:
        # Inicializar y ejecutar el análisis
        analyzer = LR1Analyzer(GRAMMAR_INPUT)
        results = analyzer.analyze(INPUT_STRING)

        # --- PARTE 2: IMPRIMIR RESULTADOS ---

        print("--- 1. GRAMÁTICA AUMENTADA Y REGLAS ---")
        for rule in results["grammar"]:
            print(rule)
        
        print(f"\nTerminales: {', '.join(results['terminals'])}")
        print(f"No-Terminales: {', '.join(results['non_terminals'])}")
        print(f"Total de Estados: {results['num_states']}")

        print("\n--- 2. COLECCIÓN CANÓNICA (ESTADOS) ---")
        for state_name, items in results["states"].items():
            print(f"\n{state_name}:")
            for item in items:
                print(f"  {item}")

        print("\n--- 3. TABLA ACTION ---")
        term_cols = [t for t in results["terminals"] if t != EOF] + [EOF]
        header = ["Estado"] + term_cols
        print("| " + " | ".join(header) + " |")
        print("|---" * (len(header)) + "|")
        
        for i in range(results["num_states"]):
            row = [f"I{i}"]
            for t in term_cols:
                action = results["action_table"].get(i, {}).get(t, "")
                row.append(action)
            print("| " + " | ".join(row) + " |")
            
        print("\n--- 4. TABLA GOTO ---")
        nt_cols = [nt for nt in results["non_terminals"] if nt != analyzer.GRAMMAR[0][0]]
        header = ["Estado"] + nt_cols
        print("| " + " | ".join(header) + " |")
        print("|---" * (len(header)) + "|")
        
        for i in range(results["num_states"]):
            row = [f"I{i}"]
            for nt in nt_cols:
                goto_state = results["goto_table"].get(i, {}).get(nt, "")
                row.append(str(goto_state))
            print("| " + " | ".join(row) + " |")

        print("\n--- 5. TRAZA DEL PARSING ---")
        print("Step |Pila | Entrada | Acción")
        print("--- |--- | --- | ---")
        i = 1
        for step in results["trace"]:
            print(f"{i}|{step['stack']} | {step['input']} | {step['action']}")
            i+=1

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

def serialize_ast_to_graph(root_node: ASTNode):
    """
    Recorre el AST (generado por ASTNode) y lo convierte en un formato de grafo
    (lista de nodos y lista de aristas)
    """
    nodes = []
    edges = []
    _counter = 0

    def traverse(node, parent_id=None):
        nonlocal _counter
        current_id = _counter
        _counter += 1

        # Formatear la etiqueta del nodo
        node_label = node.symbol
        if node.value:
            # \n es interpretado por muchas librerías de grafos como un salto de línea
            node_label += f"\n({node.value})"
        
        # Añadir el nodo a la lista
        nodes.append({
            "id": current_id, 
            "label": node_label
        })

        # Si no es la raíz, crear un enlace desde su padre
        if parent_id is not None:
            edges.append({
                "from": parent_id, 
                "to": current_id
            })
        
        # Recorrer los hijos
        for child in node.children:
            if isinstance(child, ASTNode):
                traverse(child, current_id)
    
    # Iniciar el recorrido desde el nodo raíz
    if root_node:
        traverse(root_node)
        
    return {"nodes": nodes, "edges": edges}
