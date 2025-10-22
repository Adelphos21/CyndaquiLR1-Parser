# -*- coding: utf-8 -*-
"""
Módulo del Generador de Tablas y Parser LR(1) en Python.
"""
import collections
import re

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
                # En esta implementación, todos los hijos son ASTNode
                result += f"{indent}  {child.symbol}\n"
        return result


class LR1Analyzer:
    def __init__(self, grammar_str):
        self.grammar_str = grammar_str
        self.GRAMMAR = []
        self.NON_TERMINALS = set()
        self.TERMINALS = set()
        self.START_SYMBOL = None
        self.FIRST = collections.defaultdict(set)
        self.ast = None
        self._initialize_grammar()
        self._calculate_first_sets()

    def _parse_grammar_string(self):
        rules = []
        non_terminals = set()
        all_symbols = set()

        lines = [ln.strip() for ln in self.grammar_str.strip().splitlines() if ln.strip()]
        for line in lines:
            if '->' not in line:
                raise ValueError(f"Formato de regla inválido: '{line}'. Use 'A -> B C | d'")
            nt_part, prods_part = line.split('->', 1)
            non_terminal = nt_part.strip()
            non_terminals.add(non_terminal)
            prods = [p.strip() for p in prods_part.split('|')]

            for p in prods:
                if p == "''" or p == "ε":
                    # Producción épsilon - representada como lista vacía
                    prod_symbols = []
                else:
                    prod_symbols = p.split()

                rules.append((non_terminal, prod_symbols))
                # Solo agregar símbolos no vacíos a all_symbols
                all_symbols.update(prod_symbols)

        if not rules:
            raise ValueError("La gramática no puede estar vacía.")

        self.START_SYMBOL = rules[0][0]
        start_prime = self.START_SYMBOL + "'"

        # Gramática aumentada: S' -> S
        self.GRAMMAR = [(start_prime, [self.START_SYMBOL])] + rules

        # No-terminales: los explicitados + S'
        nts = set(non_terminals)
        nts.add(start_prime)
        self.NON_TERMINALS = sorted(list(nts))

        # Terminales: todos los símbolos que aparecen en RHS menos los non-terminals
        terms = set(all_symbols) - set(self.NON_TERMINALS)
        terms.add(EOF)  # Agregar EOF como terminal
        self.TERMINALS = sorted(list(terms))

    def _calculate_first_sets(self):
        """Calcula el conjunto FIRST para todos los símbolos."""
        # Inicializar FIRST para terminales
        for t in self.TERMINALS:
            self.FIRST[t].add(t)

        # FIRST para épsilon es el conjunto vacío (no es un símbolo terminal)

        changed = True
        while changed:
            changed = False
            for A, production in self.GRAMMAR:
                # Para producciones vacías, agregar épsilon a FIRST(A)
                if not production:
                    if EMPTY not in self.FIRST[A]:
                        self.FIRST[A].add(EMPTY)
                        changed = True
                    continue

                # Para producciones no vacías
                all_epsilon = True
                for X in production:
                    old_size = len(self.FIRST[A])
                    self.FIRST[A].update(self.FIRST[X] - {EMPTY})
                    if len(self.FIRST[A]) != old_size:
                        changed = True

                    if EMPTY not in self.FIRST[X]:
                        all_epsilon = False
                        break

                if all_epsilon:
                    if EMPTY not in self.FIRST[A]:
                        self.FIRST[A].add(EMPTY)
                        changed = True

    class LR1Item:
        def __init__(self, rule_index, dot_position, lookahead, analyzer):
            self.rule_index = rule_index
            self.dot_position = dot_position
            self.lookahead = lookahead
            self.analyzer = analyzer
            self.non_terminal = self.analyzer.GRAMMAR[rule_index][0]
            self.production = self.analyzer.GRAMMAR[rule_index][1]

        def is_final(self):
            return self.dot_position >= len(self.production)

        def next_symbol(self):
            if self.is_final():
                return None
            return self.production[self.dot_position]

        def to_string(self):
            if not self.production:  # Producción épsilon
                return f"[{self.non_terminal} -> •ε, {self.lookahead}]"

            prod_display = self.production.copy()
            prod_display.insert(self.dot_position, "•")
            return f"[{self.non_terminal} -> {' '.join(prod_display)}, {self.lookahead}]"

        def __eq__(self, other):
            return (self.rule_index == other.rule_index and
                    self.dot_position == other.dot_position and
                    self.lookahead == other.lookahead)

        def __hash__(self):
            return hash((self.rule_index, self.dot_position, self.lookahead))

    def _closure(self, I):
        J = set(I)
        worklist = list(I)

        while worklist:
            item = worklist.pop(0)
            X = item.next_symbol()

            if X is None or X in self.TERMINALS:
                continue

            # Calcular FIRST(βa)
            beta = item.production[item.dot_position + 1:]
            first_beta_a = set()

            # Si β es vacío, FIRST(βa) = {a}
            if not beta:
                first_beta_a.add(item.lookahead)
            else:
                # Calcular FIRST(β)
                all_epsilon = True
                for symbol in beta:
                    first_beta_a.update(self.FIRST[symbol] - {EMPTY})
                    if EMPTY not in self.FIRST[symbol]:
                        all_epsilon = False
                        break
                if all_epsilon:
                    first_beta_a.add(item.lookahead)

            # Agregar items para cada producción de X
            for rule_index, (nt, production) in enumerate(self.GRAMMAR):
                if nt == X:
                    for b in first_beta_a:
                        new_item = self.LR1Item(rule_index, 0, b, self)
                        if new_item not in J:
                            J.add(new_item)
                            worklist.append(new_item)
        return J

    def _parse_input(self, input_tokens, action_table, goto_table):
        input_tokens.append(EOF)
        stack = [0]
        value_stack = []
        pointer = 0
        trace = []

        while True:
            current_state = stack[-1]
            current_token = input_tokens[pointer]
            action = action_table.get(current_state, {}).get(current_token)

            trace.append({
                "stack": ' '.join(map(str, stack)),
                "input": ' '.join(input_tokens[pointer:]),
                "action": action if action else "ERROR"
            })

            if action is None:
                break

            if action.startswith('s'):
                next_state = int(action[1:])
                stack.append(current_token)
                stack.append(next_state)
                value_stack.append(ASTNode(current_token))
                pointer += 1

            elif action.startswith('r'):
                rule_index = int(action[1:])
                nt, prod = self.GRAMMAR[rule_index]

                # Manejar producciones épsilon
                if not prod:  # Producción vacía = épsilon
                    prod_len = 0
                    children = []
                else:
                    prod_len = len(prod)
                    children = []
                    for _ in range(prod_len):
                        stack.pop()
                        stack.pop()
                        if value_stack:
                            children.insert(0, value_stack.pop())

                new_node = ASTNode(nt, children)
                value_stack.append(new_node)

                prev_state = stack[-1]
                next_state = goto_table.get(prev_state, {}).get(nt)

                if next_state is None:
                    trace.append({
                        "stack": ' '.join(map(str, stack)),
                        "input": ' '.join(input_tokens[pointer:]),
                        "action": f"ERROR: GOTO no definido para ({prev_state}, {nt})"
                    })
                    break

                stack.append(nt)
                stack.append(next_state)

                # Para mostrar en el trace
                prod_str = "ε" if not prod else ' '.join(prod)
                trace[-1]["action"] += f" ({nt} -> {prod_str})"

            elif action == 'acc':
                if value_stack:
                    self.ast = value_stack[0]
                break
            else:
                break

        return trace

    def analyze(self, input_string: str):
        """Ejecuta el análisis LR(1) completo y retorna un diccionario de resultados."""
        tokens = re.findall(r'\"[^"]*\"|\b\w+\b|[:\[\]\{\},:()=+\-*;/]', input_string)
        C, GOTO_MAP = self._build_canonical_collection()
        ACTION_TABLE, GOTO_TABLE = self._generate_parsing_tables(C, GOTO_MAP)
        parse_trace = self._parse_input(tokens, ACTION_TABLE, GOTO_TABLE)
        formatted_grammar = [f"({i}) {nt} -> {' '.join(prod)}" for i, (nt, prod) in enumerate(self.GRAMMAR)]
        formatted_states = {
            f"I{i}": [item.to_string() for item in sorted(list(I), key=lambda x: (x.rule_index, x.dot_position))] for
            i, I in enumerate(C)}
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


def serialize_ast_to_graph(node: ASTNode):
    """
    Convierte un árbol AST en un formato de grafo (nodos y aristas)
    que es serializable a JSON, ideal para visualización.
    """
    nodes = []
    edges = []
    _node_counter = 0

    def traverse(current_node, parent_id=None):
        nonlocal _node_counter
        node_id = _node_counter
        _node_counter += 1
        nodes.append({"id": node_id, "label": str(current_node.symbol)})
        if parent_id is not None:
            edges.append({"from": parent_id, "to": node_id})
        for child in current_node.children:
            if isinstance(child, ASTNode):
                traverse(child, node_id)

    if node:
        traverse(node)
    return {"nodes": nodes, "edges": edges}
