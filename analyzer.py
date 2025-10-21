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
    """
    Clase para manejar todo el proceso de análisis LR(1) para una gramática dada.
    """

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
                if p == "''":
                    prod_symbols = [EMPTY]
                else:
                    prod_symbols = p.split()
                rules.append((non_terminal, prod_symbols))
                # acumulamos símbolos de la producción (posibles terminales/no-terminales)
                all_symbols.update(prod_symbols)

        if not rules:
            raise ValueError("La gramática no puede estar vacía.")

        self.START_SYMBOL = rules[0][0]
        start_prime = self.START_SYMBOL + "'"
        # primer elemento: S' -> S
        self.GRAMMAR = [(start_prime, [self.START_SYMBOL])] + rules

        # no-terminals: los explicitados + S'
        nts = set(non_terminals)
        nts.add(start_prime)
        self.NON_TERMINALS = sorted(list(nts))

        # terminals: todos los símbolos que aparecen en RHS menos los non-terminals
        terms = set(all_symbols) - set(self.NON_TERMINALS)
        if EMPTY in terms:
            terms.remove(EMPTY)
        terms.add(EOF)
        self.TERMINALS = sorted(list(terms))

    def _initialize_grammar(self):
        self._parse_grammar_string()

    def _calculate_first_sets(self):
        """Calcula el conjunto FIRST para todos los símbolos."""
        for t in self.TERMINALS: self.FIRST[t].add(t)
        self.FIRST[EMPTY].add(EMPTY)
        changed = True
        while changed:
            changed = False
            for A, production in self.GRAMMAR:
                if production == [EMPTY]:
                    if EMPTY not in self.FIRST[A]:
                        self.FIRST[A].add(EMPTY)
                        changed = True
                    continue

                all_epsilon = True
                for X in production:
                    if X not in self.FIRST: continue
                    old_size = len(self.FIRST[A])
                    self.FIRST[A].update(self.FIRST[X] - {EMPTY})
                    if len(self.FIRST[A]) != old_size: changed = True
                    if EMPTY not in self.FIRST[X]:
                        all_epsilon = False
                        break
                if all_epsilon:
                    if EMPTY not in self.FIRST[A]:
                        self.FIRST[A].add(EMPTY)
                        changed = True

    class LR1Item:
        """Representa un ítem LR(1)."""

        def __init__(self, rule_index, dot_position, lookahead, analyzer):
            self.rule_index = rule_index
            self.dot_position = dot_position
            self.lookahead = lookahead
            self.analyzer = analyzer
            self.non_terminal = self.analyzer.GRAMMAR[rule_index][0]
            self.production = self.analyzer.GRAMMAR[rule_index][1]

        def is_final(self): return self.dot_position >= len(self.production) or self.production == [EMPTY]

        def next_symbol(self): return None if self.is_final() else self.production[self.dot_position]

        def to_string(self):
            prod_display = self.production[:]
            if prod_display == [EMPTY]: prod_display = ["ε"]
            prod_display.insert(self.dot_position, "•")
            return f"[{self.non_terminal} -> {' '.join(prod_display)}, {self.lookahead}]"

        def __eq__(self,
                   other): return self.rule_index == other.rule_index and self.dot_position == other.dot_position and self.lookahead == other.lookahead

        def __hash__(self): return hash((self.rule_index, self.dot_position, self.lookahead))

    def _closure(self, I):
        """Calcula el cierre (closure) del conjunto de ítems I."""
        J = set(I)
        worklist = list(I)
        while worklist:
            item = worklist.pop(0)
            X = item.next_symbol()
            if X is None or X in self.TERMINALS: continue
            beta_symbols = item.production[item.dot_position + 1:]
            first_beta_a = set()
            epsilon_in_beta = True
            for B in beta_symbols:
                first_beta_a.update(self.FIRST[B] - {EMPTY})
                if EMPTY not in self.FIRST[B]:
                    epsilon_in_beta = False
                    break
            if epsilon_in_beta: first_beta_a.add(item.lookahead)
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
                J.add(self.LR1Item(item.rule_index, item.dot_position + 1, item.lookahead, self))
        return self._closure(J)

    def _build_canonical_collection(self):
        """Construye la colección canónica C y la tabla de transiciones GOTO."""
        initial_item = self.LR1Item(0, 0, EOF, self)
        I0 = self._closure({initial_item})
        C = [I0]
        states_map = {frozenset(I0): 0}
        goto_map = {}
        symbols = self.NON_TERMINALS + self.TERMINALS
        queue = collections.deque([0])
        while queue:
            i = queue.popleft()
            I = C[i]
            for X in symbols:
                if X == EOF or X == EMPTY: continue
                J = self._goto(I, X)
                if not J: continue
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
            for nt in self.NON_TERMINALS:
                if nt != self.GRAMMAR[0][0] and (i, nt) in goto_map:
                    goto_output_table[i][nt] = goto_map[(i, nt)]
            for item in I:
                t = item.next_symbol()
                if t in self.TERMINALS and (i, t) in goto_map:
                    j = goto_map[(i, t)]
                    if t in action_table[i] and action_table[i][t] != f"s{j}":
                        raise ValueError(f"Conflicto Shift/Reduce en estado {i} con terminal '{t}'")
                    action_table[i][t] = f"s{j}"
                elif item.is_final():
                    a = item.lookahead
                    if item.rule_index == 0:
                        action_table[i][EOF] = "acc"
                    else:
                        r_index = item.rule_index
                        if a in action_table[i] and action_table[i][a] != f"r{r_index}":
                            raise ValueError(
                                f"Conflicto en estado {i} con terminal '{a}': {action_table[i][a]} vs r{r_index}. La gramática no es LR(1).")
                        action_table[i][a] = f"r{r_index}"
        return action_table, goto_output_table

    def _parse_input(self, input_tokens, action_table, goto_table):
        """Simula el proceso de parsing y construye el AST."""
        input_tokens.append(EOF)
        stack = [0]
        value_stack = []
        pointer = 0
        trace = []
        while True:
            current_state = stack[-1]
            current_token = input_tokens[pointer]
            action = action_table.get(current_state, {}).get(current_token)
            trace.append({"stack": ' '.join(map(str, stack)), "input": ' '.join(input_tokens[pointer:]),
                          "action": action if action else "ERROR"})
            if action is None: break
            if action.startswith('s'):
                next_state = int(action[1:])
                stack.append(current_token);
                stack.append(next_state)
                value_stack.append(ASTNode(current_token))
                pointer += 1
            elif action.startswith('r'):
                rule_index = int(action[1:])
                nt, prod = self.GRAMMAR[rule_index]
                prod_len = len(prod) if prod != [EMPTY] else 0
                children = []
                for _ in range(prod_len):
                    stack.pop();
                    stack.pop()
                    if value_stack: children.insert(0, value_stack.pop())
                new_node = ASTNode(nt, children)
                value_stack.append(new_node)
                prev_state = stack[-1]
                next_state = goto_table.get(prev_state, {}).get(nt)
                if next_state is None:
                    trace.append({"stack": ' '.join(map(str, stack)), "input": ' '.join(input_tokens[pointer:]),
                                  "action": f"ERROR: GOTO no definido"})
                    break
                stack.append(nt);
                stack.append(next_state)
                trace[-1]["action"] += f" ({nt} -> {' '.join(prod)})"
            elif action == 'acc':
                if value_stack: self.ast = value_stack[0]
                break
            else:
                break
        return trace

    def analyze(self, input_string: str):
        """Ejecuta el análisis LR(1) completo y retorna un diccionario de resultados."""
        tokens = re.findall(r'\"[^"]*\"|\b\w+\b|[:\[\]\{\},:()=+\-*/]', input_string)
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
