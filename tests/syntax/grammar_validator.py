"""
Grammar Validator for Behavior Trees

This module provides validation of behavior trees against custom formal grammars.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Any, Optional
import itertools


class BehaviorTreeGrammarValidator:
    """
    Validates behavior trees against a dynamically parsed formal grammar.
    
    The validator parses grammar rules, generates a set of valid base shapes,
    and compares an XML tree's structure against these shapes.
    """
    
    def __init__(self, grammar_rules: Dict[str, Any]):
        """
        Initialize the validator with grammar rules.
        
        Args:
            grammar_rules: Dictionary defining the raw grammar rules.
        """
        self.raw_grammar_rules = grammar_rules
        self.parsed_rules = self._parse_grammar_to_validation_rules(self.raw_grammar_rules)
        self.flattened_grammar_shapes = self._flatten_grammar_shapes(self.parsed_rules)

    # --- Grammar Parsing Methods (from grammar_parser_v3.py) ---

    def _parse_grammar_to_validation_rules(self, grammar_rules: Dict[str, Any]) -> Dict[str, Any]:
        parsed_rules = {}
        terminal_nodes = {}
        for symbol, rule in grammar_rules.items():
            if (symbol.islower() and 
                isinstance(rule, list) and 
                len(rule) == 2 and 
                isinstance(rule[1], list) and 
                rule[1] == ["children_nodes"]):
                terminal_nodes[symbol] = rule[0]
        
        for terminal_symbol, node_type in terminal_nodes.items():
            non_terminal_rules = []
            for symbol, rules in grammar_rules.items():
                if symbol.isupper():
                    for rule in rules:
                        if (isinstance(rule, list) and 
                            len(rule) >= 2 and 
                            rule[0] == terminal_symbol):
                            non_terminal_rules.append(rule[1])
            if not non_terminal_rules: continue
            
            combinations, amounts = [], {}
            for rule_children in non_terminal_rules:
                combination = self._process_rule_children(rule_children, grammar_rules, amounts)
                if combination:
                    combinations.append(combination)
            
            parsed_rules[node_type] = {"Combinations": combinations, "Amounts": amounts}
        return parsed_rules

    def _process_rule_children(self, children_symbols: List[str], grammar_rules: Dict[str, Any], amounts: Dict[str, int]) -> List[Any]:
        combination = []
        for symbol in children_symbols:
            if symbol.endswith('n'):
                if symbol in grammar_rules and len(grammar_rules[symbol]) > 1:
                    can_be_empty = any(rule == [] for rule in grammar_rules.get(symbol, []))
                    min_amount = 0 if can_be_empty else 1
                    single_rep = grammar_rules[symbol][1]
                    options = self._expand_symbol_fully_from_rule(single_rep, grammar_rules)
                    combination.append(options)
                    self._mark_unlimited_amounts(options, amounts, min_amount)
                else:
                    combination.append([[symbol]])
            else:
                options = self._expand_symbol_fully(symbol, grammar_rules)
                combination.append(options)
        return combination

    def _expand_symbol_fully_from_rule(self, rule: Any, grammar_rules: Dict[str, Any]) -> List[List[str]]:
        if not isinstance(rule, list): return [[str(rule)]]
        expanded_rule = []
        for symbol in rule:
            if isinstance(symbol, str):
                if symbol in grammar_rules:
                    if symbol.islower():
                        terminal_rule = grammar_rules[symbol]
                        expanded_rule.append(terminal_rule[0] if len(terminal_rule) >= 1 else symbol)
                    else:
                        sub_options = self._expand_symbol_fully(symbol, grammar_rules)
                        expanded_rule.append(sub_options[0][0] if len(sub_options) == 1 and len(sub_options[0]) == 1 else sub_options)
                else:
                    expanded_rule.append(symbol)
        return [expanded_rule]

    def _expand_symbol_fully(self, symbol: str, grammar_rules: Dict[str, Any], visited: Optional[set] = None) -> List[List[str]]:
        if visited is None: visited = set()
        if symbol in visited: return [[f"RECURSIVE:{symbol}"]]
        if symbol not in grammar_rules: return [[symbol]]

        if symbol.islower():
            terminal_rule = grammar_rules[symbol]
            if len(terminal_rule) == 1: return [[terminal_rule[0]]]
            if len(terminal_rule) == 2 and terminal_rule[1] == ["children_nodes"]: return [[terminal_rule[0]]]

        visited.add(symbol)
        rules, options = grammar_rules[symbol], []
        for rule in rules:
            if isinstance(rule, list):
                expanded_rule = []
                for sub_symbol in rule:
                    if isinstance(sub_symbol, str):
                        if sub_symbol in grammar_rules:
                            if sub_symbol.islower():
                                terminal_rule = grammar_rules[sub_symbol]
                                if len(terminal_rule) >= 1: expanded_rule.append(terminal_rule[0])
                                else: expanded_rule.append(sub_symbol)
                            else:
                                expanded_rule.append(self._expand_symbol_fully(sub_symbol, grammar_rules, visited.copy()))
                        else:
                            expanded_rule.append(sub_symbol)
                options.append(expanded_rule)
            elif isinstance(rule, str):
                if rule in grammar_rules and rule.islower():
                    terminal_rule = grammar_rules[rule]
                    if len(terminal_rule) >= 1: options.append([terminal_rule[0]])
                    else: options.append([rule])
                else:
                    options.append([rule])
            elif rule == []:
                options.append([])
        visited.remove(symbol)
        
        if all(len(opt) == 1 and isinstance(opt[0], str) for opt in options):
            unique_options, seen = [], set()
            for opt in options:
                if opt[0] not in seen:
                    unique_options.append(opt)
                    seen.add(opt[0])
            return unique_options
        return options

    def _mark_unlimited_amounts(self, options: List[List[str]], amounts: Dict[str, int], min_amount: int):
        for option_list in options:
            for option in option_list:
                if isinstance(option, str) and not option.startswith("RECURSIVE:"):
                    amounts[option] = min_amount

    # --- Shape Flattening & Extraction Methods (from grammar_parser_v3.py) ---

    def _flatten_grammar_shapes(self, parsed_rules: Dict[str, Any]) -> Dict[str, List[List[str]]]:
        flattened_shapes = {}
        for node_type, rules in parsed_rules.items():
            all_node_type_shapes = []
            for combination in rules["Combinations"]:
                shapes_from_combo = self._generate_flattened_shapes_from_combination(combination)
                for shape in shapes_from_combo:
                    if shape not in all_node_type_shapes:
                        all_node_type_shapes.append(shape)
            flattened_shapes[node_type] = all_node_type_shapes
        return flattened_shapes

    def _generate_flattened_shapes_from_combination(self, combination: List[Any]) -> List[List[str]]:
        all_choice_paths = list(itertools.product(*combination))
        all_flattened_shapes = []
        for path in all_choice_paths:
            flattened_shape = []
            for choice in path:
                flattened_shape.extend(choice)
            if flattened_shape not in all_flattened_shapes:
                all_flattened_shapes.append(flattened_shape)
        return all_flattened_shapes

    def _extract_tree_shapes(self, xml_string: str) -> Dict[str, Any]:
        try:
            root = ET.fromstring(xml_string.strip())
            shapes, node_counters = {}, {}
            
            def remove_consecutive_duplicates(child_types: List[str], parent_node_type: str) -> List[str]:
                if not child_types: return []
                unlimited_nodes = set(self.parsed_rules.get(parent_node_type, {}).get("Amounts", {}).keys())
                result = [child_types[0]]
                for i in range(1, len(child_types)):
                    if child_types[i] != child_types[i-1] or child_types[i] not in unlimited_nodes:
                        result.append(child_types[i])
                return result
            
            def extract_node_shape(node):
                node_type = node.tag
                children = [child for child in node if child.tag is not None]
                if children:
                    child_types = [child.tag for child in children]
                    collapsed_child_types = remove_consecutive_duplicates(child_types, node_type)
                    
                    node_counters[node_type] = node_counters.get(node_type, 0) + 1
                    instance_key = f"{node_type}{node_counters[node_type]}"

                    if node_type not in shapes: shapes[node_type] = {}
                    shapes[node_type][instance_key] = collapsed_child_types
                    
                    for child in children: extract_node_shape(child)
            
            extract_node_shape(root)
            return shapes
        except ET.ParseError as e: return {"error": f"XML parsing error: {e}"}
        except Exception as e: return {"error": f"Unexpected error: {e}"}

    # --- Public Validation Method ---

    def validate_tree(self, tree_xml: str) -> Tuple[bool, str]:
        tree_shapes = self._extract_tree_shapes(tree_xml)
        if "error" in tree_shapes:
            return False, f"Tree parsing error: {tree_shapes['error']}"
        
        errors = []
        for node_type, instances in tree_shapes.items():
            if node_type not in self.flattened_grammar_shapes:
                errors.append(f"Node type '{node_type}' is not defined in the grammar")
                continue
            
            allowed_shapes = self.flattened_grammar_shapes[node_type]
            for instance_id, instance_shape in instances.items():
                if instance_shape not in allowed_shapes:
                    errors.append(
                        f"Node '{instance_id}' has invalid child shape {instance_shape}. "
                        f"Allowed base shapes for '{node_type}': {allowed_shapes}"
                    )
        
        if errors: return False, f"Tree has syntax errors: {'; '.join(errors)}"
        return True, "Tree is syntactically valid according to the grammar."

    def get_grammar_info(self) -> Dict[str, Any]:
        return {"parsed_rules": self.parsed_rules, "flattened_shapes": self.flattened_grammar_shapes}
       