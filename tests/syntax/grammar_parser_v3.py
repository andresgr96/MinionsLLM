import xml.etree.ElementTree as ET

def parse_grammar_to_validation_rules(grammar_rules):
    """
    Parse the grammar rules into a format suitable for validation.
    
    This function converts the current grammar representation into a more usable format
    for validation by extracting allowed combinations and amounts for each node type.
    """
    parsed_rules = {}
    
    # Step 1: Identify terminal symbols that represent nodes with children
    terminal_nodes = {}
    for symbol, rule in grammar_rules.items():
        if (symbol.islower() and 
            isinstance(rule, list) and 
            len(rule) == 2 and 
            isinstance(rule[1], list) and 
            rule[1] == ["children_nodes"]):
            terminal_nodes[symbol] = rule[0]  # symbol -> node_type mapping
    
    # Step 2: For each terminal node, find its non-terminal rules
    for terminal_symbol, node_type in terminal_nodes.items():
        # Find the non-terminal that produces this terminal
        non_terminal_rules = []
        for symbol, rules in grammar_rules.items():
            if symbol.isupper():  # Non-terminal symbols are uppercase
                for rule in rules:
                    if (isinstance(rule, list) and 
                        len(rule) >= 2 and 
                        rule[0] == terminal_symbol):
                        non_terminal_rules.append(rule[1])  # Get the children part
        
        if not non_terminal_rules:
            continue
        
        # Step 3: Process each rule to extract combinations
        combinations = []
        amounts = {}
        
        for rule_children in non_terminal_rules:
            combination = process_rule_children(rule_children, grammar_rules, amounts)
            if combination:
                combinations.append(combination)
        
        parsed_rules[node_type] = {
            "Combinations": combinations,
            "Amounts": amounts
        }
    
    return parsed_rules


def process_rule_children(children_symbols, grammar_rules, amounts):
    """
    Process the children symbols in a rule to extract valid node combinations.
    """
    combination = []
    
    for symbol in children_symbols:
        if isinstance(symbol, str):
            if symbol.endswith('n'):
                # List symbol (e.g., "Pn", "SEQn") - can have multiple instances
                # Use the standard: index 1 always contains the single representation
                if symbol in grammar_rules and len(grammar_rules[symbol]) > 1:
                    # Check if this list symbol's rule contains an empty option, indicating it can be 0.
                    can_be_empty = any(rule == [] for rule in grammar_rules.get(symbol, []))
                    min_amount = 0 if can_be_empty else 1

                    single_representation = grammar_rules[symbol][1]  # Index 1 is the single version
                    # Expand the single representation
                    options = expand_symbol_fully_from_rule(single_representation, grammar_rules)
                    combination.append(options)
                    
                    # Mark all terminal options as unlimited, specifying the minimum amount
                    mark_unlimited_amounts(options, amounts, min_amount)
                else:
                    combination.append([[symbol]])
            else:
                # Regular symbol
                options = expand_symbol_fully(symbol, grammar_rules)
                combination.append(options)
    
    return combination


def expand_symbol_fully_from_rule(rule, grammar_rules):
    """
    Expand a rule (list of symbols) to terminal node types.
    """
    if not isinstance(rule, list):
        return [[str(rule)]]
    
    expanded_rule = []
    for symbol in rule:
        if isinstance(symbol, str):
            if symbol in grammar_rules:
                if symbol.islower():
                    # Terminal symbol - translate directly
                    terminal_rule = grammar_rules[symbol]
                    if len(terminal_rule) == 1:
                        expanded_rule.append(terminal_rule[0])
                    elif len(terminal_rule) == 2 and terminal_rule[1] == ["children_nodes"]:
                        expanded_rule.append(terminal_rule[0])
                    else:
                        expanded_rule.append(symbol)
                else:
                    # Non-terminal - recursively expand and flatten
                    sub_options = expand_symbol_fully(symbol, grammar_rules)
                    # For single representation, we want to flatten the result
                    if len(sub_options) == 1 and len(sub_options[0]) == 1:
                        expanded_rule.append(sub_options[0][0])
                    else:
                        expanded_rule.append(sub_options)
            else:
                expanded_rule.append(symbol)
    
    return [expanded_rule]


def expand_symbol_fully(symbol, grammar_rules, visited=None):
    """
    Fully expand a symbol to its terminal node types.
    """
    if visited is None:
        visited = set()
    
    if symbol in visited:
        return [[f"RECURSIVE:{symbol}"]]
    
    if symbol not in grammar_rules:
        return [[symbol]]
    
    # Direct terminal translation
    if symbol.islower():
        terminal_rule = grammar_rules[symbol]
        if len(terminal_rule) == 1:
            # Leaf terminal like "aa" -> ["ActuatorAction"] or "p" -> ["Condition"]
            return [[terminal_rule[0]]]
        elif len(terminal_rule) == 2 and terminal_rule[1] == ["children_nodes"]:
            # Node terminal like "sel" -> ["Selector", ["children_nodes"]]
            return [[terminal_rule[0]]]
    
    visited.add(symbol)
    rules = grammar_rules[symbol]
    options = []
    
    for rule in rules:
        if isinstance(rule, list):
            # Rule is a list of symbols
            expanded_rule = []
            for sub_symbol in rule:
                if isinstance(sub_symbol, str):
                    if sub_symbol in grammar_rules:
                        if sub_symbol.islower():
                            # Terminal symbol - translate directly
                            terminal_rule = grammar_rules[sub_symbol]
                            if len(terminal_rule) == 1:
                                expanded_rule.append(terminal_rule[0])
                            elif len(terminal_rule) == 2 and terminal_rule[1] == ["children_nodes"]:
                                expanded_rule.append(terminal_rule[0])
                            else:
                                expanded_rule.append(sub_symbol)
                        else:
                            # Non-terminal - recursively expand
                            sub_options = expand_symbol_fully(sub_symbol, grammar_rules, visited.copy())
                            # For non-terminals, we want to include all their options
                            expanded_rule.append(sub_options)
                    else:
                        expanded_rule.append(sub_symbol)
            options.append(expanded_rule)
        elif isinstance(rule, str):
            # Single string rule - check if it's a terminal that needs translation
            if rule in grammar_rules and rule.islower():
                terminal_rule = grammar_rules[rule]
                if len(terminal_rule) == 1:
                    options.append([terminal_rule[0]])
                elif len(terminal_rule) == 2 and terminal_rule[1] == ["children_nodes"]:
                    options.append([terminal_rule[0]])
                else:
                    options.append([rule])
            else:
                options.append([rule])
        elif rule == []:
            # Empty rule (epsilon)
            options.append([])
        else:
            options.append([str(rule)])
    
    visited.remove(symbol)
    
    # Remove duplicates for simple cases like BehaviorTree
    if all(len(option) == 1 and isinstance(option[0], str) for option in options):
        unique_options = []
        seen = set()
        for option in options:
            if option[0] not in seen:
                unique_options.append(option)
                seen.add(option[0])
        return unique_options
    
    return options


def mark_unlimited_amounts(options, amounts, min_amount):
    """
    Mark terminal node types as unlimited in the amounts dictionary, storing the minimum count.
    """
    for option_list in options:
        for option in option_list:
            if isinstance(option, str) and not option.startswith("RECURSIVE:"):
                amounts[option] = min_amount


def print_parsed_rules(parsed_rules):
    """
    Print the parsed rules in a readable format.
    """
    print("\n" + "="*60)
    print("PARSED VALIDATION RULES")
    print("="*60)
    
    for node_type, rules in parsed_rules.items():
        print(f"\n{node_type}:")
        print(f"  Combinations:")
        for i, combo in enumerate(rules['Combinations']):
            print(f"    Shape {i+1}: {combo}")
        if rules['Amounts']:
            # Display amounts with their minimums
            amount_str = ", ".join([f"{k} ({v}+)" for k, v in rules['Amounts'].items()])
            print(f"  Unlimited amounts: {amount_str}")


def create_simplified_rules(parsed_rules):
    """
    Create a simplified version of the rules for easier understanding.
    """
    simplified = {}
    
    for node_type, rules in parsed_rules.items():
        simplified[node_type] = {
            "allowed_children": [],
            "unlimited_nodes": list(rules['Amounts'].keys())
        }
        
        for combo in rules['Combinations']:
            shape = []
            for position in combo:
                position_options = []
                for option_list in position:
                    if len(option_list) == 1:
                        position_options.append(option_list[0])
                    else:
                        position_options.append(option_list)
                shape.append(position_options)
            simplified[node_type]["allowed_children"].append(shape)
    
    return simplified


def extract_tree_shapes(xml_string, parsed_rules):
    """
    Extract the shapes of all nodes with children from a behavior tree XML.
    
    This version collapses consecutive duplicates of nodes that are marked as "unlimited"
    in the context of their parent node.
    
    Args:
        xml_string: The XML string of the behavior tree
        parsed_rules: The parsed grammar rules with context-specific unlimited amounts
        
    Returns:
        Dictionary with node types as keys and their flattened, collapsed shapes as values
    """
    try:
        # Parse the XML
        root = ET.fromstring(xml_string.strip())
        
        shapes = {}
        node_counters = {}  # To handle multiple instances of the same node type
        
        def remove_consecutive_duplicates(child_types, parent_node_type):
            """Remove consecutive duplicate node types only for nodes that are allowed unlimited amounts in this parent context."""
            if not child_types:
                return child_types
            
            # Get unlimited amounts for this specific parent node type
            unlimited_nodes = set()
            if parent_node_type in parsed_rules:
                unlimited_nodes = set(parsed_rules[parent_node_type]["Amounts"].keys())
            
            result = [child_types[0]]
            for i in range(1, len(child_types)):
                # Only remove consecutive duplicates if this node type is allowed unlimited amounts for this parent
                if child_types[i] != child_types[i-1] or child_types[i] not in unlimited_nodes:
                    result.append(child_types[i])
            return result
        
        def extract_node_shape(node):
            """Recursively extract the shape of a node and its children."""
            node_type = node.tag
            
            children = [child for child in node if child.tag is not None]
            
            if children:
                # This node has children, extract their types
                child_types = [child.tag for child in children]
                
                # IMPORTANT: Collapse consecutive duplicates based on parent's rules
                collapsed_child_types = remove_consecutive_duplicates(child_types, node_type)
                
                # Use a counter to create unique keys for multiple instances of the same node type
                if node_type not in node_counters:
                    node_counters[node_type] = 0
                node_counters[node_type] += 1
                instance_count = node_counters[node_type]
                
                instance_key = f"{node_type}" if instance_count == 1 else f"{node_type}{instance_count}"

                if node_type not in shapes:
                    shapes[node_type] = {}
                shapes[node_type][instance_key] = collapsed_child_types
                
                # Recursively process children
                for child in children:
                    extract_node_shape(child)
        
        # Start extraction from root
        extract_node_shape(root)
        
        return shapes
        
    except ET.ParseError as e:
        return {"error": f"XML parsing error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


def print_tree_shapes(shapes):
    """
    Print the extracted tree shapes in a readable format.
    """
    print("\n" + "="*60)
    print("EXTRACTED TREE SHAPES")
    print("="*60)
    
    if "error" in shapes:
        print(f"Error: {shapes['error']}")
        return
    
    for node_type, node_instances in shapes.items():
        print(f"\n{node_type}:")
        if isinstance(node_instances, dict):
            # Multiple instances
            for instance_id, shape in node_instances.items():
                print(f"  {instance_id}: {shape}")
        else:
            # Should not happen with the new logic, but handle just in case
            print(f"  Shape: {node_instances}")


def validate_tree(tree_shapes, flattened_grammar_shapes):
    """
    Validate if all tree shapes are allowed by the flattened grammar shapes.
    
    Args:
        tree_shapes: Shapes extracted from the actual tree (with duplicates collapsed)
        flattened_grammar_shapes: Allowed base shapes from the grammar
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    if "error" in tree_shapes:
        validation_results["valid"] = False
        validation_results["errors"].append(tree_shapes["error"])
        return validation_results

    for node_type, instances in tree_shapes.items():
        if node_type not in flattened_grammar_shapes:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Node type '{node_type}' not found in grammar")
            continue
        
        allowed_shapes = flattened_grammar_shapes[node_type]
        
        for instance_id, instance_shape in instances.items():
            if instance_shape not in allowed_shapes:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Node '{instance_id}' has invalid shape {instance_shape}. "
                    f"Allowed base shapes for '{node_type}': {allowed_shapes}"
                )
    
    return validation_results


def flatten_grammar_shapes(parsed_rules):
    """
    Flattens the parsed grammar rules into a simple list of allowed child sequences (shapes).
    This handles the AND/OR logic but does NOT expand for unlimited repetitions.
    """
    flattened_shapes = {}
    for node_type, rules in parsed_rules.items():
        all_node_type_shapes = []
        for combination in rules["Combinations"]:
            # Generate flattened shapes for this specific combination rule
            shapes_from_combo = _generate_flattened_shapes_from_combination(combination)
            for shape in shapes_from_combo:
                if shape not in all_node_type_shapes:
                    all_node_type_shapes.append(shape)
        flattened_shapes[node_type] = all_node_type_shapes
    return flattened_shapes

def _generate_flattened_shapes_from_combination(combination):
    """
    Helper to generate shapes from a single combination rule by handling AND/OR logic.
    It computes the Cartesian product of choices at each position in the rule.
    
    Args:
        combination: A list of positions, where each position is a list of choices (OR),
                     and each choice is a list of nodes (AND).
                     e.g., [[['A'],['B']], [['C', 'D']]] -> (A or B) then (C and D)
    
    Returns:
        A list of flattened shapes, e.g., [['A', 'C', 'D'], ['B', 'C', 'D']]
    """
    import itertools
    
    # Get the Cartesian product of choices at each position.
    all_choice_paths = list(itertools.product(*combination))
    
    all_flattened_shapes = []
    for path in all_choice_paths:
        # path is a tuple of choices, e.g., (['A'], ['C', 'D'])
        flattened_shape = []
        for choice in path:
            # choice is a list of nodes to be ANDed, e.g., ['C', 'D']
            flattened_shape.extend(choice)
        
        if flattened_shape not in all_flattened_shapes:
            all_flattened_shapes.append(flattened_shape)
            
    return all_flattened_shapes


def print_flattened_shapes(flattened_shapes):
    """
    Print the flattened shapes in a readable format.
    """
    print("\n" + "="*60)
    print("FLATTENED GRAMMAR BASE SHAPES")
    print("="*60)
    
    for node_type, shapes in flattened_shapes.items():
        print(f"\n{node_type}:")
        # Sort for consistent output
        sorted_shapes = sorted(shapes, key=lambda x: (len(x), str(x)))
        for i, shape in enumerate(sorted_shapes):
            print(f"  {i+1}. {shape}")


def test_all_examples():
    """
    Test all correct and incorrect examples from new_syntax_check.py using the new logic.
    """
    # Import the examples
    import sys
    import os
    # Ensure the current directory is in the path to find new_syntax_check
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))
    from new_syntax_check import get_correct_examples, get_incorrect_examples
    
    # Get the grammar rules and parse them
    grammar_rules_new = {                                                                 
        "B":   [["b", ["SEL"]], ["b", ["SEQ"]]],                                                          
        "SEL": [["sel", ["SEQn", "As"]], ["sel", ["SEQn"]]],                                               
        "SEQn":[["SEQ", "SEQn"], ["SEQ"]], 
        "SEQ": [["seq", ["Pn", "A"]], ["seq", ["As", "Pn", "A"]]],
        "b":   ["BehaviorTree", ["children_nodes"]],     
        "sel": ["Selector", ["children_nodes"]],
        "seq": ["Sequence", ["children_nodes"]],                                            
        "A":   [["aa", "sa"], ["aa"], ["sa"]],                                                                  
        "As":  [["aa"], ["sa"]],                                                                  
        "aa":  ["ActuatorAction"],                                                    
        "sa":  ["StateAction"],
        "Pn":  [["p", "Pn"], ["p"], []], 
        "p":   ["Condition"]
    }
    
    parsed_rules = parse_grammar_to_validation_rules(grammar_rules_new)
    flattened_grammar_shapes = flatten_grammar_shapes(parsed_rules)
    
    print("\n" + "="*80)
    print("TESTING ALL EXAMPLES FROM new_syntax_check.py")
    print("="*80)
    
    # Test correct examples
    correct_examples = get_correct_examples()
    print(f"\n🟢 TESTING {len(correct_examples)} CORRECT EXAMPLES:")
    print("-" * 50)
    
    correct_passed = 0
    for i, example in enumerate(correct_examples, 1):
        tree_xml = example["tree"]
        
        # Extract shapes (with collapsing) and validate
        tree_shapes = extract_tree_shapes(tree_xml, parsed_rules)
        validation = validate_tree(tree_shapes, flattened_grammar_shapes)
        
        if validation["valid"]:
            print(f"✅ Correct Example {i}: PASSED (correctly identified as valid)")
            correct_passed += 1
        else:
            print(f"❌ Correct Example {i}: FAILED (incorrectly identified as invalid)")
            print(f"   Got errors: {validation['errors']}")
    
    # Test incorrect examples
    incorrect_examples = get_incorrect_examples()
    print(f"\n🔴 TESTING {len(incorrect_examples)} INCORRECT EXAMPLES:")
    print("-" * 50)
    
    incorrect_passed = 0
    for i, example in enumerate(incorrect_examples, 1):
        tree_xml = example["tree"]
        expected_error_part = example["expected_error"] # Part of the expected error message
        
        # Extract shapes (with collapsing) and validate
        tree_shapes = extract_tree_shapes(tree_xml, parsed_rules)
        validation = validate_tree(tree_shapes, flattened_grammar_shapes)
        
        if not validation["valid"]:
            print(f"✅ Incorrect Example {i}: PASSED (correctly identified as invalid)")
            incorrect_passed += 1
        else:
            print(f"❌ Incorrect Example {i}: FAILED (incorrectly identified as valid)")
            print(f"   Expected error involving: '{expected_error_part}'")
            print(f"   But validation passed")

    # Summary
    total_correct = len(correct_examples)
    total_incorrect = len(incorrect_examples)
    total_examples = total_correct + total_incorrect
    total_passed = correct_passed + incorrect_passed
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Correct Examples:   {correct_passed}/{total_correct} passed")
    print(f"Incorrect Examples: {incorrect_passed}/{total_incorrect} passed")
    print(f"Overall Score:      {total_passed}/{total_examples} ({total_passed/total_examples*100:.1f}%)")
    
    if total_passed == total_examples:
        print("\n🎉 PERFECT SCORE! All examples validated correctly!")
    else:
        print("\n🟡 NEEDS IMPROVEMENT. Some examples failed validation.")


# Test the parser
if __name__ == "__main__":
    # Test with the new grammar rules
    grammar_rules_new = {                                                                 
        "B":   [["b", ["SEL"]], ["b", ["SEQ"]]],                                                          
        "SEL": [["sel", ["SEQn", "As"]], ["sel", ["SEQn"]]],                                               
        "SEQn":[["SEQ", "SEQn"], ["SEQ"]], 
        "SEQ": [["seq", ["Pn", "A"]], ["seq", ["As", "Pn", "A"]]],
        "b":   ["BehaviorTree", ["children_nodes"]],     
        "sel": ["Selector", ["children_nodes"]],
        "seq": ["Sequence", ["children_nodes"]],                                            
        "A":   [["aa", "sa"], ["aa"], ["sa"]],                                                                  
        "As":  [["aa"], ["sa"]],                                                                  
        "aa":  ["ActuatorAction"],                                                    
        "sa":  ["StateAction"],
        "Pn":  [["p", "Pn"], ["p"], []], 
        "p":   ["Condition"]
    }
    
    parsed_rules = parse_grammar_to_validation_rules(grammar_rules_new)
    print_parsed_rules(parsed_rules)
    
    # Generate flattened base shapes from grammar
    flattened_grammar_shapes = flatten_grammar_shapes(parsed_rules)
    print_flattened_shapes(flattened_grammar_shapes)

    # Example of a single incorrect tree for demonstration
    incorrect_tree_single_test = """<BehaviorTree>
        <Selector>
            <Sequence>
                <Condition>is_agent_holding_good_part</Condition>
                <ActuatorAction>drop_part</ActuatorAction>
                <Sequence> <!-- This makes the parent Sequence invalid -->
                    <Condition>is_agent_holding_good_part</Condition>
                </Sequence>
            </Sequence>
        </Selector>
    </BehaviorTree>"""
    
    print("\n" + "="*60)
    print("SINGLE TREE VALIDATION EXAMPLE")
    print("="*60)
    
    # Extract shapes from the tree, collapsing duplicates
    tree_shapes_single = extract_tree_shapes(incorrect_tree_single_test, parsed_rules)
    print_tree_shapes(tree_shapes_single)
    
    # Validate against flattened grammar shapes
    validation_single = validate_tree(tree_shapes_single, flattened_grammar_shapes)
    
    if validation_single["valid"]:
        print("\n✅ Tree is syntactically valid!")
    else:
        print("\n❌ Tree has syntax errors:")
        for error in validation_single["errors"]:
            print(f"  - {error}")

    # Run the full test suite from new_syntax_check.py
    test_all_examples()

