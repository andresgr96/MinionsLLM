#!/usr/bin/env python3
"""
Test script for grammar validation using the BehaviorTreeGrammarValidator class.

This script tests all correct and incorrect examples from new_syntax_check.py
using the encapsulated grammar validator.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
# This is important for running the script directly
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from grammar_validator import BehaviorTreeGrammarValidator
from new_syntax_check import get_correct_examples, get_incorrect_examples


def main():
    """
    Main function to test all examples using the grammar validator.
    """
    # Define the grammar rules
    grammar_rules = {                                                                 
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
    
    # Initialize the validator
    print("Initializing BehaviorTreeGrammarValidator...")
    validator = BehaviorTreeGrammarValidator(grammar_rules)
    
    # Display grammar information for verification
    grammar_info = validator.get_grammar_info()
    print("\n" + "="*80)
    print("FLATTENED GRAMMAR BASE SHAPES (from Validator Class)")
    print("="*80)
    
    for node_type, shapes in grammar_info["flattened_shapes"].items():
        print(f"\n{node_type}:")
        # Sort for consistent, readable output
        sorted_shapes = sorted(shapes, key=lambda x: (len(x), str(x)))
        for i, shape in enumerate(sorted_shapes, 1):
            print(f"  {i}. {shape}")
    
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
        
        # Validate using the class
        is_valid, message = validator.validate_tree(tree_xml)
        
        if is_valid:
            print(f"✅ Correct Example {i}: PASSED (correctly identified as valid)")
            correct_passed += 1
        else:
            print(f"❌ Correct Example {i}: FAILED (incorrectly identified as invalid)")
            print(f"   Got error: {message}")
    
    # Test incorrect examples
    incorrect_examples = get_incorrect_examples()
    print(f"\n🔴 TESTING {len(incorrect_examples)} INCORRECT EXAMPLES:")
    print("-" * 50)
    
    incorrect_passed = 0
    for i, example in enumerate(incorrect_examples, 1):
        tree_xml = example["tree"]
        expected_error_part = example["expected_error"]
        
        # Validate using the class
        is_valid, message = validator.validate_tree(tree_xml)
        
        if not is_valid:
            print(f"✅ Incorrect Example {i}: PASSED (correctly identified as invalid)")
            incorrect_passed += 1
        else:
            print(f"❌ Incorrect Example {i}: FAILED (incorrectly identified as valid)")
            print(f"   Expected error involving: '{expected_error_part}'")
            print(f"   But validation passed with message: {message}")
    
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
    
    return total_passed == total_examples


if __name__ == "__main__":
    success = main()
    # Exit with code 1 on failure, useful for CI/CD pipelines
    sys.exit(0 if success else 1) 