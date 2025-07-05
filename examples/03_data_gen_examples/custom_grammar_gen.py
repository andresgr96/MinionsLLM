"""
Example showing how to use the DatasetGenerator class to generate
behavior trees and datasets while providing a custom grammar.
"""

from data_grammar import DatasetGenerator

""" 
    Here we define everything regarding the grammar, its production rules and parameters to control the size of the trees generated.
    Its critical to understand that there are several ways too change the gramar, this is the most simple one which directly changes the current default rules.
    But adding new nodes types also fundamentally changes the grammar, and will be explained in the full setup example.
"""

grammar_rules = {                                                                 
    "B":   [["b", ["SEL"]]],          # Removed SEL node from first rule to check if custom grammar works, now all trees should start with a SEQ node after root (B) node                                              
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

''' 
    This dictionary allows to further customize the tree generation according to the usecase.

    Params:
        "list_max": For list expansions, it controls the max amount of times that recursion is allowed to 'list_max', forcing expanding to the single version of the symbol.
        "list_always": For list expansions, it controls the exact amount of the nodes to be chosen, forcing always having 'list_always' amount of the list expansion.
        "only": For any rule, it forces choosing the option at the given index, regardless of the current integer in the list. Pay attention when choosing 0 for lists expansions,
            not declaring any of the two params  above can result in infinite recursion.
        "exclude[]": For any rule, it excludes the options at the indexes contained in ths list, regardless of the current integer in the list.
        "parent": For any rule, if the immediate parent is of this type, it forces choosing the option at the given index, regardless of the current integer 
            in the list. It is a dictionary so it can be used to force different options for different parents.

 '''

grammar_parameters = {                                                                                                              
    "SEQn": {"list_max": 5, "parent": {"SEL": 0}},  # Changed from 11 to 5, all trees should have max 5 SEQn nodes in a row
    "SEQ": {"only": 0},                                                                    
    "Pn": {"list_max": 1},     # Changed from 4 to 1, all trees should have max one condition in a row

}

""" 
    This dictionary controls how to connect the translation of the nodes depending on the type of node to achieve a more natural language for prompts.
    The node translations are taken directly form the agents class method docstrings. This makes the process more streamlined while also
    encapsulating the design of the agent, its capabilities and how they ae reflected into natual language in one place.
"""

# Obvius changes to spot in the dataset
node_connectors = {
    "Selector": "oooor",
    "Sequence": "aaaand",
    "Condition": "iiiif",
    "StateAction": "theeeen",
    "ActuatorAction": "theeeen"
}

# Initialize the generator with a small output directory
generator = DatasetGenerator(
    output_dir="./examples/03_data_gen_examples/output/grammar_test_run",
    seed=42,
    grammar_rules=grammar_rules,
    grammar_parameters=grammar_parameters,
    node_connectors=node_connectors
)

# Generate a small dataset using method A (populated trees)
print("\n=== Generating Dataset A ===")
dataset_a_path = generator.generate_dataset_a(
    dataset_name="grammar_dataset_a",
    n_trees=5,  # Small number for testing
    max_trees_to_process=5  # Process only 5 trees
)
print(f"Dataset A saved to: {dataset_a_path}")

# Generate a small dataset using method B (unpopulated trees with placeholders)
print("\n=== Generating Dataset B ===")
dataset_b_path = generator.generate_dataset_b(
    dataset_name="grammar_dataset_b",
    n_trees=5,  # Small number for testing
    max_trees_to_process=5  # Process only 5 trees
)
print(f"Dataset B saved to: {dataset_b_path}")

