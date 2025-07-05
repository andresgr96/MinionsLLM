"""
Example showing how to use the DatasetGenerator class to generate
behavior trees and datasets with mixed structures.

Instead of having to run a filtered dataset generation for each structure,
we can generate a mixed dataset with all the structures, additionally we can filter it.
"""

from vi import Config, Window

from data_grammar import DatasetGenerator
from control_layer.simulation.agents import RobotAgent
from control_layer.simulation.envs import RobotEnvironment 

filter_env = RobotEnvironment(
    config=Config(
        radius=50,
        window=Window.square(500),
        movement_speed=1.0,   # Its important to be consistent with speed and duration, otherwise the metrics will be inconsistent
        duration=500 
    ),
    bt_path="dummy_path", # This is a dummy path, we will use the trees in the dataset to test the metrics
    n_agents=10,
    n_parts=10,
    task="dummy_task",
    headless=True
)

# Define custom rules for the grammar
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

simple_parameters = {   
    "B": {"only": 0},                                                                                                           
    "SEL": {"only": 0},                                                                    
    "SEQn": {"only": 1},                                                                    
    "SEQ": {"only": 0},                                                                    
    "As": {"only": 1},                                                                    
    "A": {"only": 1},                                                                    
    "Pn": {"only": 1},                                                                    
}

mid1_parameters = {   
    "B": {"only": 0},                                                                                                           
    "SEL": {"only": 0},                                                                    
    "SEQn": {"only": 0, "list_always": 2},                                                                    
    "SEQ": {"only": 0},                                                                    
    "As": {"only": 1},                                                                    
    "Pn": {"list_max": 2, "exclude": [2]},                                                                    
}

"""
This is the main parameter that controls the generation of the mixed dataset.   [n_trees, parameters, metrics]
If the metrics are provided as a list, the filtering uses an OR operator to decide validity.
If the metrics are provided as a dictionary, the filtering uses an AND operator to decide validity.
"""

trees_per_params = [

    # Simple trees, get only good parts.
    [2, simple_parameters, [
        "good_parts_picked_up",
    ]],

    # Simple trees, get only bad parts.
    [2, simple_parameters, [
        "bad_parts_picked_up"
    ]],

    # Mid-complexity trees structure 1, get only good parts.
    [1, mid1_parameters, {
        "good_parts_picked_up": 1
    }],

]

# Initialize the generator with mixed structures + filtering
generator_filtered = DatasetGenerator(
    agent_class=RobotAgent,
    output_dir="./examples/data_gen_examples/output/mixed_test_run",
    seed=42,
    grammar_parameters=trees_per_params,  # Feed the structure param here instead of the specific generation methods
    grammar_rules=grammar_rules
)


print("\n=== Generating Dataset A with Mixed Structures + Filtering + Enrichment ===")
dataset_a_path = generator_filtered.generate_dataset_a(
    dataset_name="mixed_structure_dataset_a_filtered",
    # max_trees_to_process will be calculated automatically from trees_per_params when using mixed structures
    filter_env=filter_env,
)

print("\n=== Generating Dataset B with Mixed Structures + Filtering + Enrichment ===")
dataset_b_path = generator_filtered.generate_dataset_b(
    dataset_name="mixed_structure_dataset_b_filtered",
    # max_trees_to_process will be calculated automatically from trees_per_params when using mixed structures
    filter_env=filter_env,
)



