from vi import Config, Window

from data_grammar import DatasetGenerator
from control_layer.simulation.agents import RobotAgent
from control_layer.simulation.envs import RobotEnvironment 

filter_env = RobotEnvironment(
    config=Config(
        window=Window.square(500),
        movement_speed=1.0,   # Its important to be consistent with speed and duration, otherwise the metrics will be inconsistent
        duration=500 
    ),
    bt_path="dummy_path", # This is a dummy path, we will use the trees in the dataset to test the metrics
    n_agents=5,
    n_parts=5,
    task="dummy_task",
    headless=True
)

# Just adding them as reminder
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

simple1_parameters = {   
    "B": {"only": 1},                                                                                                           
    "SEQ": {"only": 1},                                                                    
    "As": {"only": 1},                                                                    
    "A": {"only": 1},                                                                    
    "Pn": {"only": 1},                                                                    
}

simple2_parameters = {   
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

mid2_parameters = {   
    "B": {"only": 0},                                                                                                           
    "SEL": {"only": 0},                                                                    
    "SEQn": {"only": 0, "list_always": 3},                                                                    
    "SEQ": {"only": 0},                                                                    
    "As": {"only": 1},                                                                    
    "Pn": {"list_max": 2, "exclude": [2]},                                                                    
}

complex_parameters = {   
    "B": {"only": 0},                                                                                                           
    "SEL": {"only": 0},                                                                    
    "SEQn": {"only": 0, "list_always": 6},                                                                    
    "SEQ": {"only": 0},                                                                    
    "As": {"only": 1},
    "A": {"exclude": [0]},                                                                                                                         
    "Pn": {"list_max": 2, "exclude": [2]},                                                                    
}

trees_per_params_no_filtering = [

    
    [5, simple2_parameters],

    # Mid-complexity parameters with part dropping metrics
    [5, mid1_parameters],


    [5, mid2_parameters],

    # Complex parameters with part dropping metrics
    [5, complex_parameters]
]

trees_per_params = [
    # Simple parameters with basic part pickup metrics
    # [2, simple1_parameters, {
    #     "good_parts_picked_up": 1,
    #     "bad_parts_picked_up": 1
    # }],
    
    # # Also test legacy mode
    # [50, simple2_parameters, [
    #     "good_parts_picked_up",
    #     "bad_parts_picked_up"
    # ]],

    # Mid-complexity parameters with part dropping metrics
    [20, mid1_parameters, {
        "good_parts_picked_up": 1
    }],
    # Mid-complexity parameters with part dropping metrics
    [20, mid1_parameters, {
        "bad_parts_picked_up": 1,
        "parts_dropped_in_waste": [0, 1]
    }],

    # [20, mid2_parameters, {
    #     "parts_dropped_in_base": [1, 0],
    #     "good_parts_picked_up": 1,
    #     "bad_parts_picked_up": 1,
    #     "parts_dropped_in_waste": [0, 1]
    # }],

    # Complex parameters with part dropping metrics
    [10, complex_parameters, {
        "parts_dropped_in_base": [1, 0],
        "good_parts_picked_up": 1,
        "bad_parts_picked_up": 1,
        "parts_dropped_in_waste": [0, 1]
    }]
]





# # Initialize the generator with mixed structures
# generator = DatasetGenerator(
#     output_dir="./tests/output/grammar_test_run/new_grammar_test",
#     seed=42,
#     grammar_parameters=trees_per_params_no_filtering,  
#     grammar_rules=grammar_rules
# )


# Generate dataset with mixed structures using method A (populated trees)
# Note: n_trees parameter is ignored when using mixed structures

# print("\n=== Generating Dataset A with Mixed Structures ===")
# dataset_a_path = generator.generate_dataset_a(
#     dataset_name="mixed_structure_dataset_a",
# )

# # Generate dataset with mixed structures using method B (placeholder trees)
# print("\n=== Generating Dataset B with Mixed Structures ===")
# dataset_b_path = generator.generate_dataset_b(
#     dataset_name="mixed_structure_dataset_b",
# )

# Initialize the generator with mixed structures + filtering
generator_filtered = DatasetGenerator(
    agent_class=RobotAgent,
    output_dir="./tests/output/grammar_test_run/metrics_filtering_a",
    seed=42,
    grammar_parameters=trees_per_params,  # This is now a list of [count, params] pairs
    grammar_rules=grammar_rules
)

# print("\n=== Generating Dataset A with Mixed Structures + Filtering ===")
# dataset_a_path = generator_filtered.generate_dataset_a(
#     dataset_name="mixed_structure_dataset_a_filtered",
#     # max_trees_to_process will be calculated automatically from trees_per_params when using mixed structures
#     filter_env=filter_env,
#     # filter_metrics=["good_parts_picked_up", "bad_parts_picked_up"]  
# )


print("\n=== Generating Dataset B with Mixed Structures + Filtering ===")
dataset_b_path = generator_filtered.generate_dataset_b(
    dataset_name="mixed_structure_dataset_b_filtered",
    # max_trees_to_process will be calculated automatically from trees_per_params when using mixed structures
    filter_env=filter_env
)



