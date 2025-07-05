from vi import Config, Window

from data_grammar import DatasetGenerator
from agent_control.simulation.agents import RobotAgent
from agent_control.simulation.envs import RobotEnvironment 

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

# Lower numbers and commented mid complexity trees for brevity.
trees_per_params = [

    # Simple trees, get only good parts.
    [5, simple2_parameters, [
        "good_parts_picked_up",
    ]],

    # Simple trees, get only bad parts.
    [5, simple2_parameters, [
        "bad_parts_picked_up"
    ]],

    # # Mid-complexity trees structure 1, get only good parts.
    # [10, mid1_parameters, {
    #     "good_parts_picked_up": 1
    # }],

    # # Mid-complexity trees structure 1, get only bad parts.
    # [10, mid1_parameters, {
    #     "bad_parts_picked_up": 1,
    # }],

    # # Mid-complexity trees structure 2, get only good parts.
    # [10, mid2_parameters, {
    #     "good_parts_picked_up": 1,
    # }],

    # # Mid-complexity trees structure 2, get only bad parts.
    # [10, mid2_parameters, {
    #     "bad_parts_picked_up": 1,
    # }],

]

# Initialize the generator with mixed structures + filtering
generator_filtered = DatasetGenerator(
    agent_class=RobotAgent,
    output_dir="./examples/04_experimental_setup_example/datasets/structured_filtered_v1",
    seed=42,
    grammar_parameters=trees_per_params,  
    grammar_rules=grammar_rules
)


print("\n=== Generating Dataset A with Mixed Structures + Filtering + Enrichment ===")
dataset_a_path = generator_filtered.generate_dataset_a(
    dataset_name="mixed_structure_dataset_a_filtered_enriched",
    # max_trees_to_process will be calculated automatically from trees_per_params when using mixed structures
    filter_env=filter_env,
    enrich_dataset=True
)

print("\n=== Generating Dataset B with Mixed Structures + Filtering + Enrichment ===")
dataset_b_path = generator_filtered.generate_dataset_b(
    dataset_name="mixed_structure_dataset_b_filtered_enriched",
    # max_trees_to_process will be calculated automatically from trees_per_params when using mixed structures
    filter_env=filter_env,
    enrich_dataset=True
)



