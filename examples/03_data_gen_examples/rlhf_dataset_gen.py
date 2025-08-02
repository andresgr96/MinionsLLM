"""
Example showing how to use the UnifiedRLHFUI class to generate
behavior trees and datasets through a collaboration between LLM and human.
"""
from agent_control import RobotAgent, RobotEnvironment
from data_grammar.rlhf_generation.rlhf_unified import UnifiedRLHFUI

# Step 1: Define grammar rules for behavior tree validation
grammar_rules = {
    "B": [["b", ["SEL"]], ["b", ["SEQ"]]],                    
    "SEL": [["sel", ["SEQn", "As"]], ["sel", ["SEQn"]]],      
    "SEQn": [["SEQ", "SEQn"], ["SEQ"]],                       
    "SEQ": [["seq", ["Pn", "A"]], ["seq", ["As", "Pn", "A"]]],  
    "b": ["BehaviorTree", ["children_nodes"]],                
    "sel": ["Selector", ["children_nodes"]],                  
    "seq": ["Sequence", ["children_nodes"]],                  
    "A": [["aa", "sa"], ["aa"], ["sa"]],                      
    "As": [["aa"], ["sa"]],                                   
    "aa": ["ActuatorAction"],                                 
    "sa": ["StateAction"],                                    
    "Pn": [["p", "Pn"], ["p"], []],                          
    "p": ["Condition"],                                       
}

# Step 2: Configure simulation parameters
config_kwargs = {
    "radius": 40,                    
    "duration": 500,                 
    "movement_speed": 1.0,           
    "window_size": 800,              
    "visualise_chunks": True,        
}

# Step 3: Configure environment parameters
environment_kwargs = {
    "n_agents": 10,                  
    "n_parts": 10,                   
    "task": "custom_task",           
    "headless": False,               
}

# Step 4: Initialize the RLHF dataset generation UI
# This creates a unified interface for interactive behavior tree generation
ui = UnifiedRLHFUI(
    dataset_path="./examples/03_data_gen_examples/output/dataset_path.json",  # Output dataset file
    dataset_size_goal=10,                                                     # Target number of datapoints
    agent_class=RobotAgent,                                                   # Agent class for validation
    grammar_rules=grammar_rules,                                              # Grammar rules defined above
    environment_class=RobotEnvironment,                                       # Environment class for simulation
    environment_kwargs=environment_kwargs,                                    # Environment parameters
    config_kwargs=config_kwargs,                                              # Simulation configuration
)

# Step 5: Start the interactive RLHF dataset generation process
# - Define tasks in natural language
# - Generate behavior trees using LLM
# - Validate trees against grammar and primitives
# - Run simulations to test behavior
# - Provide human feedback for iterative improvement
# - Save successful datapoints to the dataset
ui.run()