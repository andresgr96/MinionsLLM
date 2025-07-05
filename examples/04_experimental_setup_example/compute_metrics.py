import os
import xml.etree.ElementTree as ET
import pandas as pd
import argparse
from tree_parser.grammar_validator import BehaviorTreeGrammarValidator
from tree_parser.primitives_validator import validate_primitives
from agent_control.simulation.agents import RobotAgent
from agent_control.simulation import RobotEnvironment
from vi import Config, Window


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

def extract_behavior_tree(file_path: str) -> tuple[str, str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        bt_start = content.find("<BehaviorTree>")
        bt_end = content.find("</BehaviorTree>") + len("</BehaviorTree>")
        
        if bt_start == -1 or bt_end == -1:
            return None, content

        bt_xml = content[bt_start:bt_end]
        
        return bt_xml, content
    except Exception as e:
        print(f"Error extracting behavior tree from {file_path}: {e}")
        return None, ""


def get_metadata_from_file(file_content: str) -> dict:
    """Extract metadata from the file content."""
    metadata = {
        "ModelName": "unknown",
        "PromptTechnique": "unknown",
        "PromptStyle": "unknown",
        "EnvironmentName": "unknown",
        "Task": "unknown",
        "AgentClass": "unknown"
    }
    
    try:
        _, _, metadata_str = file_content.partition("</BehaviorTree>")
        
        metadata_lines = metadata_str.strip().splitlines()
        for line in metadata_lines:
            line = line.strip()
            if line.startswith("<") and ">" in line and "</" in line:
                try:
                    tag_start = line.find('<') + 1
                    tag_end = line.find('>')
                    tag = line[tag_start:tag_end]

                    value_start = tag_end + 1
                    value_end = line.rfind('</')
                    value = line[value_start:value_end].strip()

                    if tag in metadata:
                        metadata[tag] = value
                except Exception:
                    continue # Ignore malformed metadata lines
                    
    except Exception as e:
        print(f"Error reading metadata: {e}")
    
    return metadata

def get_agent_class(class_name: str):
    if class_name == "RobotAgent":
        return RobotAgent
    # Add other agent classes here if needed
    raise ValueError(f"Unknown agent class: {class_name}")


def calculate_metrics_for_file(file_path: str, grammar_validator: BehaviorTreeGrammarValidator) -> dict:
    bt_xml, file_content = extract_behavior_tree(file_path)
    metadata = get_metadata_from_file(file_content)

    metrics = {
        "model_name": metadata["ModelName"],
        "prompt_technique": metadata["PromptTechnique"],
        "prompt_style": metadata["PromptStyle"],
        "environment_name": metadata["EnvironmentName"],
        "task": metadata["Task"],
        "agent_class": metadata["AgentClass"],
        "syntactic_validity": 0,
        "primitive_validity": 0,
        "total_parts_placed": 0,
        "good_parts_picked_up": 0,
        "bad_parts_picked_up": 0,
        "nest_integrity": 0,
        "parts_dropped_in_base": "[0, 0]",
        "parts_dropped_in_construction": "[0, 0]",
        "parts_dropped_in_storage": "[0, 0]",
        "parts_dropped_in_source": "[0, 0]",
        "parts_dropped_in_waste": "[0, 0]",
        "stopped_moving": False
    }

    if not bt_xml:
        print(f"Could not find BehaviorTree in {file_path}. Skipping.")
        return metrics

    # 1. Syntactic Validity
    is_syntactically_valid, syntax_feedback = grammar_validator.validate_tree(bt_xml)
    if not is_syntactically_valid:
        # print(f"Syntactic validation failed for {file_path}: {syntax_feedback}")
        return metrics
    metrics["syntactic_validity"] = 1

    # 2. Primitive Validity (Hallucinations)
    try:
        agent_class = get_agent_class(metadata["AgentClass"])
        are_primitives_valid, primitive_feedback = validate_primitives(bt_xml, agent_class)
        if not are_primitives_valid:
            print(f"Primitive validation failed for {file_path}: {primitive_feedback}")
            return metrics
        metrics["primitive_validity"] = 1
    except ValueError as e:
        print(f"Error getting agent class for {file_path}: {e}")
        return metrics
    except Exception as e:
        print(f"An unexpected error occurred during primitive validation for {file_path}: {e}")
        return metrics

    # 3. Task Metrics (Simulation)
    print(f"Running simulation for {file_path}...")
    try:
        temp_bt_path = file_path + ".tmp"
        with open(temp_bt_path, 'w', encoding='utf-8') as f:
            f.write(bt_xml)
        if os.path.basename(file_path) == "Gemma_3_12B_QAT_clean_layman_two_shot_trial1.xml":
            print(f"Tree: {bt_xml}")


        config = Config(radius=50, visualise_chunks=False, window=Window.square(500), movement_speed=1.0, duration=1000)
        environment = RobotEnvironment(config=config, bt_path=temp_bt_path, n_agents=10, n_parts=10, task=metrics["task"], headless=True)
        
        environment.setup()
        simulation_results = environment.run()
        
        # Add a conditional print for debugging the specified file
        if os.path.basename(file_path) == "Gemma_3_12B_QAT_clean_layman_two_shot_trial1.xml":
            print("--- DEBUG: Simulation results for target file ---")
            print(simulation_results)
            print("--- END DEBUG ---")

        # Update metrics with simulation results
        metrics.update({
            "total_parts_placed": simulation_results.get("total_parts_placed", 0),
            "good_parts_picked_up": simulation_results.get("good_parts_picked_up", 0),
            "bad_parts_picked_up": simulation_results.get("bad_parts_picked_up", 0),
            "nest_integrity": simulation_results.get("nest_integrity", 0),
            "parts_dropped_in_base": str(simulation_results.get("parts_dropped_in_base", [0, 0])),
            "parts_dropped_in_construction": str(simulation_results.get("parts_dropped_in_construction", [0, 0])),
            "parts_dropped_in_storage": str(simulation_results.get("parts_dropped_in_storage", [0, 0])),
            "parts_dropped_in_source": str(simulation_results.get("parts_dropped_in_source", [0, 0])),
            "parts_dropped_in_waste": str(simulation_results.get("parts_dropped_in_waste", [0, 0])),
            "stopped_moving": simulation_results.get("stopped_moving", False)
        })

        if os.path.exists(temp_bt_path):
            os.remove(temp_bt_path)

    except Exception as e:
        print(f"An unexpected error occurred during simulation for {file_path}: {e}")
        # Reset task metrics if simulation fails
        metrics.update({
            "total_parts_placed": 0, "good_parts_picked_up": 0, "bad_parts_picked_up": 0,
            "nest_integrity": 0, "parts_dropped_in_base": "[0, 0]", "parts_dropped_in_construction": "[0, 0]",
            "parts_dropped_in_storage": "[0, 0]", "parts_dropped_in_source": "[0, 0]", "parts_dropped_in_waste": "[0, 0]",
            "stopped_moving": False
        })

    return metrics



def process_experiment_folder(folder_path: str, output_file: str) -> None:
    metrics_list = []
    grammar_validator = BehaviorTreeGrammarValidator(grammar_rules)

    for root_dir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root_dir, file)
                print(f"---Processing file: {file_path}")
                metrics = calculate_metrics_for_file(file_path, grammar_validator)
                metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list)
    df.to_pickle(output_file)
    print(f"\nMetrics saved to {output_file}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute metrics for experiment results. Saves the data as a dataframe in a pickle file.")
    parser.add_argument(
        '--experiment_folder', 
        type=str, 
        default="experiments/results/final_results_a", 
        help="Path to the experiment folder.")
    parser.add_argument(
        '--output_file', 
        type=str, 
        default=None, 
        help="Path to save the output metrics file. Defaults to saving the file named 'experiment_metrics.pkl' in the exp folder")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = os.path.join(args.experiment_folder, "experiment_metrics.pkl")

    process_experiment_folder(args.experiment_folder, args.output_file)
