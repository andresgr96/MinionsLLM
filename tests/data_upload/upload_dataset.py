from datasets import Dataset
import json
from typing import List, Dict
import os
from control_layer.simulation import RobotAgent

def load_unformatted_dataset(dataset_path: str) -> List[Dict]:
    """Load the unformatted dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def load_xml_template(file_path: str) -> str:
    """Reads the content of an XML template file."""

    with open(file_path, 'r') as file:
        return file.read()

def call_behaviors(agent_class = RobotAgent) -> dict:
    """
    Get all the behaviors from the agent class, extracting only Node Type and Description.
    
    Returns:
        Dictionary of function names and their processed docstrings (Node Type and Description only)
    """
    class_obj = agent_class
    function_names_and_docstrings = {}
    
    for func_name in dir(class_obj):
        if callable(getattr(class_obj, func_name)) and not func_name.startswith("__")\
            and not func_name.startswith("update")\
            and not func_name.startswith("helper")\
                and getattr(class_obj, func_name).__qualname__.startswith(class_obj.__name__ + "."):
            func = getattr(class_obj, func_name)
            if func.__doc__:
                # Split docstring into lines and process
                doc_lines = func.__doc__.strip().split('\n')
                processed_doc = []
                
                for line in doc_lines:
                    line = line.strip()
                    if line.startswith("Node Type:") or line.startswith("Description:"):
                        processed_doc.append(line)
                
                # Join the processed lines back together
                function_names_and_docstrings[func_name] = '\n'.join(processed_doc)
            else:
                function_names_and_docstrings[func_name] = "No docstring found."

    return function_names_and_docstrings

def format_conversation(prompt: str, tree: str, style: str) -> List[Dict]:
    """Format a single conversation in Llama-3 ShareGPT style."""
    # Load system prompt and example
    system_prompt = load_xml_template("./llm_layer/prompt_techniques/system_prompt.xml")
    bt_3_example = load_xml_template("./llm_layer/prompt_techniques/bt_3.xml")
    behaviors = call_behaviors(RobotAgent)
    
    # Replace placeholders in system prompt
    system_prompt = system_prompt.replace("{bt_3.xml}", bt_3_example)
    system_prompt = system_prompt.replace("{BEHAVIORS}", str(behaviors))

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""USER REQUEST: Generate behavior tree to "{prompt}". Output only the XML behavior tree without extra text or explanations of the tree.

RESPONSE:"""
        },
        {
            "role": "assistant",
            "content": tree
        }
    ]

def create_formatted_dataset(unformatted_data: List[Dict]) -> Dataset:
    """Create a formatted dataset with separate entries for each prompt type."""
    formatted_data = []
    
    for entry in unformatted_data:
        # Create layman version
        formatted_data.append({
            "conversations": format_conversation(
                prompt=entry["layman_task"],
                tree=entry["tree"],
                style="layman"
            )
        })
        
        # Create technical version
        formatted_data.append({
            "conversations": format_conversation(
                prompt=entry["technical_task"],
                tree=entry["tree"],
                style="technical"
            )
        })
        
        # Create spoon-fed version
        formatted_data.append({
            "conversations": format_conversation(
                prompt=entry["spoon_task"],
                tree=entry["tree"],
                style="spoon"
            )
        })
    
    return Dataset.from_list(formatted_data)

def upload_to_hub(dataset: Dataset, repo_id: str):
    """Upload the dataset to the Hugging Face Hub."""
    dataset.push_to_hub(
        repo_id,
        private=False  # Set to False if you want it public
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Format and upload dataset to HuggingFace Hub")
    parser.add_argument(
        '--input_path',
        type=str,
        default="./experiments/datasets/structured_filtered_v1/datasets/mixed_structure_dataset_a_filtered_enriched.json",
        help="Path to unformatted dataset JSON"
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        required=False,
        default="Andresgr96/bt_dataset_a_v1",
        help="HuggingFace Hub repository ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        '--token',
        type=str,
        required=False,
        help="HuggingFace Hub API token"
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=None,
        help="Number of examples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load and possibly truncate dataset
    data = load_unformatted_dataset(args.input_path)
    if args.test_size:
        data = data[:args.test_size]
        print(f"Processing {args.test_size} examples for testing")
    
    # Format dataset
    formatted_dataset = create_formatted_dataset(data)
    print(f"Created dataset with {len(formatted_dataset)} examples")
    print(formatted_dataset[0])
    
    # Upload to Hub
    upload_to_hub(formatted_dataset, args.repo_id)
    print(f"Dataset uploaded to {args.repo_id}")