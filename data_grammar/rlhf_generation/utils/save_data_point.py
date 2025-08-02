""" Save a datapoint in the given path after producing other prompt styles """

import json
import os
from typing import Any, Type

from data_grammar.grammar_gen.node_translations import node_connectors
from data_grammar.grammar_gen.tree_to_prompt import (
    generate_spoon_prompt_from_string,
    generate_technical_prompt_from_string,
)
from tree_parser.agent_doc_parser import AgentDocstringParser


def save_datapoint(
    dataset_path: str, task_description: str, tree_str: str, agent_class: Type[Any]
) -> None:
    """
    Save a single datapoint to the dataset file after generating other prompt styles.

    Args:
        dataset_path: Path to the JSON dataset file (should end with .json)
        task_description: The layman task description
        tree_str: The behavior tree XML string
        agent_class: The agent class to extract translations from
    """
    # Ensure the dataset path has .json extension
    if not dataset_path.endswith(".json"):
        dataset_path = dataset_path + ".json"

    # Retrieve the translations
    doc_parser = AgentDocstringParser(agent_class=agent_class)
    extracted_info_dict = doc_parser.extract_docstring_config()
    spoon_translations = extracted_info_dict["spoon_node_translations"]
    technical_translations = extracted_info_dict["node_translations"]

    # Produce the prompts
    layman_prompt = task_description
    try:
        spoon_prompt = generate_spoon_prompt_from_string(
            tree_string=tree_str,
            spoon_translations=spoon_translations,
            connectors=node_connectors,
        )
    except Exception as e:
        spoon_prompt = f"Error generating spoon task: {e}"

    try:
        tech_prompt = generate_technical_prompt_from_string(
            tree_string=tree_str,
            translations=technical_translations,
            connectors=node_connectors,
        )
    except Exception as e:
        tech_prompt = f"Error generating technical task: {e}"

    # Create the datapoint in the same format as the other generators
    datapoint = {
        "layman_task": layman_prompt,
        "technical_task": tech_prompt,
        "spoon_task": spoon_prompt,
        "tree": tree_str,
    }

    # Load existing dataset or create new one
    dataset = []
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r") as json_file:
                dataset = json.load(json_file)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file exists but is corrupted or empty, start fresh
            dataset = []

    # Append the new datapoint
    dataset.append(datapoint)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    # Save the updated dataset
    with open(dataset_path, "w") as json_file:
        json.dump(dataset, json_file, indent=4)

    print(
        f"Datapoint saved to {dataset_path}. Dataset now contains {len(dataset)} samples."
    )
