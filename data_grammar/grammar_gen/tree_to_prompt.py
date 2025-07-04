"""Convert behavior trees to natural language prompts."""

import os
import xml.etree.ElementTree as ET
from typing import Dict


def translate_node(
    node: ET.Element, translations: Dict[str, str], connectors: Dict[str, str]
) -> str:
    """Recursively translates a node and its children into natural language."""
    tag = node.tag
    children = list(node)
    is_selector = tag == "Selector"
    is_sequence = tag == "Sequence"

    if not children:
        text_content = node.text.strip() if node.text else ""
        translation = translations.get(text_content, text_content)
        if tag in connectors:
            return f"{connectors[tag]} {translation}"
        return translation

    if tag in connectors:
        connector = f" {connectors[tag]} "
        child_translations = []

        if is_selector:
            child_translations = [
                translate_node(child, translations, connectors) + "," + connector
                for child in children[:-1]
            ]
        elif is_sequence:
            # Remove "and" between Condition and Action nodes for better readability
            for i, child in enumerate(children):
                if (
                    i > 0
                    and children[i - 1].tag == "Condition"
                    and child.tag in {"ActuatorAction", "StateAction"}
                ):
                    child_translations.append(
                        " " + translate_node(child, translations, connectors).strip()
                    )
                else:
                    if i > 0:
                        child_translations.append(
                            connector
                            + translate_node(child, translations, connectors).strip()
                        )
                    else:
                        child_translations.append(
                            translate_node(child, translations, connectors).strip()
                        )
        else:
            child_translations = [
                translate_node(child, translations, connectors).strip()
                for child in children
            ]

        return "".join(child_translations)

    return "".join(
        translate_node(child, translations, connectors) for child in children
    )


def generate_technical_prompt_from_string(
    tree_string: str, translations: Dict[str, str], connectors: Dict[str, str]
) -> str:
    """Generate a technical-style prompt from a tree represented as a string."""
    try:
        root = ET.fromstring(tree_string)
        if root.tag != "BehaviorTree" or len(root) != 1:
            raise ValueError("Invalid BehaviorTree structure")

        main_node = root[0]
        prompt = translate_node(main_node, translations, connectors)

        # Add "otherwise" for Selectors
        if main_node.tag == "Selector" and len(main_node) > 1:
            prompt += (
                "otherwise "
                + translate_node(main_node[-1], translations, connectors)
                + "."
            )
        else:
            prompt += "."

        # Remove leading "then" if present
        if prompt.lower().startswith("then "):
            prompt = prompt[5:]

        return prompt

    except ET.ParseError as e:
        raise ValueError(f"Error parsing tree string: {e}")


def generate_spoon_prompt_from_string(
    tree_string: str, spoon_translations: Dict[str, str], connectors: Dict[str, str]
) -> str:
    """Generate a spoon-style prompt from a tree represented as a string."""
    try:
        root = ET.fromstring(tree_string)
        if root.tag != "BehaviorTree" or len(root) != 1:
            raise ValueError("Invalid BehaviorTree structure")

        main_node = root[0]
        prompt = translate_node(main_node, spoon_translations, connectors)

        # Add "otherwise" for Selectors
        if main_node.tag == "Selector" and len(main_node) > 1:
            prompt += (
                "otherwise "
                + translate_node(main_node[-1], spoon_translations, connectors)
                + "."
            )
        else:
            prompt += "."

        # Remove leading "then" if present
        if prompt.lower().startswith("then "):
            prompt = prompt[5:]

        return prompt

    except ET.ParseError as e:
        raise ValueError(f"Error parsing tree string: {e}")


def generate_tech_prompt(
    tree_path: str, translations: Dict[str, str], connectors: Dict[str, str]
) -> str:
    """Generate a technical-style prompt from a behavior tree XML file."""
    tree = ET.parse(tree_path)
    root = tree.getroot()

    if root.tag != "BehaviorTree" or len(root) != 1:
        raise ValueError("Invalid BehaviorTree structure")

    main_node = root[0]
    prompt = translate_node(main_node, translations, connectors)

    # Add "otherwise" for Selectors for better understanding of if-else relationships
    if main_node.tag == "Selector" and len(main_node) > 1:
        prompt += (
            "otherwise " + translate_node(main_node[-1], translations, connectors) + "."
        )
    else:
        prompt += "."

    # Remove leading "then" if present
    if prompt.lower().startswith("then "):
        prompt = prompt[5:]

    return prompt


def generate_spoon_prompt(
    tree_path: str, spoon_translations: Dict[str, str], connectors: Dict[str, str]
) -> str:
    """Generate a spoonfed-style prompt from a behavior tree XML file."""
    tree = ET.parse(tree_path)
    root = tree.getroot()

    if root.tag != "BehaviorTree" or len(root) != 1:
        raise ValueError("Invalid BehaviorTree structure")

    main_node = root[0]
    prompt = translate_node(main_node, spoon_translations, connectors)

    # Add "otherwise" for Selectors for better understanding of if-else relationships
    if main_node.tag == "Selector" and len(main_node) > 1:
        prompt += (
            "otherwise "
            + translate_node(main_node[-1], spoon_translations, connectors)
            + "."
        )
    else:
        prompt += "."

    # Remove leading "then" if present
    if prompt.lower().startswith("then "):
        prompt = prompt[5:]

    return prompt


# Backward compatibility functions that use default translations
def generate_prompts_from_folder(folder_path: str) -> None:
    """Process all XML files in the folder and generate prompts using default translations."""
    from .node_translations import node_connectors, node_translations

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xml"):
            file_path = os.path.join(folder_path, file_name)
            try:
                print(f"Processing file: {file_name}")
                prompt = generate_tech_prompt(
                    file_path, node_translations, node_connectors
                )
                print(f"Generated Prompt:\n{prompt}\n")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    folder_path = "./data_generation_grammar/example_trees"
    generate_prompts_from_folder(folder_path)
