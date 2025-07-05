"""Middle-layer parser for behavior tree XML processing and node manipulation."""

import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

from .base_nodes import (
    ActuatorActionNode,
    BaseNode,
    ConditionNode,
    SelectorNode,
    SequenceNode,
    StateActionNode,
)


class UnknownNodeTypeError(Exception):
    """Raised when an unknown node type is encountered in the behavior tree."""

    pass


def parse_behavior_tree(file_path: str) -> BaseNode:
    """
    Parse a behavior tree XML file and return a list of nodes.

    Args:
        file_path: Path to the XML file to parse

    Returns:
        BaseNode: The parsed behavior tree root node
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    return parse_node(root)


def parse_behavior_tree_with_metadata(
    file_path: str,
) -> Tuple[Optional[BaseNode], Dict[str, str]]:
    """
    Parse the XML file containing the behavior tree and metadata.

    Args:
        file_path: Path to the XML file

    Returns:
        Tuple[Optional[BaseNode], Dict[str, str]]: The parsed behavior tree and a dictionary containing metadata
    """
    with open(file_path, "r") as file:
        content = file.read()

    # Split the content between the behavior tree and the metadata
    behavior_tree_str, _, metadata_str = content.partition("</BehaviorTree>")
    behavior_tree_str += (
        "</BehaviorTree>"  # Add the tag back to the behavior tree string
    )

    # Parse the behavior tree
    try:
        root = ET.fromstring(behavior_tree_str)
        behavior_tree = parse_node(root)
    except ET.ParseError as e:
        print(f"Error parsing behavior tree in {file_path}: {e}")
        behavior_tree = None  # Mark the behavior tree as None if parsing fails

    # Parse the metadata
    metadata: Dict[str, str] = {}
    if metadata_str.strip():
        # Parse the metadata elements manually
        metadata_lines = metadata_str.strip().splitlines()
        for line in metadata_lines:
            line = line.strip()
            if line.startswith("<") and line.endswith(">"):
                tag = line[1 : line.find(">")]
                value = line[line.find(">") + 1 : line.rfind("<")].strip()
                metadata[tag] = value

    return behavior_tree, metadata


def parse_node(node: ET.Element) -> BaseNode:
    """
    Parse an individual XML node and return its data structure.

    Args:
        node: The XML element to parse

    Returns:
        BaseNode: The parsed node structure

    Raises:
        UnknownNodeTypeError: If an unknown node type is encountered
        ValueError: If BehaviorTree node structure is invalid
    """
    node_type = node.tag.lower()

    if node_type == "condition":
        condition_name = node.text.strip() if node.text else ""
        return ConditionNode(condition_name)
    elif node_type == "behaviortree":
        if len(node) != 1:
            raise ValueError("BehaviorTree should have exactly one top-level node.")
        top_level_node = parse_node(node[0])
        return top_level_node
    elif node_type == "actuatoraction":
        action_name = node.text.strip() if node.text else ""
        return ActuatorActionNode(action_name)
    elif node_type == "stateaction":
        action_name = node.text.strip() if node.text else ""
        return StateActionNode(action_name)
    elif node_type == "sequence":
        children = [parse_node(child) for child in node]
        return SequenceNode(children)
    elif node_type == "selector":
        children = [parse_node(child) for child in node]
        return SelectorNode(children)
    else:
        # Instead of returning None, raise our custom exception
        raise UnknownNodeTypeError(
            f"Hallucination Error. Unknown node type: {node_type}"
        )


def print_nodes(node: BaseNode, indent: int = 0) -> None:
    """
    Print the parsed nodes in a readable format.

    Args:
        node: The node to print
        indent: Indentation level for formatting

    Raises:
        ValueError: If unknown node type is encountered
    """
    indent_str = "  " * indent

    if isinstance(node, SequenceNode):
        print(f"{indent_str}SequenceNode:")
        for child in node.children:
            print_nodes(child, indent + 1)
    elif isinstance(node, SelectorNode):
        print(f"{indent_str}SelectorNode:")
        for child in node.children:
            print_nodes(child, indent + 1)
    elif isinstance(node, ConditionNode):
        print(f"{indent_str}ConditionNode: {node.condition_name}")
    elif isinstance(node, ActuatorActionNode):
        print(f"{indent_str}ActuatorActionNode: {node.action_name}")
    elif isinstance(node, StateActionNode):
        print(f"{indent_str}StateActionNode: {node.action_name}")
    else:
        raise ValueError(f"Unknown node type: {type(node).__name__}")


def save_behavior_tree_xml(data: str, file_path: str) -> None:
    """
    Save parsed nodes back to an XML file.

    Args:
        data: The XML data string to save
        file_path: Path where to save the XML file
    """
    if not data:
        print("No valid XML content to save.")
        return
    try:
        root = ET.fromstring(data)
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        print(f"Data: {data}")


def save_behavior_tree_with_metadata(
    data: str,
    behaviors: Dict[str, str],
    agent_class_name: str,
    env_name: str,
    task_name: str,
    model_name: str,
    technique: str,
    style_name: str,
    file_path: str,
) -> None:
    """
    Save behavior tree data with metadata to an XML file.

    Args:
        data: The XML data string to save
        behaviors: Dictionary of behavior names
        agent_class_name: Name of the agent class
        env_name: Name of the environment
        task_name: Name of the task
        model_name: Name of the model
        technique: Prompt technique used
        style_name: Style name used
        file_path: Path where to save the XML file
    """
    if not data:
        print("No valid XML content to save.")
        return

    behavior_names = list(behaviors.keys())
    behaviors_str = ", ".join(behavior_names)

    try:
        root = ET.fromstring(data)
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        print(f"Data: {data}")

        dummy_xml = '<?xml version="1.0" encoding="utf-8"?>\n<BehaviorTree>syntactically_incorrect</BehaviorTree>'
        with open(file_path, "w") as file:
            file.write(dummy_xml)

    # Always append metadata, regardless of whether the XML was valid so we can compute metrics
    with open(file_path, "a") as file:
        file.write("\n\n<!-- Metadata -->\n")
        file.write(f"<ModelName>{model_name}</ModelName>\n")
        file.write(f"<EnvironmentName>{env_name}</EnvironmentName>\n")
        file.write(f"<Task>{task_name}</Task>\n")
        file.write(f"<PromptTechnique>{technique}</PromptTechnique>\n")
        file.write(f"<PromptStyle>{style_name}</PromptStyle>\n")
        file.write(f"<AgentClass>{agent_class_name}</AgentClass>\n")
        file.write(f"<Behaviors>{behaviors_str}</Behaviors>\n")
