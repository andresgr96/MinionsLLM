import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional, Any, Union
from .int_to_list import *

def list_to_xml(node_list: List[Any], placeholders: bool, conditions: Optional[List[str]] = None, actuator_actions: Optional[List[str]] = None, state_actions: Optional[List[str]] = None) -> ET.Element:
    """Converts the nested list representation of a behavior tree into XML."""
    # print(f"Processing root node: {node_list[0]}")
    # Root node is always the first 
    root = ET.Element(node_list[0])
    child_list = node_list[1]
    if isinstance(child_list, list) and len(child_list) == 2:
        child_node = process_node(child_list, placeholders, conditions, actuator_actions, state_actions)
        root.append(child_node)
    else:
        # print("Error: Root node does not have a single child.")
        raise ValueError(f"Root node does not have a single child.")
    return root


def process_node(node_list: List[Any], placeholders: bool, conditions: Optional[List[str]] = None, actuator_actions: Optional[List[str]] = None, state_actions: Optional[List[str]] = None) -> ET.Element:
    """Recursively processes non-root nodes into XML."""
    # print(f"Processing node: {node_list[0]}")
    node = ET.Element(node_list[0])
    children = node_list[1]

    for child in children:
        if isinstance(child, list):  # Handle nested nodes
            if child[0] == "Sequence":
                sequence_node = create_sequence_node(child[1], placeholders, conditions, actuator_actions, state_actions)
                node.append(sequence_node)
            elif child[0] == "Selector":
                selector_node = create_selector_node(child[1], placeholders, conditions, actuator_actions, state_actions)
                node.append(selector_node)
            else:
                # print(f"Unexpected nested node: {child}")
                nested_node = process_node(child, placeholders, conditions, actuator_actions, state_actions)
                node.append(nested_node)
        elif isinstance(child, str):  # Terminal node
            # print(f"Adding terminal node to {node_list[0]}: {child}")
            terminal_node = ET.SubElement(node, child)
            terminal_node.text = assign_terminal_text(child, placeholders, conditions, actuator_actions, state_actions)

    return node


def create_sequence_node(sequence_children: List[Any], placeholders: bool, conditions: Optional[List[str]] = None, actuator_actions: Optional[List[str]] = None, state_actions: Optional[List[str]] = None) -> ET.Element:
    """Creates an XML node for a Sequence with its children."""
    # print(f"Creating Sequence node with children: {sequence_children}")
    sequence_node = ET.Element("Sequence")
    for child in sequence_children:
        if isinstance(child, str):  # Terminal
            # print(f"Adding terminal node to Sequence: {child}")
            terminal_node = ET.SubElement(sequence_node, child)
            terminal_node.text = assign_terminal_text(child, placeholders, conditions, actuator_actions, state_actions)
        elif isinstance(child, list):  # Nested
            nested_node = process_node(child, placeholders, conditions, actuator_actions, state_actions)
            sequence_node.append(nested_node)
        else:
            raise ValueError(f"Unexpected child in Sequence: {child}")
    return sequence_node


def create_selector_node(selector_children: List[Any], placeholders: bool, conditions: Optional[List[str]] = None, actuator_actions: Optional[List[str]] = None, state_actions: Optional[List[str]] = None) -> ET.Element:
    """Creates an XML node for a Selector with its children."""
    # print(f"Creating Selector node with children: {selector_children}")
    selector_node = ET.Element("Selector")
    for child in selector_children:
        if isinstance(child, list) and child[0] == "Sequence":
            sequence_node = create_sequence_node(child[1], placeholders, conditions, actuator_actions, state_actions)
            selector_node.append(sequence_node)
        elif isinstance(child, str):  # Terminal in Selector
            # print(f"Adding terminal to Selector: {child}")
            terminal_node = ET.SubElement(selector_node, child)
            terminal_node.text = assign_terminal_text(child, placeholders, conditions, actuator_actions, state_actions)
        else:
            raise ValueError(f"Unexpected child in Selector: {child}")
    return selector_node


def assign_terminal_text(node_type: str, placeholders: bool, conditions: Optional[List[str]] = None, actuator_actions: Optional[List[str]] = None, state_actions: Optional[List[str]] = None) -> str:
    """Assigns appropriate text to terminal nodes."""
    if node_type == "Condition":
        return random.choice(conditions) if not placeholders and conditions else "placeholder_condition"
    elif node_type == "ActuatorAction":
        return random.choice(actuator_actions) if not placeholders and actuator_actions else "placeholder_actuator_action"
    elif node_type == "StateAction":
        return random.choice(state_actions) if not placeholders and state_actions else "placeholder_state_action"
    else:
        raise ValueError(f"Unexpected node type: {node_type}")
    


def pretty_print_xml(element: ET.Element) -> str:
    """Returns a pretty-printed XML string for an ElementTree element."""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")



