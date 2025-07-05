"""Validation utilities for behavior tree primitives."""

import xml.etree.ElementTree as ET
from typing import List, Type

from vi import Agent

from .agent_doc_parser import AgentDocstringParser


def _validate_node_recursive(
    node: ET.Element,
    conditions: List[str],
    actuator_actions: List[str],
    state_actions: List[str],
    feedback: List[str],
    path: str,
) -> bool:
    """
    Recursively validate a node and its children against lists of valid primitives.

    Args:
        node: The XML element node to validate
        conditions: List of valid condition primitives
        actuator_actions: List of valid actuator action primitives
        state_actions: List of valid state action primitives
        feedback: List to collect validation feedback messages
        path: Current path in the XML tree for error reporting

    Returns:
        bool: True if validation passes, False otherwise
    """
    current_path = f"{path}/{node.tag}" if path else node.tag

    if node.text and node.text.strip():
        content = node.text.strip()

        primitive_map = {
            "Condition": (conditions, "condition"),
            "ActuatorAction": (actuator_actions, "actuator action"),
            "StateAction": (state_actions, "state action"),
        }

        if node.tag in primitive_map:
            valid_primitives, node_type = primitive_map[node.tag]
            if content not in valid_primitives:
                feedback.append(
                    f"Invalid primitive at {current_path}: '{content}' is not a valid {node_type}."
                )
                return False

    for child in node:
        if not _validate_node_recursive(
            child, conditions, actuator_actions, state_actions, feedback, current_path
        ):
            return False

    return True


def validate_primitives(tree_xml: str, agent_class: Type[Agent]) -> tuple[bool, str]:
    """
    Validate that the primitives in a behavior tree are available to the agent.

    This function traverses a behavior tree and checks if each primitive (Condition,
    ActuatorAction, StateAction) corresponds to a method in the provided agent class.

    Args:
        tree_xml: The behavior tree XML string.
        agent_class: The agent class to validate against.

    Returns:
        tuple[bool, str]: A tuple containing validation result and feedback message
    """
    feedback: List[str] = []
    try:
        parser = AgentDocstringParser(agent_class)
        config = parser.extract_docstring_config()
        conditions = config["conditions"]
        actuator_actions = config["actuator_actions"]
        state_actions = config["state_actions"]

        root = ET.fromstring(tree_xml)
        is_valid = _validate_node_recursive(
            root, conditions, actuator_actions, state_actions, feedback, ""
        )
        return is_valid, "\n".join(feedback)
    except ET.ParseError as e:
        return (
            False,
            f"XML parsing error at line {e.position[0]}, column {e.position[1]}: {str(e)}",
        )
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"
