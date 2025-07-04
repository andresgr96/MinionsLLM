"""
Agent Docstring Parser that automatically generates grammar configuration from agent classes.
"""

import re
import warnings
from typing import Any, Dict, List, Type

from vi import Agent


class AgentDocstringParser:
    """
    Parses agent class methods' docstrings to extract node information like type and translations.

    This class is used to automatically generate configuration for behavior tree nodes
    by inspecting the docstrings of methods in an agent class. For a method to be
    parsed correctly, its docstring must follow a specific format.

    Example of a correctly formatted docstring for an agent method:
    .. code-block:: python

        def is_agent_in_base_area(self) -> bool:
            \"\"\"
            Node Type: Condition
            Description: Checks whether the agent is in the base area. Returns True if the agent is within the base, and False otherwise.
            Translation: you are in the base
            Spoon Translation: you are in the base area
            \"\"\"
            if self.is_agent_in_base_flag:
                return True

            return False
    """

    def __init__(self, agent_class: Type[Agent]):
        """
        Initialize the docstring parser with an agent class.

        Args:
            agent_class: The agent class to parse docstrings from.
        """
        self.agent_class = agent_class
        self.conditions: List[str] = []
        self.actuator_actions: List[str] = []
        self.state_actions: List[str] = []
        self.node_translations: Dict[str, str] = {}
        self.spoon_node_translations: Dict[str, str] = {}

    def extract_docstring_config(self) -> Dict[str, Any]:
        """
        Extract complete docstring configuration from the agent class.

        Returns:
            Dictionary containing all docstring configuration
        """
        self._parse_agent_methods()

        return {
            "conditions": self.conditions,
            "actuator_actions": self.actuator_actions,
            "state_actions": self.state_actions,
            "node_translations": self.node_translations,
            "spoon_node_translations": self.spoon_node_translations,
        }

    def _parse_agent_methods(self) -> None:
        """
        Parse all methods in the agent class and categorize them by node type.
        """
        for method_name in dir(self.agent_class):
            if self._should_include_method(method_name):
                method = getattr(self.agent_class, method_name)
                if callable(method) and hasattr(method, "__doc__") and method.__doc__:
                    self._parse_method_docstring(method_name, method.__doc__)

    def _should_include_method(self, method_name: str) -> bool:
        """
        Determine if a method should be included in the docstring extraction.

        Args:
            method_name: Name of the method

        Returns:
            True if the method should be included
        """
        # Same filtering logic as in layer_LLM.py
        if (
            callable(getattr(self.agent_class, method_name))
            and not method_name.startswith("__")
            and not method_name.startswith("update")
            and not method_name.startswith("obstacle")
            and not method_name.startswith("helper")
        ):

            method = getattr(self.agent_class, method_name)
            if hasattr(method, "__qualname__"):
                return bool(
                    method.__qualname__.startswith(self.agent_class.__name__ + ".")
                )

        return False

    def _parse_method_docstring(self, method_name: str, docstring: str) -> None:
        """
        Parse a method's docstring to extract node information.

        Args:
            method_name: Name of the method
            docstring: The method's docstring
        """
        # Extract node type
        node_type = self._extract_field(docstring, "Node Type")

        # Extract translations
        translation = self._extract_field(docstring, "Translation")
        spoon_translation = self._extract_field(docstring, "Spoon Translation")

        # Warn if translations are missing
        if not translation:
            warnings.warn(
                f"'Translation' not found in docstring for method '{method_name}'. If using this for dataset generation, you will need to provide a translation for this method, otherwise ignore this warning."
            )
        if not spoon_translation:
            warnings.warn(
                f"'Spoon Translation' not found in docstring for method '{method_name}'. If using this for dataset generation, you will need to provide a spoon translation for this method, otherwise ignore this warning."
            )

        # Categorize by node type
        if node_type.lower() == "condition":
            self.conditions.append(method_name)
        elif node_type.lower() == "actuatoraction":
            self.actuator_actions.append(method_name)
        elif node_type.lower() == "stateaction":
            self.state_actions.append(method_name)

        # Store translations
        self.node_translations[method_name] = translation
        self.spoon_node_translations[method_name] = spoon_translation

    def _extract_field(self, docstring: str, field_name: str) -> str:
        """
        Extract a specific field from a docstring.

        Args:
            docstring: The docstring to parse
            field_name: The field name to extract

        Returns:
            The field value or empty string if not found
        """
        pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, docstring, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else ""
