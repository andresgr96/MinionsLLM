"""Class to build system prompt for the RLHF agent."""

import os
from typing import Dict, Type

from vi import Agent


class PromptBuilder:
    """Builds system prompts for RLHF agents using XML templates and agent class behaviors."""

    def __init__(self, agent_class: Type[Agent]) -> None:
        """
        Initialize the PromptBuilder with an agent class.

        Args:
            agent_class: The agent class to extract behaviors from
        """
        self.agent_class = agent_class

    def load_xml_template(self, file_name: str) -> str:
        """
        Load an XML template from the prompt_techniques directory.

        Args:
            file_name: The name of the template file

        Returns:
            The contents of the template file
        """
        directory = "llm_interface/prompt_techniques"
        file_path = os.path.join(directory, file_name)
        with open(file_path, "r") as file:
            return file.read()

    def extract_agent_behaviors(self) -> Dict[str, str]:
        """
        Get all the behaviors from the agent class, extracting only Node Type and Description.

        Returns:
            Dictionary of function names and their processed docstrings (Node Type and Description only)
        """
        class_obj = self.agent_class
        function_names_and_docstrings = {}

        for func_name in dir(class_obj):
            if (
                callable(getattr(class_obj, func_name))
                and not func_name.startswith("__")
                and not func_name.startswith("update")
                and not func_name.startswith("helper")
                and getattr(class_obj, func_name).__qualname__.startswith(
                    class_obj.__name__ + "."
                )
            ):
                func = getattr(class_obj, func_name)
                if func.__doc__:
                    # Split docstring into lines and process
                    doc_lines = func.__doc__.strip().split("\n")
                    processed_doc = []

                    for line in doc_lines:
                        line = line.strip()
                        if line.startswith("Node Type:") or line.startswith(
                            "Description:"
                        ):
                            processed_doc.append(line)

                    # Join the processed lines back together
                    function_names_and_docstrings[func_name] = "\n".join(processed_doc)
                else:
                    function_names_and_docstrings[func_name] = "No docstring found."

        return function_names_and_docstrings

    def build_system_prompt(self) -> str:
        """
        Build the complete system prompt by loading the XML template and replacing placeholders.

        Returns:
            The complete system prompt with all placeholders replaced
        """
        # Load and prepare system prompt
        system_prompt = self.load_xml_template("system_prompt.xml")
        bt_3_example = self.load_xml_template("bt_3.xml")
        behaviors = self.extract_agent_behaviors()

        # Format behaviors dictionary to avoid template variable conflicts
        # Convert to a safe string representation and escape all curly braces
        behaviors_str = "{{\n"  # Start with escaped curly brace
        for key, value in behaviors.items():
            # Escape any curly braces in the content and format safely
            safe_value = value.replace("{", "{{").replace("}", "}}")
            safe_key = key.replace("{", "{{").replace("}", "}}")
            behaviors_str += f'    "{safe_key}": "{safe_value}",\n'
        behaviors_str = (
            behaviors_str.rstrip(",\n") + "\n}}"
        )  # End with escaped curly brace

        # Replace placeholders in system prompt
        system_prompt = system_prompt.replace("{bt_3.xml}", bt_3_example)
        system_prompt = system_prompt.replace("{BEHAVIORS}", behaviors_str)

        return system_prompt
