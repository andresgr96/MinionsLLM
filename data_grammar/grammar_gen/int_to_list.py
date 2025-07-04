import random
import sys
from typing import Any, Dict, List, Optional, Union

sys.tracebacklimit = 1


class BaseNode:
    """Base class for all grammar nodes."""

    def __init__(
        self,
        integers: List[int],
        index: List[int],
        recursion_depth: Optional[Dict[str, Any]] = None,
        grammar_rules: Optional[Dict[str, List[str]]] = None,
        grammar_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ):

        self.integers = integers
        self.index = index
        self.recursion_depth = recursion_depth if recursion_depth else {}
        self.grammar_rules = grammar_rules if grammar_rules else {}
        self.grammar_parameters = grammar_parameters if grammar_parameters else {}

    def expand_symbol(self, symbol: str) -> Union[str, List[Any]]:
        """Helper method to automatically instantiate and expand a node for a given symbol."""
        symbol_to_class = {
            "SEL": SELNode,
            "SEQ": SEQNode,
            "SEQn": SEQNNode,
            "Pn": PNode,
            "A": ANode,
            "As": ASNode,
            "b": BTerminalNode,
            "aa": ActuatorActionNode,
            "sa": StateActionNode,
        }

        if symbol in symbol_to_class:
            node_class = symbol_to_class[symbol]
            return node_class(
                self.integers,
                self.index,
                self.recursion_depth,
                self.grammar_rules,
                self.grammar_parameters,
            ).expand()
        else:
            return symbol  # Return the symbol as-is if it's not in the mapping

    def choose_option(self, options: List[Any], rule_name: str) -> Any:
        """Choose an option based on the integer sequence and grammar parameters."""
        params = self.grammar_parameters.get(rule_name, {})
        current_depth = self.recursion_depth.get(rule_name, 0)

        # Handle "exclude" parameter - filter out excluded indices
        available_indices = list(range(len(options)))
        if "exclude" in params:
            # print(f"Original Available Indices: {available_indices}")
            # print(f"Excluding indices: {params['exclude']}")
            exclude_list = params["exclude"]
            available_indices = [i for i in available_indices if i not in exclude_list]
            # print(f"Available Indices after exclusion: {available_indices}")
            if not available_indices:
                raise ValueError(
                    f"All options excluded for rule '{rule_name}'. Available options: {len(options)}, Excluded: {exclude_list}"
                )

        chosen_index = self.integers[self.index[0]] % len(available_indices)
        chosen_index = available_indices[chosen_index]  # Map back to original index

        # Handle "only" parameter
        if "only" in params:
            chosen_index = params["only"]

        # Handle "parent" parameter
        if "parent" in params and current_depth == 1:
            parent = self.recursion_depth.get("parent", None)
            if parent and parent in params["parent"]:
                forced_choice = params["parent"][parent]
                chosen_index = forced_choice
                # print(f"Forced Choice: {forced_choice}")

        # Handle "max" parameter for recursive rules
        if "list_max" in params:
            max_depth = params["list_max"]
            if current_depth == max_depth:
                # print("Recursion Stopped")
                chosen_index = 1  # Force the single version at index 1

        # Handle "list_always" parameter for recursive rules
        if "list_always" in params:
            always_depth = params["list_always"]
            if current_depth < always_depth:
                # Force recursive option to continue expanding
                chosen_index = 0  # Force the recursive version at index 0
            elif current_depth == always_depth:
                # Force terminal option to stop exactly at the specified depth
                chosen_index = 1  # Force the single version at index 1

        # print(f"-----Current Integer: {self.integers[self.index[0]]},  Num Options: {len(options)}, Mod: {chosen_index}----")
        self.index[0] = (self.index[0] + 1) % len(self.integers)
        option = options[chosen_index]
        # print(f"Options: {options}")
        # print(f"Chosen Option: {option}")

        return option

    def expand(self) -> Union[str, List[Any]]:
        """Expand the node. This should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the expand method.")


class BNode(BaseNode):
    """Node for the 'B' rule."""

    def expand(self) -> List[Any]:
        options = self.grammar_rules["B"]
        chosen_option = self.choose_option(options, "B")

        # chosen_option will be either ["b", ["SEL"]] or ["b", ["SEQ"]]
        terminal_symbol = chosen_option[0]  # This is "b"
        children = chosen_option[1]  # This is ["SEL"] or ["SEQ"]

        # Expand the terminal "b" to get the node structure
        terminal_expansion = self.expand_symbol(terminal_symbol)

        # Expand the children
        expanded_children = []
        for child in children:
            expanded_child = self.expand_symbol(child)
            expanded_children.append(expanded_child)

        # Replace the placeholder children_nodes with actual expanded children
        if (
            isinstance(terminal_expansion, list)
            and terminal_expansion[0] == "BehaviorTree"
        ):
            return [terminal_expansion[0], expanded_children[0]]

        # Cast to List[Any] to satisfy type checker
        if isinstance(terminal_expansion, list):
            return terminal_expansion
        else:
            return [terminal_expansion]

    def expand_child(self, child: str) -> Union[str, List[Any]]:
        """Expand a child node for B rule."""
        return self.expand_symbol(child)


class SELNode(BaseNode):
    """Node for the 'SEL' rule."""

    def expand(self) -> List[Any]:
        options = self.grammar_rules["SEL"]
        chosen_option = self.choose_option(options, "SEL")
        if chosen_option[0] == "sel":
            child_nodes = []
            for child in chosen_option[1]:
                expanded_child = self.expand_child(child)
                if expanded_child:
                    if (
                        isinstance(expanded_child, list)
                        and len(expanded_child) > 0
                        and isinstance(expanded_child[0], list)
                    ):
                        # Ensure that nested lists are treated as distinct options
                        child_nodes.extend(expanded_child)
                    else:
                        child_nodes.append(expanded_child)
            return ["Selector", child_nodes]
        # Add default return for when chosen_option[0] is not "sel"
        return ["Selector", []]

    def expand_child(self, child: str) -> Union[str, List[Any]]:
        """Expand a child node."""
        self.recursion_depth["parent"] = "SEL"
        return self.expand_symbol(child)


class SEQNNode(BaseNode):
    """Node for the 'SEQn' rule."""

    def expand(self) -> List[Any]:
        self.recursion_depth["SEQn"] = self.recursion_depth.get("SEQn", 0) + 1
        options = self.grammar_rules["SEQn"]
        chosen_option = self.choose_option(options, "SEQn")

        child_nodes: List[Any] = []
        for child in chosen_option:
            expanded_child = self.expand_child(child)
            if expanded_child:
                if (
                    isinstance(expanded_child, list)
                    and len(expanded_child) > 0
                    and expanded_child[0] == "Sequence"
                ):
                    # Preserve sequences as individual entries
                    child_nodes.append(expanded_child)
                else:
                    # Ensure we extend with a list
                    if isinstance(expanded_child, list):
                        child_nodes.extend(expanded_child)
                    else:
                        child_nodes.append(expanded_child)

        self.recursion_depth["SEQn"] -= 1
        return child_nodes

    def expand_child(self, child: str) -> Union[str, List[Any]]:
        return self.expand_symbol(child)


class SEQNode(BaseNode):
    """Node for the 'SEQ' rule."""

    def expand(self) -> List[Any]:
        options = self.grammar_rules["SEQ"]
        chosen_option = self.choose_option(options, "SEQ")
        child_nodes = []
        for child in chosen_option[1]:
            expanded_child = self.expand_child(child)
            if expanded_child:
                if isinstance(expanded_child, list):  # If the child is a list, extend
                    child_nodes.extend(expanded_child)
                else:  # If the child is a terminal string, append
                    child_nodes.append(expanded_child)
        return ["Sequence", child_nodes]

    def expand_child(self, child: str) -> Union[str, List[Any]]:
        self.recursion_depth["parent"] = "SEQ"
        return self.expand_symbol(child)


## ----------------------------------------------------- Terminals -----------------------------------------------------


class PNode(BaseNode):
    """Node for the 'Pn' rule."""

    def expand(self) -> List[Any]:
        self.recursion_depth["Pn"] = self.recursion_depth.get("Pn", 0) + 1
        options = self.grammar_rules["Pn"]
        chosen_option = self.choose_option(options, "Pn")

        child_nodes = []
        for child in chosen_option:
            expanded_child = self.expand_child(child)
            if expanded_child:  # Skip None or empty results
                if isinstance(expanded_child, list):
                    child_nodes.extend(expanded_child)
                else:
                    child_nodes.append(expanded_child)

        self.recursion_depth["Pn"] -= 1
        return child_nodes

    def expand_child(self, child: str) -> Optional[Union[str, List[Any]]]:
        if child == "p":  # Terminal case
            return "Condition"
        elif not child:  # Epsilon rule
            return None
        else:
            result = self.expand_symbol(child)
            # expand_symbol can return str or List[Any], both are valid for this method
            return result


class ANode(BaseNode):
    """Node for the 'A' rule."""

    def expand(self) -> Union[str, List[Any]]:
        options = self.grammar_rules["A"]
        chosen_option = self.choose_option(options, "A")
        child_nodes = []
        for child in chosen_option:
            expanded_child = self.expand_symbol(child)
            child_nodes.append(expanded_child)

        return child_nodes if len(child_nodes) > 1 else child_nodes[0]


class ASNode(BaseNode):
    """Node for the 'As' rule."""

    def expand(self) -> Union[str, List[Any]]:
        options = self.grammar_rules["As"]
        chosen_option = self.choose_option(options, "As")
        child_nodes = []
        for child in chosen_option:
            expanded_child = self.expand_symbol(child)
            child_nodes.append(expanded_child)

        return child_nodes if len(child_nodes) > 1 else child_nodes[0]


class ActuatorActionNode(BaseNode):
    """Terminal node for Actuator Actions."""

    def expand(self) -> str:
        return random.choice(["ActuatorAction"])


class StateActionNode(BaseNode):
    """Terminal node for State Actions."""

    def expand(self) -> str:
        return random.choice(["StateAction"])


class BTerminalNode(BaseNode):
    """Terminal node for BehaviorTree."""

    def expand(self) -> List[Any]:
        return ["BehaviorTree", ["children_nodes"]]


# Main driver
def generate_nested_list(
    integer_list: List[int],
    grammar_rules: Dict[str, List[str]],
    grammar_parameters: Dict[str, Dict[str, Any]],
) -> List[Any]:
    """Generates a nested list for the behavior tree."""
    root = BNode(
        integer_list,
        index=[0],
        recursion_depth={},
        grammar_rules=grammar_rules,
        grammar_parameters=grammar_parameters,
    )  # Shared index as a list
    return root.expand()
