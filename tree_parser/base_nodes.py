"""Base node classes for behavior tree parsing and execution."""

from abc import ABC, abstractmethod
from typing import List

from vi import Agent


class BaseNode(ABC):
    """Abstract base class for all behavior tree nodes."""

    @abstractmethod
    def run(self, agent: Agent) -> bool:
        """
        Execute the node's behavior.

        Args:
            agent: The agent executing this node

        Returns:
            bool: True if the node succeeds, False if it fails

        Raises:
            Exception: May raise exceptions for error conditions
        """
        pass


class SequenceNode(BaseNode):
    """Sequence node that executes children in order until one fails."""

    def __init__(self, children: List["BaseNode"]):
        """
        Initialize sequence node with child nodes.

        Args:
            children: List of child nodes to execute in sequence
        """
        self.children: List[BaseNode] = children

    def run(self, agent: Agent) -> bool:
        """
        Execute all child nodes in sequence, returning False if any fail.

        Args:
            agent: The agent executing this node

        Returns:
            bool: True if all children succeed, False if any fail
        """
        for child in self.children:
            if not child.run(agent):
                return False
        return True


class SelectorNode(BaseNode):
    """Selector node that executes children until one succeeds."""

    def __init__(self, children: List["BaseNode"]):
        """
        Initialize selector node with child nodes.

        Args:
            children: List of child nodes to try in order
        """
        self.children: List[BaseNode] = children

    def run(self, agent: Agent) -> bool:
        """
        Execute child nodes until one succeeds, returning True if any succeed.

        Args:
            agent: The agent executing this node

        Returns:
            bool: True if any child succeeds, False if all fail
        """
        for child in self.children:
            if child.run(agent):
                return True
        return False


class ConditionNode(BaseNode):
    """Condition node that evaluates a boolean condition."""

    def __init__(self, condition_name: str):
        """
        Initialize condition node with a condition function.

        Args:
            condition_name: Name of the condition method to call on the agent
        """
        self.condition_name: str = condition_name

    def run(self, agent: Agent) -> bool:
        """
        Execute the condition function and return its result.

        Args:
            agent: The agent executing this node

        Returns:
            bool: Result of the condition evaluation

        Raises:
            ValueError: If condition function is not found in agent class
        """
        condition_func = getattr(agent, self.condition_name, None)
        if not condition_func:
            raise ValueError(
                f"Condition function '{self.condition_name}' not found in the agent class."
            )
        result = condition_func()
        # print(f"Condition {self.condition_name} returned {result}")
        return bool(result)


class ActuatorActionNode(BaseNode):
    """Actuator action node that performs physical actions."""

    def __init__(self, action_name: str):
        """
        Initialize actuator action node with an action function.

        Args:
            action_name: Name of the action method to call on the agent
        """
        self.action_name: str = action_name

    def run(self, agent: Agent) -> bool:
        """
        Execute the actuator action and return its result.

        Args:
            agent: The agent executing this node

        Returns:
            bool: Result of the action execution

        Raises:
            ValueError: If action function is not found in agent class
        """
        action_func = getattr(agent, self.action_name, None)
        if not action_func:
            raise ValueError(
                f"Action function '{self.action_name}' not found in the agent class."
            )
        # print(f"Executing action {self.action_name}")
        return bool(action_func())


class StateActionNode(BaseNode):
    """State action node that modifies internal agent state."""

    def __init__(self, action_name: str):
        """
        Initialize state action node with a state function.

        Args:
            action_name: Name of the state action method to call on the agent
        """
        self.action_name: str = action_name

    def run(self, agent: Agent) -> bool:
        """
        Execute the state action and return its result.

        Args:
            agent: The agent executing this node

        Returns:
            bool: Result of the state action execution

        Raises:
            ValueError: If action function is not found in agent class
        """
        action_func = getattr(agent, self.action_name, None)
        if not action_func:
            raise ValueError(
                f"Action function '{self.action_name}' not found in the agent class."
            )
        # print(f"Executing action {self.action_name}")
        return bool(action_func())
