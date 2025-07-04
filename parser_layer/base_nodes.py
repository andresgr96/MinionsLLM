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
        """Initialize sequence node with child nodes."""
        self.children: List[BaseNode] = children

    def run(self, agent: Agent) -> bool:
        """Execute all child nodes in sequence, returning False if any fail."""
        for child in self.children:
            if not child.run(agent):
                return False
        return True


class SelectorNode(BaseNode):
    """Selector node that executes children until one succeeds."""

    def __init__(self, children: List["BaseNode"]):
        """Initialize selector node with child nodes."""
        self.children: List[BaseNode] = children

    def run(self, agent: Agent) -> bool:
        """Execute child nodes until one succeeds, returning True if any succeed."""
        for child in self.children:
            if child.run(agent):
                return True
        return False


class ConditionNode(BaseNode):
    """Condition node that evaluates a boolean condition."""

    def __init__(self, condition_name: str):
        """Initialize condition node with a condition function."""
        self.condition_name: str = condition_name

    def run(self, agent: Agent) -> bool:
        """Execute the condition function and return its result."""
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
        """Initialize actuator action node with an action function."""
        self.action_name: str = action_name

    def run(self, agent: Agent) -> bool:
        """Execute the actuator action and return its result."""
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
        """Initialize state action node with a state function."""
        self.action_name: str = action_name

    def run(self, agent: Agent) -> bool:
        """Execute the state action and return its result."""
        action_func = getattr(agent, self.action_name, None)
        if not action_func:
            raise ValueError(
                f"Action function '{self.action_name}' not found in the agent class."
            )
        # print(f"Executing action {self.action_name}")
        return bool(action_func())
