import xml.etree.ElementTree as ET
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
    def __init__(self, children: List[BaseNode]):
        self.children: List[BaseNode] = children

    def run(self, agent: Agent) -> bool:
        for child in self.children:
            if not child.run(agent):
                return False
        return True

class SelectorNode(BaseNode):
    def __init__(self, children: List[BaseNode]):
        self.children: List[BaseNode] = children

    def run(self, agent: Agent) -> bool:
        for child in self.children:
            if child.run(agent):
                return True
        return False

class ConditionNode(BaseNode):
    def __init__(self, condition_name: str):
        self.condition_name: str = condition_name

    def run(self, agent: Agent) -> bool:
        condition_func = getattr(agent, self.condition_name, None)
        if not condition_func:
            raise ValueError(f"Condition function '{self.condition_name}' not found in the agent class.")
        result = condition_func()
        # print(f"Condition {self.condition_name} returned {result}")
        return result

class ActuatorActionNode(BaseNode):
    def __init__(self, action_name: str):
        self.action_name: str = action_name

    def run(self, agent: Agent) -> bool:
        action_func = getattr(agent, self.action_name, None)
        if not action_func:
            raise ValueError(f"Action function '{self.action_name}' not found in the agent class.")
        # print(f"Executing action {self.action_name}")
        return action_func()
    
class StateActionNode(BaseNode):
    def __init__(self, action_name: str):
        self.action_name: str = action_name

    def run(self, agent: Agent) -> bool:
        action_func = getattr(agent, self.action_name, None)
        if not action_func:
            raise ValueError(f"Action function '{self.action_name}' not found in the agent class.")
        # print(f"Executing action {self.action_name}")
        return action_func()