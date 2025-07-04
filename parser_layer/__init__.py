"""
Parser Layer package for parsing behavior trees.

This package provides functionality for:
1. Parsing XML behavior trees into executable node structures
2. Defining base node classes for behavior trees
3. Validating behavior trees against a grammar
"""

from .middle_parser import parse_behavior_tree, parse_behavior_tree_with_metadata, save_behavior_tree_xml, save_behavior_tree_with_metadata
from .base_nodes import BaseNode, SequenceNode, SelectorNode, ActuatorActionNode, StateActionNode, ConditionNode
from .grammar_validator import BehaviorTreeGrammarValidator
from .agent_doc_parser import AgentDocstringParser

__all__ = [
    'parse_behavior_tree',
    'parse_behavior_tree_with_metadata',
    'save_behavior_tree_xml',
    'save_behavior_tree_with_metadata',
    'BaseNode',
    'SequenceNode',
    'SelectorNode',
    'ActuatorActionNode',
    'StateActionNode',
    'ConditionNode',
    'BehaviorTreeGrammarValidator',
    'AgentDocstringParser'
] 