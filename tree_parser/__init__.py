"""
Parser Layer package for parsing behavior trees.

This package provides functionality for:
1. Parsing XML behavior trees into executable node structures
2. Defining base node classes for behavior trees
3. Validating behavior trees against a grammar
"""

from .agent_doc_parser import AgentDocstringParser
from .base_nodes import (
    ActuatorActionNode,
    BaseNode,
    ConditionNode,
    SelectorNode,
    SequenceNode,
    StateActionNode,
)
from .grammar_validator import BehaviorTreeGrammarValidator
from .middle_parser import (
    parse_behavior_tree,
    parse_behavior_tree_with_metadata,
    save_behavior_tree_with_metadata,
    save_behavior_tree_xml,
)

__all__ = [
    "ActuatorActionNode",
    "AgentDocstringParser",
    "BaseNode",
    "BehaviorTreeGrammarValidator",
    "ConditionNode",
    "SelectorNode",
    "SequenceNode",
    "StateActionNode",
    "parse_behavior_tree",
    "parse_behavior_tree_with_metadata",
    "save_behavior_tree_with_metadata",
    "save_behavior_tree_xml",
]
