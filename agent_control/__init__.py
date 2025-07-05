"""
Control Layer package for simulation and robot control.

This package provides functionality for:
1. Simulating behavior trees in various environments
"""

from .simulation import RobotAgent, RobotEnvironment

__all__ = ["RobotAgent", "RobotEnvironment"]
