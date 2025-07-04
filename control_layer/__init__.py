"""
Control Layer package for simulation and robot control.

This package provides functionality for:
1. Simulating behavior trees in various environments
2. Controlling physical robots with behavior trees
"""

from .simulation.run_simulation import run_simulation

__all__ = ["run_simulation"]
