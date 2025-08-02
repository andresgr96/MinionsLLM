""" Run a simulation in the robot environment """

import os
from typing import Any, Dict, Optional, Type

from vi import Config, Window

from agent_control import RobotEnvironment
from tree_parser import save_behavior_tree_xml


def run_robot_sim(
    bt_str: str,
    environment_class: Optional[Type[Any]] = None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a robot environment simulation with behavior trees.

    Args:
        bt_str: The behavior tree XML string
        environment_class: The environment class to use (defaults to RobotEnvironment)
        environment_kwargs: Dict of keyword arguments for the environment constructor
            Example: {'n_agents': 10, 'n_parts': 15, 'task': 'collect', 'headless': False}
        config_kwargs: Dict of keyword arguments for the Config constructor
            Example: {'radius': 25, 'duration': 1000, 'window_size': 500, 'movement_speed': 1.0}

    Returns:
        Dict containing simulation metrics
    """
    # Use defaults if not provided
    environment_class = environment_class or RobotEnvironment
    environment_kwargs = environment_kwargs or {}
    config_kwargs = config_kwargs or {}

    # Set default config values if not provided
    default_config = {
        "radius": 25,  # Agent detection radius
        "visualise_chunks": True,  # Show spatial partitioning
        "window_size": 500,  # Window size (will create square window)
        "movement_speed": 1.0,  # Agent movement speed
        "duration": 1000,  # Simulation duration in steps
    }

    # Merge default config with provided config
    final_config = {**default_config, **config_kwargs}

    # Handle window_size parameter - convert to Window object
    if "window_size" in final_config:
        window_size = final_config.pop("window_size")
        final_config["window"] = Window.square(window_size)
    elif "window" not in final_config:
        # If neither window_size nor window is provided, use default
        final_config["window"] = Window.square(500)

    # Configuration for the simulation
    config = Config(**final_config)

    # Create a temp path for the behavior tree
    bt_path = "./temp_bt.xml"
    save_behavior_tree_xml(bt_str, bt_path)

    # Set default environment values if not provided
    default_environment = {
        "config": config,
        "bt_path": bt_path,
        "n_agents": 10,  # Number of robot agents
        "n_parts": 10,  # Number of parts in environment
        "task": "collect",  # Task description
        "headless": False,  # Show GUI (set to True for no GUI)
    }

    # Merge default environment with provided environment kwargs
    # Note: provided kwargs override defaults, including config and bt_path
    final_environment = {**default_environment, **environment_kwargs}
    final_environment["config"] = config  # Always use the constructed config
    final_environment["bt_path"] = bt_path  # Always use the temp bt_path

    # Create the environment
    environment = environment_class(**final_environment)
    environment.setup()

    try:
        # Run the simulation and collect metrics
        results = environment.run()
    finally:
        # Ensure pygame is properly cleaned up
        try:
            import pygame

            pygame.display.quit()
            pygame.quit()
        except:
            pass  # Ignore errors if pygame is not available or already cleaned up

    # Delete the temp behavior tree
    os.remove(bt_path)

    return results
