""" Run a simulation int he robot environment """

from typing import Dict, Any
from agent_control import RobotEnvironment
from vi import Config, Window
from tree_parser import save_behavior_tree_xml
import os

def run_robot_sim(bt_str: str)-> Dict[str, Any]:
    """Run a robot environment simulation with behavior trees."""
    # Configuration for the simulation
    config = Config(
        radius=25,                    # Agent detection radius
        visualise_chunks=True,        # Show spatial partitioning
        window=Window.square(500),    # 500x500 window
        movement_speed=1.0,           # Agent movement speed
        duration=1000,                # Simulation duration in steps
    )
    
    # Create a temp path for the behavior tree
    bt_path = "./temp_bt.xml"
    save_behavior_tree_xml(bt_str, bt_path)
    
    # Create the robot environment
    environment = RobotEnvironment(
        config=config,
        bt_path=bt_path,
        n_agents=10,                   # Number of robot agents
        n_parts=10,                   # Number of parts in environment
        task="collect",               # Task description
        headless=False,               # Show GUI (set to True for no GUI)
    )
    environment.setup()
    
    # Run the simulation and collect metrics
    results = environment.run()

    # Delete the temp behavior tree
    os.remove(bt_path)

    return results