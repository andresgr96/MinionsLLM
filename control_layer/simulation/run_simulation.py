"""
This module provides the run_simulation function that serves as the main entry point
for running behavior tree simulations.
"""

from typing import Any, Dict, Optional

from vi import Config, Window

from .envs.robot_env import RobotEnvironment


def run_simulation(
    bt_path: str,
    env_type: str = "robot",
    n_agents: int = 10,
    n_parts: int = 10,
    scenario: str = "robot",
    headless: bool = False,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """
    Run a behavior tree simulation with the specified parameters.

    Args:
        bt_path: Path to the behavior tree XML file
        env_type: Type of environment to simulate ('robot')
        n_agents: Number of agents in the environment
        n_parts: Number of parts in the environment
        scenario: Scenario for the simulation
        headless: Whether to run the simulation in headless mode (no display)
        config: Optional custom configuration for the simulation

    Returns:
        Metrics from the simulation run
    """
    # Create default config if none provided
    if config is None:
        config = Config(
            radius=25,
            visualise_chunks=True,
            window=Window.square(500),
            movement_speed=1,
            duration=1000,
        )

    # Create the appropriate environment
    if env_type.lower() == "robot":
        environment = RobotEnvironment(
            config=config,
            bt_path=bt_path,
            n_agents=n_agents,
            n_parts=n_parts,
            task=scenario,
            headless=headless,
        )
    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. Only 'robot' is supported."
        )

    # Setup and run the simulation
    environment.setup()
    metrics = environment.run()

    return metrics
