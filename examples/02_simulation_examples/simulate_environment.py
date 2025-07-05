"""
Example: Running a Behavior Tree in Robot Environment

This example demonstrates how to create and run a robot environment simulation
with behavior trees. It shows the proper way to:
1. Create a robot environment
2. Configure simulation parameters
3. Run the simulation with behavior trees
4. Collect and display results

The example uses the "collect" scenario where robots find good parts and
bring them back to the base area.
"""

from agent_control.simulation.envs.robot_env import RobotEnvironment
from vi import Config, Window


def main():
    """Run a robot environment simulation with behavior trees."""
    # Configuration for the simulation
    config = Config(
        radius=25,                    # Agent detection radius
        visualise_chunks=True,        # Show spatial partitioning
        window=Window.square(500),    # 500x500 window
        movement_speed=1.0,           # Agent movement speed
        duration=1000,                # Simulation duration in steps
    )
    
    # Path to the behavior tree XML file
    # This tree implements a "collect good parts" behavior
    bt_path = "./examples/02_simulation_examples/collect.xml"
    
    print("Creating Robot Environment...")
    
    # Create the robot environment
    environment = RobotEnvironment(
        config=config,
        bt_path=bt_path,
        n_agents=10,                   # Number of robot agents
        n_parts=10,                   # Number of parts in environment
        task="collect",               # Task description
        headless=False,               # Show GUI (set to True for no GUI)
    )
    
    print("Setting up simulation...")
    
    # Setup the environment (spawn agents, parts, obstacles)
    environment.setup()
    
    print("Running simulation...")
    print("- Agents will search for good parts")
    print("- When found, they'll bring them to the base")
    print("- Close the window or wait for completion")
    
    # Run the simulation and collect metrics
    results = environment.run()
    
    # Display results
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\nSimulation completed!")


if __name__ == "__main__":
    main()