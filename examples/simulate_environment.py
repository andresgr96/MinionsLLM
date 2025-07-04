"""
This example shows how to simulate a given environment by importing the run_simulation
function from the control_layer package.
"""

from control_layer import run_simulation
from vi import Config, Window

def main():
    # Define the path to the behavior tree XML file
    bt_path = "./control_layer/simulation/trees/maintain.xml"
    
    # Create a custom configuration (optional)
    config = Config(
        radius=50,
        visualise_chunks=True,
        window=Window.square(500),
        movement_speed=1.0,
        duration=1000  # Shorter duration for the example
    )
    
    # Run the simulation with the robot environment
    print("\nRunning Robot Environment Simulation...")
    metrics = run_simulation(
        bt_path=bt_path,
        env_type="robot",
        n_agents=10,
        n_parts=10,
        scenario="clean",
        headless=False,
        config=config
    )
    
    # Print the metrics
    print("\nSimulation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()