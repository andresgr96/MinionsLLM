"""
Example showing how to use the DatasetGenerator class to generate
behavior trees and datasets with filtering method.
"""

from vi import Config, Window

from data_grammar import DatasetGenerator
from agent_control.simulation.agents import RobotAgent
from agent_control.simulation.envs import RobotEnvironment 

# Initialize the generator with a small output directory
generator = DatasetGenerator(
    agent_class=RobotAgent,  # Since class is provided, the framework automatically extracts the translation of its nodes from docstrings
    output_dir="./examples/03_data_gen_examples/output/filtered_test_run",
    seed=42
)

# Next you need to define the environment we will test the trees for metrics
filter_env = RobotEnvironment(
    config=Config(
        window=Window.square(500),
        movement_speed=1.0,   # Its important to be consistent with speed and duration, otherwise the metrics will be inconsistent
        duration=500 
    ),
    bt_path="dummy_path", # This is a dummy path, we will use the trees in the dataset to test the metrics
    n_agents=10,
    n_parts=10,
    task="dummy_task",
    headless=True
)

# Generate a small dataset using method A with metric filtering
print("\n=== Generating Dataset A ===")
dataset_a_path = generator.generate_dataset_a(
    dataset_name="filtered_test_dataset_a",
    n_trees=5,  
    max_trees_to_process=5,
    filter_env=filter_env,
    filter_metrics=["good_parts_picked_up", "bad_parts_picked_up"] # Specifying metrics allows to controll the strictness of the filtering. 
                                                                   # List of metrics validates with OR operator. Dictionary uses AND operator.
)
print(f"FilteredDataset A saved to: {dataset_a_path}")

# Generate a filtered dataset using method B with metric filtering
print("\n=== Generating Dataset B ===")

dataset_b_path = generator.generate_dataset_b(
    dataset_name="filtered_test_dataset_b",
    n_trees=1,
    max_trees_to_process=1, # Small number for example purposes
    filter_env=filter_env,
    filter_metrics=["good_parts_picked_up", "bad_parts_picked_up"] 
)
print(f"FilteredDataset B saved to: {dataset_b_path}")

print("\nDataset generation complete!")
print("You can find the generated trees and datasets in the ./output/test_run directory") 