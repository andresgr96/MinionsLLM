"""
Example showing how to use the DatasetGenerator class to generate
behavior trees and datasets.
"""

from data_grammar import DatasetGenerator

# Initialize the generator with a small output directory
generator = DatasetGenerator(
    output_dir="./examples/output/basic_test_run",
    seed=42
)

# Generate a small dataset using method A (populated trees)
print("\n=== Generating Dataset A ===")
dataset_a_path = generator.generate_dataset_a(
    dataset_name="test_dataset_a",
    n_trees=10,  
    tree_size=8,
    max_trees_to_process=5  # Process only 5 trees
)
print(f"Dataset A saved to: {dataset_a_path}")

# Generate a small dataset using method B (unpopulated trees with placeholders)
print("\n=== Generating Dataset B ===")
dataset_b_path = generator.generate_dataset_b(
    dataset_name="test_dataset_b",
    n_trees=10,  
    tree_size=8,
    max_trees_to_process=5  
)
print(f"Dataset B saved to: {dataset_b_path}")

# Optional: Generate an enriched version of a dataset
print("\n=== Generating Enriched Dataset A ===")
enriched_path = generator.generate_dataset_a(
    dataset_name="test_dataset_a_enriched",
    n_trees=20,
    tree_size=8,
    max_trees_to_process=20,
    enrich_dataset=True
)
print(f"Enriched dataset saved to: {enriched_path}")

print("\nDataset generation complete!")
print(f"You can find the generated trees and datasets in the {enriched_path} directory") 
