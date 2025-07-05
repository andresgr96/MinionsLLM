"""
Example: Using BehaviorTreeGenerator with LlamaCPP Backend

This example demonstrates how to use the BehaviorTreeGenerator class with
the llamacpp backend to generate behavior trees from natural language prompts.
"""

import time
from llm_layer import BehaviorTreeGenerator
from control_layer.simulation.agents.robot_agent import RobotAgent


def main():
    """Generate behavior trees using LlamaCPP backend."""
    print("=== BehaviorTreeGenerator with LlamaCPP Backend ===")
    print("This example uses a local GGUF model file for generation.\n")
    
    # Initialize the generator with LlamaCPP backend
    print("Initializing BehaviorTreeGenerator...")
    generator = BehaviorTreeGenerator(
        agent_class=RobotAgent,
        backend="llamacpp",
        model_path_or_url="./llm_layer/models/gemma-3-1b-it-qat-B.Q8_0.gguf",
        chat_format="llama-3",
        context_length=1024*4,
        gpu_layers=0,        
        temperature=0.2,
        verbose=False
    )
    
    # Define the prompt
    prompt = ("Find any part in the environment and pick it up")
    
    print(f"Prompt: {prompt}\n")
    print("Generating behavior tree...")
    
    # Generate the behavior tree
    start_time = time.time()
    error_count, behavior_tree = generator.generate_behavior_tree(
        prompt=prompt,
        which_prompt=3,      # Two-shot learning
        log_prompt=False     # Set to True to see the full prompt
    )
    end_time = time.time()
    
    # Display results
    print("\n" + "="*50)
    print("GENERATION RESULTS")
    print("="*50)
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    print(f"Error count: {error_count}")
    print(f"Generated behavior tree:\n{behavior_tree}")
    
    if error_count == 0:
        print("\nSuccessfully generated a valid behavior tree!")
    else:
        print(f"\nGenerated tree has {error_count} validation errors.")
    
    print("\nTo use this tree in a simulation:")
    print("1. Save the XML to a file")
    print("2. Use it with the robot environment simulation")
    print("3. Check examples/simulation_examples for simulation usage")


if __name__ == "__main__":
    main()