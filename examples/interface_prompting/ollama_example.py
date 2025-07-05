"""
Example: Using BehaviorTreeGenerator with Ollama Backend

This example demonstrates two ways to use the BehaviorTreeGenerator class with
the Ollama backend:
1. Using an existing Ollama model
2. Auto-importing a GGUF model from Hugging Face

DISCLAIMER: You need to have Ollama installed and running.
"""

import time
from llm_layer import BehaviorTreeGenerator
from control_layer.simulation.agents.robot_agent import RobotAgent


def example_existing_model():
    """Example using an existing Ollama model from their library."""
    print("=== Example 1: Pulling and using Existing Ollama Model ===")
    print("This example uses a model that's already available in Ollama.\n")
    
    try:
        # Initialize with existing Ollama model
        print("Initializing BehaviorTreeGenerator with existing model...")
        generator = BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="ollama",
            ollama_model_name="gemma3:1b",  # This also works if you already pulled it from the library
            temperature=0.2
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
        print("GENERATION RESULTS (Existing Model)")
        print("="*50)
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        print(f"Error count: {error_count}")
        print(f"Generated behavior tree:\n{behavior_tree}")
        
        if error_count == 0:
            print("\n✅ Successfully generated a valid behavior tree!")
        else:
            print(f"\n⚠️  Generated tree has {error_count} validation errors.")
            
    except Exception as e:
        print(f"❌ Error with existing model: {e}")

def example_auto_import():
    """Example using auto-import of GGUF model from Hugging Face."""
    print("\n=== Example 2: Auto-Import GGUF Model ===")
    print("This example automatically imports a GGUF model from Hugging Face.\n")
    
    try:
        # Initialize with auto-import from Hugging Face
        print("Initializing BehaviorTreeGenerator with auto-import...")
        print("This will download and import the model if not already available.")
        
        generator = BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="ollama",
            model_path_or_url="hf.co/Andresgr96/gemma-3-1b-it-qat-B:Q8_0",
            temperature=0.2
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
        print("GENERATION RESULTS (Auto-Import)")
        print("="*50)
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        print(f"Error count: {error_count}")
        print(f"Generated behavior tree:\n{behavior_tree}")
        
        if error_count == 0:
            print("\n✅ Successfully generated a valid behavior tree!")
        else:
            print(f"\n⚠️  Generated tree has {error_count} validation errors.")
            
    except Exception as e:
        print(f"❌ Error with auto-import: {e}")
        print("Make sure Ollama is running and you have internet access.")


def main():
    """Run both Ollama examples."""
    print("BehaviorTreeGenerator Ollama Examples")
    print("=" * 50)
    print("Prerequisites:")
    print("- Ollama must be installed and running")
    print()
    
    # Run both examples
    example_existing_model()
    example_auto_import()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Try different prompts to see various behavior trees")
    print("2. Use the generated trees in simulation (see examples/simulate_environment.py)")
    print("3. Experiment with different prompt types (which_prompt parameter)")
    print("4. Enable log_prompt=True to see the full prompts sent to the model")


if __name__ == "__main__":
    main()