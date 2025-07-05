"""
Example: Generating and Saving Behavior Trees

This example demonstrates how to:
1. Use Ollama to pull and use a model from Hugging Face
2. Generate a behavior tree from a natural language prompt
3. Save the generated tree as an XML file
4. Provide the saved file path to the user

The example uses the gemma2:2b model and saves the output in the same directory.
"""

import os
import time
from pathlib import Path
from llm_layer import BehaviorTreeGenerator
from control_layer.simulation.agents.robot_agent import RobotAgent
from parser_layer.middle_parser import save_behavior_tree_xml


def main():
    """Generate a behavior tree and save it as an XML file."""
    print("=== Behavior Tree Generation and Saving Example ===")
    print("This example will:")
    print("1. Pull the gemma3:1b model using Ollama")
    print("2. Generate a behavior tree from a prompt")
    print("3. Save the tree as an XML file")
    print("4. Show you where the file was saved\n")
    
    # Get the current directory (where this example is located)
    current_dir = Path(__file__).parent
    output_filename = "generated_behavior_tree.xml"
    output_path = current_dir / output_filename
    
    try:
        # Initialize the generator with Ollama backend
        print("Initializing BehaviorTreeGenerator with Ollama...")
        print("This will automatically pull gemma3:1b if not already available.")
        
        generator = BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="ollama",
            ollama_model_name="gemma3:1b",  # This will be pulled automatically
            temperature=0.3,
            verbose=True  # Print the full formatted prompt fed to the llm
        )
        
        # Define prompt
        prompt = """
        Find any part in the environment and pick it up.
        """
        
        print(f"Prompt: {prompt.strip()}\n")
        print("Generating behavior tree...")
        print("This may take a moment depending on your hardware...\n")
        
        # Generate the behavior tree
        start_time = time.time()
        error_count, behavior_tree_xml = generator.generate_behavior_tree(
            prompt=prompt,
            which_prompt=3,      # Two-shot learning
            log_prompt=False     # Set to True to see the full prompt
        )
        end_time = time.time()
        
        # Display generation results
        print("="*60)
        print("GENERATION RESULTS")
        print("="*60)
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        print(f"Validation errors: {error_count}")
        print(f"Generated XML:\n{behavior_tree_xml}")
        
        if error_count == 0:
            print("\n✅ Successfully generated a valid behavior tree!")
        else:
            print(f"\n⚠️  Generated tree has {error_count} validation errors.")
            print("The tree will still be saved for inspection.")
        
        # Save the behavior tree to XML file
        print("\n" + "="*60)
        print("SAVING BEHAVIOR TREE")
        print("="*60)
        
        if behavior_tree_xml and behavior_tree_xml.strip():
            # Use the middle_parser function to save the XML
            save_behavior_tree_xml(behavior_tree_xml, str(output_path))
            
            # Verify the file was created
            if output_path.exists():
                file_size = output_path.stat().st_size
                print(f"✅ Behavior tree saved successfully!")
                print(f"📁 File location: {output_path.absolute()}")
                print(f"📊 File size: {file_size} bytes")
                
                # Show the saved content
                print(f"\n📄 Saved content preview:")
                with open(output_path, 'r') as f:
                    content = f.read()
                    print(content[:500] + "..." if len(content) > 500 else content)
                    
            else:
                print("❌ Error: File was not created successfully.")
        else:
            print("❌ Error: No valid XML content to save.")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is installed and running")
        print("2. Check your internet connection (needed to pull the model)")
        print("3. Ensure you have enough disk space for the model")
        print("4. Try running 'ollama pull gemma2:2b' manually first")
        
        # Create a sample XML file for demonstration
        sample_xml = """<?xml version="1.0" encoding="utf-8"?>
<BehaviorTree>
    <Selector>
        <Sequence>
            <Condition>is_agent_holding_scrap_part</Condition>
            <Condition>is_agent_in_waste_area</Condition>
            <ActuatorAction>drop_part</ActuatorAction>
        </Sequence>
        <Sequence>
            <Condition>is_scrap_part_detected</Condition>
            <ActuatorAction>pick_up_part</ActuatorAction>
        </Sequence>
        <Sequence>
            <Condition>is_agent_holding_scrap_part</Condition>
            <StateAction>state_seek_waste_area</StateAction>
        </Sequence>
        <StateAction>state_random_walk</StateAction>
    </Selector>
</BehaviorTree>"""
        
        sample_path = current_dir / "sample_behavior_tree.xml"
        save_behavior_tree_xml(sample_xml, str(sample_path))
        print(f"\n📄 Created sample XML file at: {sample_path.absolute()}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("Now you can:")
    print("1. Open the XML file in any text editor to inspect it")
    print("2. Use the XML file in a simulation (see examples/simulation_examples)")
    print("3. Modify the prompt above to generate different behaviors")
    print("4. Try different prompt techniques (change which_prompt parameter)")
    print("5. Experiment with different models by changing ollama_model_name")


if __name__ == "__main__":
    main()