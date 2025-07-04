"""
This example shows how to use the LLM layer to generate behavior trees from natural language prompts.
"""

from llm_layer import BehaviorTreeGenerator
from control_layer.simulation import RobotAgent
import os
from openai import OpenAI
from lancedb.pydantic import LanceModel, Vector
import lancedb

# # Define a simple schema for the vector database, youll need to define this if you want to use RAG
# class TreeExample(LanceModel):
#     prompt: str
#     tree: str
#     embedding: Vector(1536)  # type: ignore # OpenAI embeddings are 1536 dimensions

def main():
#     # Set up OpenAI client
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
#     # Set up a simple in-memory vector database
#     db = lancedb.connect("~/.lancedb")
    
#     try:
#         table = db.open_table("tree_examples")
#         print("Opened existing table 'tree_examples'")
#     except Exception:
#         table = db.create_table("tree_examples", schema=TreeExample)
#         print("Created new table 'tree_examples'")

    client = None
    table = None
    TreeExample = None
    
    # Create the behavior tree generator
    generator = BehaviorTreeGenerator(
        agent_class=RobotAgent,
        model_file="./llm_layer/models/Llama-3.1-8B-Instruct-BT-B2-1000.Q8_0.gguf",  # Path to your model file
        chat_format="llama-3",
        context_length=1024*4,
        gpu_layers=10,
        n_threads=os.cpu_count(),
        temperature=0.0,
        top_p=0.95,
        top_k=150,
        repeat_penalty=1.2,
        seed=100,
        vector_tb=table,
        client=client,
        schema=TreeExample
    )
    
    # Generate a behavior tree from a natural language prompt
    prompt = "Find all the good parts and bring them to the base"
    error_count, behavior_tree = generator.generate_behavior_tree(
        prompt=prompt,
        which_prompt=4,  # Using two-shot learning
        log_prompt=True  # This will log the full prompt to the console for debugging purposes
    )
    
    # # Print the results
    # print("\nGenerated Behavior Tree:")
    # print(behavior_tree)
    # print(f"\nError count: {error_count}")

if __name__ == "__main__":
    main()