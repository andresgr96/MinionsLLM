import os
import json
from openai import OpenAI
from .sys_prompt import system_prompt_a
from ..grammar_gen.tree_to_prompt import generate_tech_prompt, generate_spoon_prompt 

client = OpenAI(
    project="proj_g4ndQRSmeGRrVbVKaKJC88Su",
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_tree_content(file_path):
    """Reads the content of an XML file."""
    with open(file_path, "r") as f:
        return f.read().strip()

def process_tree_with_api(tree_content, node_translations, node_connectors, spoon_node_translations):
    """Sends the tree and technical prompt to the API and retrieves the layman task."""

    tech_prompt = generate_tech_prompt(tree_content, node_translations, node_connectors)
    spoon_prompt = generate_spoon_prompt(tree_content, spoon_node_translations, node_connectors)
    
    user_prompt = f"""Please rephrase the given prompt in a natural way (layman style) a normal individual might communicate a task.

    Tree:
    
    {tree_content}

    Technical Prompt: {tech_prompt} """
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": system_prompt_a
            },
            {"role": "user", "content": user_prompt}
        ]
    )

    layman_prompt = completion.choices[0].message.content

    return layman_prompt, tech_prompt, spoon_prompt


def process_trees_in_folder(folder_path, output_json_path, max_trees=None, 
                          node_translations=None, node_connectors=None, 
                          spoon_node_translations=None):
    """
    Processes trees in a folder and saves results to a JSON file.
    """
    dataset = []
    processed_count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xml") and (max_trees is None or processed_count < max_trees):
            tree_path = os.path.join(folder_path, file_name)
            try:
                print(f"Processing file: {file_name}")
                tree_content = get_tree_content(tree_path)

                layman_task, tech_task, spoon_task = process_tree_with_api(
                    tree_path, #this should be the actial tree path, not the content
                    node_translations, 
                    node_connectors, 
                    spoon_node_translations
                )

                if layman_task and tree_content:
                    print(f"Task: {layman_task}\n\nTree:\n{tree_content}\n")
                    
                    dataset.append({
                        "layman_task": layman_task,
                        "technical_task": tech_task,
                        "spoon_task": spoon_task,
                        "tree": tree_content
                    })
                    processed_count += 1
                    print(f"------------Produced: {processed_count} of {max_trees} trees------------")

                else:
                    print(f"Unexpected response format for {file_name}: {tree_content}\n")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    with open(output_json_path, "w") as json_file:
        json.dump(dataset, json_file, indent=4)
        print(f"Dataset saved to {output_json_path}")

