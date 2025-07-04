import lancedb
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from lancedb.embeddings import EmbeddingFunctionRegistry, get_registry
from openai import OpenAI
import pandas as pd
import pyarrow as pa
from typing import Type, List
import os
import json

load_dotenv()
xml_examples_path = "./simulation/trees/RAG_examples"
embedder = get_registry().get("openai").create(name="text-embedding-ada-002")


class BehaviorTreeSchema(LanceModel):
    id: str 
    prompt: str = embedder.SourceField()
    tree: str = embedder.SourceField()
    vector: Vector(embedder.ndims()) = embedder.VectorField() # type: ignore


def read_json_examples() -> pd.DataFrame:
    data = []
    
    # Iterate through all JSON files in the directory
    for filename in os.listdir(xml_examples_path):
        if filename.endswith('.json'):
            file_path = os.path.join(xml_examples_path, filename)
            
            # Read the JSON file
            with open(file_path, 'r') as file:
                content = json.load(file)
                
                # Get file ID (filename without .json)
                file_id = os.path.splitext(filename)[0]
                
                # Add to data list
                data.append({
                    'id': file_id,
                    'prompt': content['prompt'],
                    'tree': content['tree']
                })
    
    # Create DataFrame
    return pd.DataFrame(data)

df = read_json_examples()
print(df)
db = lancedb.connect("./lancedb")

if "behavior_trees" in db.table_names():
    db.drop_table("behavior_trees")
table = db.create_table("behavior_trees", schema=BehaviorTreeSchema)
table.add(df)

# Get the query and its embedding
query = df.iloc[0]['prompt']
client = OpenAI()

response = client.embeddings.create(
    input=query,
    model="text-embedding-ada-002"
)

embedded_query = response.data[0].embedding
results = table.search(embedded_query).limit(2).to_pydantic(BehaviorTreeSchema)

for chunk in results:
    print(chunk.prompt)
    print(chunk.tree)








