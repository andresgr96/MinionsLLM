"""
Dataset enrichment functionality for adding handcoded examples to generated datasets.
"""

import json
from typing import Dict, List, Any, Optional
from ..grammar_gen.tree_to_prompt import generate_technical_prompt_from_string, generate_spoon_prompt_from_string



def process_example(example: Dict[str, Any], node_translations: Dict[str, str], node_connectors: Dict[str, str], spoon_node_translations: Dict[str, str]) -> Dict[str, Any]:
    """
    Process a handcoded example by generating technical and spoon prompts.
    
    Args:
        example: Dictionary with 'layman_task' and 'tree' keys
        node_translations: Dictionary mapping node types to their translations
        node_connectors: Dictionary of connector words
        spoon_node_translations: Dictionary mapping node types to spoon-fed translations
        
    Returns:
        Processed example with all prompt types
    """
    try:
        technical_task = generate_technical_prompt_from_string(
            example["tree"], 
            node_translations, 
            node_connectors
        )
    except Exception as e:
        technical_task = f"Error generating technical task: {e}"

    try:
        spoon_task = generate_spoon_prompt_from_string(
            example["tree"], 
            spoon_node_translations, 
            node_connectors
        )
    except Exception as e:
        spoon_task = f"Error generating spoon task: {e}"

    return {
        "layman_task": example["layman_task"],
        "technical_task": technical_task,
        "spoon_task": spoon_task,
        "tree": example["tree"]
    }


def get_handcoded_examples() -> List[Dict[str, Any]]:
    """
    Get the default handcoded examples.
    
    Returns:
        List of handcoded example dictionaries
    """
    bring_good_base = {
        "layman_task": "Find good parts and bring them to the base.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <Condition>is_agent_in_base_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_seek_base_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    bring_good_storage = {
        "layman_task": "Find good parts and bring them to the storage.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <Condition>is_agent_in_storage_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_seek_storage_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    bring_scrap_waste = {
        "layman_task": "Find scrap parts and bring them to the storage.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <Condition>is_agent_in_storage_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_seek_storage_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    bring_good_waste = {
        "layman_task": "Find good parts and bring them to the waste.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <Condition>is_agent_in_waste_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_seek_waste_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    bring_scrap_source = {
        "layman_task": "Find scrap parts and bring them to the source.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <Condition>is_agent_in_source_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_seek_source_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_scrap_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    bring_scrap_construction = {
        "layman_task": "Find scrap parts and bring them to the construction.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <Condition>is_agent_in_construction_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_seek_construction_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_scrap_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    find_bad_stop = {
        "layman_task": "Find scrap parts and stop moving when you find one.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>   

                        <Sequence>
                            <Condition>is_scrap_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>   

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    find_good_stop = {
        "layman_task": "Find good parts and stop moving when you find one.",
        "tree": """<BehaviorTree>
                    <Selector>
  
                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>     

                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>   

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    maintain_freeze = {
        "layman_task": "Collect as many parts as you can. Bring good parts to the base, while if you find a scrap part, stop moving.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <Condition>is_agent_in_base_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>


                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_seek_base_area</StateAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence> 

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <Condition>is_agent_in_waste_area</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>


                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_seek_waste_area</StateAction>
                        </Sequence>     

                        <Sequence>
                            <Condition>is_scrap_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>     

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    maintain_construction = {
        "layman_task": "Collect as many parts as you can. Bring good parts to the construction area, while if you find a scrap part, bring them to the storage.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <Condition>is_agent_in_construction_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_seek_construction_area</StateAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence> 

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <Condition>is_agent_in_storage_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>


                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_seek_storage_area</StateAction>
                        </Sequence>   


                        <Sequence>
                            <Condition>is_scrap_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    return [
        bring_good_base,
        bring_good_storage,
        bring_scrap_waste,
        bring_good_waste,
        bring_scrap_source,
        bring_scrap_construction,
        find_bad_stop,
        find_good_stop,
        maintain_freeze,
        maintain_construction
    ]


def enrich_dataset(input_file: str, 
                  output_file: str, 
                  handcoded_examples: Optional[List[Dict[str, Any]]] = None,
                  node_translations: Optional[Dict[str, str]] = None,
                  node_connectors: Optional[Dict[str, str]] = None,
                  spoon_node_translations: Optional[Dict[str, str]] = None) -> None:
    """
    Enrich a dataset by appending handcoded examples.
    
    Args:
        input_file: Path to the input dataset JSON file
        output_file: Path to save the enriched dataset
        handcoded_examples: List of handcoded examples (uses default if None)
        node_translations: Dictionary mapping node types to their translations
        node_connectors: Dictionary of connector words
        spoon_node_translations: Dictionary mapping node types to spoon-fed translations
    """
    # Load the original dataset
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    # Get handcoded examples
    if handcoded_examples is None:
        handcoded_examples = get_handcoded_examples()
    
    # Process handcoded examples
    processed_examples = []
    for example in handcoded_examples:
        processed_example = process_example(
            example, 
            node_translations or {}, 
            node_connectors or {}, 
            spoon_node_translations or {}
        )
        processed_examples.append(processed_example)
    
    # Create a set of existing trees for efficient duplicate checking
    existing_trees = {item['tree'] for item in dataset}
    
    # Append handcoded examples if they are not duplicates
    added_count = 0
    for example in processed_examples:
        dataset.append(example)
        existing_trees.add(example['tree'])
        added_count += 1
    
    # Save the enriched dataset
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Enriched dataset: Added {added_count} new handcoded examples.")
    print(f"Total dataset size: {len(dataset)}")

