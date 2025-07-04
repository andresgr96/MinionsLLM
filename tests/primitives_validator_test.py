import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser_layer.primitives_validator import validate_primitives
from control_layer.simulation.agents.robot_agent import RobotAgent


def get_correct_examples() -> list:
    """
    Get the default handcoded examples.
    
    Returns:
        List of handcoded example dictionaries
    """
    bring_good_base = {
        "expected_error": "This tree is correct.",
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

    bring_scrap_waste = {
        "expected_error": "This tree is correct.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <Condition>is_agent_in_waste_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_seek_waste_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_scrap_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    bring_good_waste = {
        "expected_error": "This tree is correct.",
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
        "expected_error": "This tree is correct.",
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

    find_bad_stop = {
        "expected_error": "This tree is correct.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_scrap_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>  

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>     

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    find_good_stop = {
        "expected_error": "This tree is correct.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>  

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>     

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    find_base = {
        "expected_error": "This tree is correct.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_agent_in_base_area</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>        

                        <StateAction>state_seek_base_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    find_source_back = {
        "expected_error": "This tree is correct.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_agent_in_source_area</Condition>
                            <StateAction>state_seek_base_area</StateAction>
                        </Sequence>        

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    find_waste = {
        "expected_error": "This tree is correct.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_agent_in_waste_area</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>        

                        <StateAction>state_seek_waste_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    find_part = {
        "expected_error": "This tree is correct.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    maintain_freeze = {
        "expected_error": "This tree is correct.",
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
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <Condition>is_agent_in_waste_area</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>


                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
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

    return [
        bring_good_base,
        bring_scrap_waste,
        bring_good_waste,
        bring_scrap_source,
        find_bad_stop,
        find_good_stop,
        find_base,
        find_source_back,
        find_waste,
        find_part,
        maintain_freeze
    ]



def get_incorrect_examples() -> list:
    """
    Get the incorrect handcoded examples.
    
    Returns:
        List of incorrect example dictionaries
    """
    bring_good_base = {
        "expected_error": "is_agent_holding_goofy_part is not a valid condition.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_goofy_part</Condition>
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

    bring_scrap_waste = {
        "expected_error": "drop_part_in_waste is not a valid actuator action.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <Condition>is_agent_in_waste_area</Condition>
                            <ActuatorAction>drop_part_in_waste</ActuatorAction>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_seek_waste_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_scrap_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    bring_good_waste = {
        "expected_error": "state_seek_eating_area is not a valid state action.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <Condition>is_agent_in_waste_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                        </Sequence>

                        <Selector>

                            <Sequence>
                                <Condition>is_agent_holding_good_part</Condition>
                                <Condition>is_agent_in_waste_area</Condition>
                                <ActuatorAction>drop_part</ActuatorAction>
                            </Sequence>

                            <Sequence>
                                <Condition>is_agent_holding_good_part</Condition>
                                <StateAction>state_seek_eating_area</StateAction>
                            </Sequence>
                    
                            <Sequence>
                                <Condition>is_part_detected</Condition>
                                <ActuatorAction>pick_up_part</ActuatorAction>
                            </Sequence>       

                            <StateAction>state_seek_source_area</StateAction>
                        </Selector>
                    </Selector>
                </BehaviorTree>"""
    }

    find_waste = {
        "expected_error": "is_agent_in_amsterdam_area is not a valid condition.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_agent_in_amsterdam_area</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>        

                        <StateAction>state_seek_waste_area</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    find_part = {
        "expected_error": "state_random_stop is not a valid state action.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_good_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_random_stop</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }
    return [
        bring_good_base,
        bring_scrap_waste,
        bring_good_waste,
        find_waste,
        find_part
    ]

if __name__ == "__main__":
    print("--- Testing Correct Examples ---")
    correct_examples = get_correct_examples()
    all_correct_passed = True
    for i, example in enumerate(correct_examples):
        print(f"--- Correct Example {i+1} ---")
        is_valid, feedback = validate_primitives(example["tree"], RobotAgent)
        print(f"Is Valid: {is_valid}")
        print(f"Feedback: '{feedback}'")
        if not is_valid or feedback != "":
            all_correct_passed = False
            print(">>> FAILED. Expected valid with no feedback.")
        else:
            print(">>> PASSED")
        print("-" * (len(str(i+1)) + 20))


    print("\n\n--- Testing Incorrect Examples ---")
    incorrect_examples = get_incorrect_examples()
    all_incorrect_passed = True
    for i, example in enumerate(incorrect_examples):
        print(f"--- Incorrect Example {i+1} ---")
        is_valid, feedback = validate_primitives(example["tree"], RobotAgent)
        expected_primitive = example["expected_error"].split(" ")[0]

        print(f"Is Valid: {is_valid}")
        print(f"Feedback: '{feedback}'")
        print(f"Expected to fail and contain: '{expected_primitive}'")

        if is_valid or expected_primitive not in feedback:
            all_incorrect_passed = False
            print(">>> FAILED.")
        else:
            print(">>> PASSED.")
        print("-" * (len(str(i+1)) + 22))

    print("\n\n--- Summary ---")
    if all_correct_passed:
        print("All correct examples PASSED.")
    else:
        print("Some correct examples FAILED.")
    
    if all_incorrect_passed:
        print("All incorrect examples PASSED.")
    else:
        print("Some incorrect examples FAILED.")

    if all_correct_passed and all_incorrect_passed:
        print("\nValidation script finished successfully!")
    else:
        print("\nValidation script finished with errors.") 