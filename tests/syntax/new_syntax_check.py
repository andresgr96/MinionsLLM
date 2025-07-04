# Defining the grammar rules
grammar_rules_new = {                                                                 
    "B":   [["b", ["SEL"]], ["b", ["SEQ"]]],                                                          
    "SEL": [["sel", ["SEQn", "As"]], ["sel", ["SEQn"]]],                                               
    "SEQn":[["SEQ", "SEQn"], ["SEQ"]], 
    "SEQ": [["seq", ["Pn", "A"]], ["seq", ["As", "Pn", "A"]]],
    "b":   ["BehaviorTree", ["children_nodes"]],     
    "sel": ["Selector", ["children_nodes"]],
    "seq": ["Sequence", ["children_nodes"]],                                            
    "A":   [["aa", "sa"], ["aa"], ["sa"]],                                                                  
    "As":  [["aa"], ["sa"]],                                                                  
    "aa":  ["ActuatorAction"],                                                    
    "sa":  ["StateAction"],
    "Pn":  [["p", "Pn"], ["p"], []], 
    "p":   ["Condition"]
}

parsed_grammar_rules = {                                                                 
    "BehaviorTree":  {
        "Combinations": [["Selector"],
                        ["Sequence"]],
        "Amounts": {
            "Selector": 1,
            "Sequence": 1
        }
    },                                                          
    "Selector": {
        "Combinations": [["Sequence", "ActuatorAction"], 
                         ["Sequence", "StateAction"],
                         ["Sequence"]],
        "Amounts": {
            "Sequence": -1,         # -1 means it can have any amount of Sequence nodes as children
            "ActuatorAction": 1,
            "StateAction": 1,
        }
    },
    "Sequence": {   # this is not properly done and should serve as an example of what i mean.
        "Combinations": [["Condition", ["ActuatorAction", "StateAction"]],
                         ["Condition", "ActuatorAction"], 
                         ["Condition", "StateAction"],
                         ["etc"]],
        "Amounts": {
            "Condition": -1,
            "ActuatorAction": 1,
            "StateAction": 1,
        }
    }
}

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
        "expected_error": "Sequence node cannot contain another sequence node.",
        "tree": """<BehaviorTree>
                    <Selector>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <Condition>is_agent_in_base_area</Condition>
                            <ActuatorAction>drop_part</ActuatorAction>
                            <Sequence>
                                <Condition>is_agent_holding_good_part</Condition>
                                <StateAction>state_seek_base_area</StateAction>
                            </Sequence>
                        </Sequence>

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_seek_base_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_seek_source_area</StateAction>
                    </Selector>
                </BehaviorTree>""",
                "shape": {
                    "BehaviorTree": [[["Selector"]]],
                    "Selector": [[["Sequence"]], [["StateAction"]]],
                    "Sequence": { #We have several sequences so we need to specify each of their shapes for comparison
                        "Sequence1": "its shape",
                        "Sequence2": "its shape",
                        "Sequence....n": "its shape"
                    }
                }
    }

    bring_scrap_waste = {
        "expected_error": "A Sequence node cannot contain two StateAction nodes.",
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
                            <StateAction>state_seek_base_area</StateAction>
                        </Sequence>
                
                        <Sequence>
                            <Condition>is_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>       

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                </BehaviorTree>"""
    }

    bring_good_waste = {
        "expected_error": "A Selector node cannot contain another Selector node.",
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
                                <StateAction>state_seek_waste_area</StateAction>
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

    bring_scrap_source = {
        "expected_error": "Behavior tree should start with a Selector or Sequence node.",
        "tree": """<BehaviorTree>
                    <StateAction>state_random_walk</StateAction>
                </BehaviorTree>"""
    }

    find_bad_stop = {
        "expected_error": "Behavior tree can only have a single child, either a Selector or Sequence node.",
        "tree": """<BehaviorTree>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>  

                        <Sequence>
                            <Condition>is_agent_holding_scrap_part</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>     

                        <StateAction>state_random_walk</StateAction>
                    </Selector>
                    <Selector>
                
                        <Sequence>
                            <Condition>is_part_detected</Condition>
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
        "expected_error": "Behavior tree can only have a single child, either a Selector or Sequence node.",
        "tree": """<BehaviorTree>
                        <Sequence>
                            <Condition>is_part_detected</Condition>
                            <ActuatorAction>pick_up_part</ActuatorAction>
                        </Sequence>  

                        <Sequence>
                            <Condition>is_agent_holding_good_part</Condition>
                            <StateAction>state_movement_freeze</StateAction>
                        </Sequence>     
                </BehaviorTree>"""
    }

    find_base = {
        "expected_error": "Whats that node buddy",
        "tree": """<BehaviorTree>syntactically_incorrect</BehaviorTree>
"""
    }

    # find_source_back = {
    #     "layman_task": "Your task is to find the source and then come back to the base.",
    #     "tree": """<BehaviorTree>
    #                 <Selector>
                
    #                     <Sequence>
    #                         <Condition>is_agent_in_source_area</Condition>
    #                         <StateAction>state_seek_base_area</StateAction>
    #                     </Sequence>        

    #                     <StateAction>state_seek_source_area</StateAction>
    #                 </Selector>
    #             </BehaviorTree>"""
    # }

    # find_waste = {
    #     "layman_task": "Your task is to find the waste area and stop when you reach it.",
    #     "tree": """<BehaviorTree>
    #                 <Selector>
                
    #                     <Sequence>
    #                         <Condition>is_agent_in_waste_area</Condition>
    #                         <StateAction>state_movement_freeze</StateAction>
    #                     </Sequence>        

    #                     <StateAction>state_seek_waste_area</StateAction>
    #                 </Selector>
    #             </BehaviorTree>"""
    # }

    # find_part = {
    #     "layman_task": "Your task is to find any part and pick it up.",
    #     "tree": """<BehaviorTree>
    #                 <Selector>
                
    #                     <Sequence>
    #                         <Condition>is_part_detected</Condition>
    #                         <ActuatorAction>pick_up_part</ActuatorAction>
    #                     </Sequence>       

    #                     <StateAction>state_random_walk</StateAction>
    #                 </Selector>
    #             </BehaviorTree>"""
    # }

    # maintain_freeze = {
    #     "layman_task": "Your task is to maintain the base by bringing good parts to it and taking scrap parts to the waste. If you're holding a scrap part and you're in the waste area, stop moving.",
    #     "tree": """<BehaviorTree>
    #                 <Selector>
                
    #                     <Sequence>
    #                         <Condition>is_agent_holding_good_part</Condition>
    #                         <Condition>is_agent_in_base_area</Condition>
    #                         <ActuatorAction>drop_part</ActuatorAction>
    #                     </Sequence>


    #                     <Sequence>
    #                         <Condition>is_agent_holding_good_part</Condition>
    #                         <StateAction>state_seek_base_area</StateAction>
    #                     </Sequence>


    #                     <Sequence>
    #                         <Condition>is_agent_holding_scrap_part</Condition>
    #                         <Condition>is_agent_in_waste_area</Condition>
    #                         <StateAction>state_movement_freeze</StateAction>
    #                     </Sequence>


    #                     <Sequence>
    #                         <Condition>is_agent_holding_scrap_part</Condition>
    #                         <StateAction>state_seek_waste_area</StateAction>
    #                     </Sequence>   


    #                     <Sequence>
    #                         <Condition>is_part_detected</Condition>
    #                         <ActuatorAction>pick_up_part</ActuatorAction>
    #                     </Sequence>       

    #                     <StateAction>state_seek_source_area</StateAction>
    #                 </Selector>
    #             </BehaviorTree>"""
    # }

    return [
        bring_good_base,
        bring_scrap_waste,
        bring_good_waste,
        bring_scrap_source,
        find_bad_stop,
        find_good_stop,
        find_base,
        # find_source_back,
        # find_waste,
        # find_part,
        # maintain_freeze
    ]