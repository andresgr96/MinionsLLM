""" 
    This script defines the dictionaries that will be used to automatically generate a technical prompt
 """

from typing import Dict

node_translations: Dict[str, str] = {
    "is_agent_in_base_area": "you are in the base",
    "is_agent_in_storage_area": "you are in the storage",
    "is_agent_in_construction_area": "you are in the construction",
    "is_agent_in_source_area": "you are in the source",
    "is_agent_in_waste_area": "you are in the waste",
    "is_part_detected": "you detect a part",
    "is_agent_holding_good_part": "you are holding a good part",
    "is_agent_holding_scrap_part": "you are holding a scrap part",
    "pick_up_part": "pick up the part",
    "drop_part": "drop the part",
    "state_seek_base_area": "go to the base",
    "state_seek_storage_area": "go to the storage",
    "state_seek_construction_area": "go to the construction",
    "state_seek_source_area": "go to the source",
    "state_seek_waste_area": "go to the waste",
    "state_movement_freeze": "stop moving",
    "state_random_walk": "search randomly",
}

spoon_node_translations: Dict[str, str] = {
    "is_agent_in_base_area": "you are in the base area",
    "is_agent_in_storage_area": "you are in the storage area",
    "is_agent_in_construction_area": "you are in the construction area",
    "is_agent_in_source_area": "you are in the source area",
    "is_agent_in_waste_area": "you are in the waste area",
    "is_part_detected": "you detect a part",
    "is_agent_holding_good_part": "you are holding a good part",
    "is_agent_holding_scrap_part": "you are holding a scrap part",
    "pick_up_part": "pick up the part",
    "drop_part": "drop the part",
    "state_seek_base_area": "seek the base area",
    "state_seek_storage_area": "seek the storage area",
    "state_seek_construction_area": "seek the construction area",
    "state_seek_source_area": "seek the source area",
    "state_seek_waste_area": "seek the waste area",
    "state_movement_freeze": "freeze movement",
    "state_random_walk": "walk randomly",
}

node_connectors: Dict[str, str] = {
    "Selector": "or",
    "Sequence": "and",
    "Condition": "if",
    "StateAction": "then",
    "ActuatorAction": "then",
}
