import math
import pygame as pg
import random
from pygame.math import Vector2
from vi import Agent, Simulation
from parser_layer.middle_parser import parse_behavior_tree
from typing import List, TYPE_CHECKING
from .agent_sensors import LightSensor
from .elements import Part

if TYPE_CHECKING:
    from ..envs.robot_env import RobotEnvironment

class RobotTestAgent(Agent):
    def __init__(self, images: List[pg.Surface], simulation: Simulation, pos: Vector2, env: 'RobotEnvironment', xml_path: str):
        super().__init__(images=images, simulation=simulation, pos=pos)
        self.simulation = simulation
        self.env = env
        self.task = self.env.task
        self.root_node = parse_behavior_tree(xml_path)
        self.light_sensor = LightSensor(self)
        self.state = "static" 
        self.is_agent_in_nest_flag = True
        self.is_agent_in_source_flag = False
        self.is_agent_in_waste_flag = False
        self.is_agent_in_storage_flag = False
        self.is_agent_in_repair_flag = False
        self.is_agent_holding_good_part_flag = False
        self.is_agent_holding_bad_part_flag = False
        self.food_delivered = False
        self.turned_white = True
        self.turned_green = False
        self.turned_red = False
        self.good_parts_correctly_placed = 0
        self.bad_parts_correctly_placed = 0
        self.total_parts_misplaced = 0
        self.iterations = 0


    def update(self) -> None:
        self.root_node.run(self)          # Then we get the state from the tree
        self.iterations += 1
        print("---")
        if self.iterations % 10 == 0:
            self.env.nest_integrity -= 1

    # ---------------------------------------------------- Condition Nodes ----------------------------------------------------

    def condition_true(self) -> bool:
        """
        Condition node: Checks whether the agent is in the nest area. Returns True if the agent is within the nest, and False otherwise.
        """

        print("True Cond")
        
        return True
    
    def condition_false(self) -> bool:
        """
        Condition node: Checks whether the agent is in the nest area. Returns True if the agent is within the nest, and False otherwise.
        """

        print("False Cond")
        
        return False

    # ---------------------------------------------------- Action Nodes ----------------------------------------------------

    def action_one(self) -> bool:
        """
        Action node: Makes the agent move in a random direction. Returns: Always returns True, indicating the action was executed.
        """
        print("Action One")

        return True

    def action_two(self) -> bool:
        """
        Action node: Makes the agent move in a random direction. Returns: Always returns True, indicating the action was executed.
        """
        print("Action Two")

        return True
    
    def action_three(self) -> bool:
        """
        Action node: Makes the agent move in a random direction. Returns: Always returns True, indicating the action was executed.
        """
        print("Action Three")

        return True
    
    def action_four(self) -> bool:
        """
        Action node: Makes the agent move in a random direction. Returns: Always returns True, indicating the action was executed.
        """
        print("Action Four")

        return True








            
            




