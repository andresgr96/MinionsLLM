import math
import pygame as pg
import random
from pygame.math import Vector2
from vi import Agent, Simulation
from parser_layer import parse_behavior_tree
from typing import List, TYPE_CHECKING
from .agent_sensors import LightSensor

if TYPE_CHECKING:
    from ..envs.foraging_env import ForagingEnvironment

class ForageAgent(Agent):
    def __init__(self, images: List[pg.Surface], simulation: Simulation, pos: Vector2, env: 'ForagingEnvironment', xml_path: str):
        super().__init__(images=images, simulation=simulation, pos=pos)
        self.simulation = simulation
        self.env = env
        self.task = self.env.task
        self.root_node = parse_behavior_tree(xml_path)
        self.light_sensor = LightSensor(self)
        self.state = "static" 
        self.is_agent_in_source_flag = False
        self.is_agent_in_nest_flag = True
        self.is_agent_holding_food_flag = False
        self.food_delivered = False
        self.turned_green = False
        self.total_food_delivered = 0
        self.food_timer = 0
        self.iterations = 0

        # self.move = Vector2(0, 0)

    def update(self) -> None:
        self.helper_update_flags()        # First we update the agents flags
        self.root_node.run(self)          # Then we get the state from the tree
        self.helper_control_from_state()  # After we calculate the control input from the state
        self.helper_check_task()      # Finally we check if the task has been solved
        self.iterations += 1
        # print(self.state)
        # print(f"Agent holding food: {self.is_agent_holding_food_flag}")

    # ---------------------------------------------------- Condition Nodes ----------------------------------------------------

    def is_agent_in_nest(self) -> bool:
        """
        Condition node: Checks whether the agent is in the nest area. Returns True if the agent is within the nest, and False otherwise.
        """
        if self.is_agent_in_nest_flag:
            return True
        
        return False

    def is_agent_in_source(self) -> bool:
        """
        Condition node: Checks whether the agent is in the source area where the food is. Returns True if the agent is within the source, and False otherwise.
        """
        if self.is_agent_in_source_flag:
            return True
        
        return False

    def is_agent_holding_food(self) -> bool:
        """
        Condition node: Checks whether the agent is holding food. Returns True if the agent is holding food, and False otherwise.
        """
        if self.is_agent_holding_food_flag:
            return True
                
        return False

    # ---------------------------------------------------- Action Nodes ----------------------------------------------------
    
    def move_towards_nest(self) -> bool:
        """
        Action node: Makes the agent move in the direction of the nest. Returns: Always returns True, indicating the action was executed.
        """
        self.state = "searching_nest"

        return True
    
    def move_towards_source(self) -> bool:
        """
        Action node: Makes the agent move in the direction of the source of food. Returns: Always returns True, indicating the action was executed.
        """
        self.state = "searching_source"

        return True

    def move_away_from_nest(self) -> bool:
        """
        Action node: Makes the agent move in the opposite direction of the nest. Returns: Always returns True, indicating the action was executed.
        """
        self.state = "searching_source"

        return True
    

    def move_away_from_source(self) -> bool:
        """
        Action node: Makes the agent move in the opposite direction of the source of food. Returns: Always returns True, indicating the action was executed.
        """
        self.state = "searching_nest"

        return True
    
    def wander(self) -> bool:
        """
        Action node: Makes the agent move in a random direction. Returns: Always returns True, indicating the action was executed.
        """
        self.state = "wandering"

        return True

    def stop_agent_movement(self) -> bool:
        """
        Action node: Freeze the agent's and stops its movement. Returns: Always returns True, indicating the action was executed.
        """
        self.state = "static"

        return True
    

    def change_color_to_green(self) -> bool:
        """
        Action node: Change the agent's color to green. Returns: Always returns True, indicating the action was executed.
        """
        self.change_image(2)
        self.turned_green = True
        return True     
    

    # ---------------------------------------------------- Helper Functions ----------------------------------------------------

    def helper_update_flags(self) -> None:
        """
        Updates the conditionals of the agent 
        """
        # Check for agent in nest, and if its holding food, updates metrics
        if self.on_site_id() == 0:
            self.is_agent_in_nest_flag = True

            if self.is_agent_holding_food_flag:
                # print("Searching for place to drop food")
                rand = random.randint(1, 5)
                self.food_timer += 1

                if self.food_timer >= rand:
                    # print(f"Food Dropped")
                    self.food_delivered = True
                    self.is_agent_holding_food_flag = False
                    self.total_food_delivered += 1
                    self.food_timer = 0
        else:
            self.is_agent_in_nest_flag = False

        # Check if agent is in the source, if not holding food, start procedure to hold it
        if self.on_site_id() == 1:
            self.is_agent_in_source_flag = True

            if not self.is_agent_holding_food_flag:
                # print("Searching for food")
                rand = random.randint(10, 20)
                self.food_timer += 1

                if self.food_timer >= rand:
                    # print("Food Found")
                    self.is_agent_holding_food_flag = True
                    self.food_timer = 0
        else:
            self.is_agent_in_source_flag = False

    def helper_check_task(self) -> None:
        """
        Checks if the task of the task has been completed.
        """
        task = self.task
        # print(task)

        if task == "search":
            if self.is_agent_holding_food_flag:
                # print(f"Solved: {task}") 
                self.env.success = True
                self.env.num_metric = self.iterations

        elif task == "source":
            if self.is_agent_in_source() and self.move == Vector2(0, 0):
                # print(f"Solved: {task}") 
                self.env.success = True
                self.env.num_metric = self.iterations
        else:
            if self.food_delivered:
                # print(f"Solved: {task}") 
                self.env.success = True
                self.env.num_metric = self.total_food_delivered


    def helper_control_from_state(self) -> None:
        """
        Produces a control command from the state of the agent defined by the behaviour tree.
        """
        state = self.state

        if state == "static":
            self.move = Vector2(0, 0)

        elif state == "searching_source":
            light_dir = self.light_sensor.sense_light()
            direction = light_dir.normalize()  

            if direction.length() > 0:
                direction.scale_to_length(self.config.movement_speed)
            self.move = direction

        elif state == "searching_nest":
            light_dir = self.light_sensor.sense_light()
            direction = -light_dir.normalize()  

            if direction.length() > 0:
                direction.scale_to_length(self.config.movement_speed)
            self.move = direction

        elif state == "wandering":
            angle = self.shared.prng_move.uniform(-math.pi, math.pi)  
            direction = Vector2(math.cos(angle), math.sin(angle))

            if direction.length() > 0:
                direction.scale_to_length(self.config.movement_speed)
            self.move = direction






            
            




