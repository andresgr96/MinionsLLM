"""Robot agent implementation for behavior tree simulations."""

import math
from typing import TYPE_CHECKING, List

import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation

from tree_parser.middle_parser import parse_behavior_tree

from .agent_sensors import LightSensor
from .elements import Part

if TYPE_CHECKING:
    from ..envs.base_env import SimEnvironment


class RobotAgent(Agent):  # type: ignore
    """Robot agent that executes behavior trees in simulation environments."""

    def __init__(
        self,
        images: List[pg.Surface],
        simulation: Simulation,
        pos: Vector2,
        env: "SimEnvironment",
        xml_path: str,
    ):
        """
        Initialize robot agent with behavior tree and environment.

        Args:
            images: List of image surfaces for the agent
            simulation: Simulation object this agent belongs to
            pos: Initial position of the agent
            env: Environment reference for agent operations
            xml_path: Path to the behavior tree XML file
        """
        super().__init__(images=images, simulation=simulation, pos=pos)
        self.simulation = simulation
        self.env = env
        self.task = getattr(env, "task", None)
        self.root_node = parse_behavior_tree(xml_path)
        self.light_sensor = LightSensor(self)
        self.state = "static"
        self.is_agent_in_base_flag = True
        self.is_agent_in_source_flag = False
        self.is_agent_in_waste_flag = False
        self.is_agent_in_storage_flag = False
        self.is_agent_in_construction_flag = False
        self.is_agent_holding_good_part_flag = False
        self.is_agent_holding_bad_part_flag = False
        self.holding_any_part = False
        self.good_parts_correctly_placed = 0
        self.bad_parts_correctly_placed = 0
        self.total_parts_misplaced = 0
        self.rand_walk_cooldown = 0
        self.iterations = 0
        self.holding_part = None  # Reference to the part being held

    def update(self) -> None:
        """Update the agent's state and execute behavior tree."""
        # print("---")
        self.helper_update_flags()  # First we update the agents flags
        self.root_node.run(self)  # Then we get the state from the tree
        self.helper_control_from_state()  # After we calculate the control input from the state
        self.update_others()  # Finally we do any other updates here to keep it clean

    # ---------------------------------------------------- Condition Nodes ----------------------------------------------------

    def is_agent_in_base_area(self) -> bool:
        """
        Check if agent is in the base area.

        Node Type: Condition
        Description: Checks whether the agent is in the base area. Returns True if the agent is within the base, and False otherwise.
        Translation: you are in the base
        Spoon Translation: you are in the base area

        Returns:
            bool: True if agent is in base area, False otherwise
        """
        if self.is_agent_in_base_flag:
            return True

        return False

    def is_agent_in_construction_area(self) -> bool:
        """
        Check if agent is in the construction area.

        Node Type: Condition
        Description: Checks whether the agent is in the construction zone within the base area. Returns True if the agent is within the construction, and False otherwise.
        Translation: you are in the construction
        Spoon Translation: you are in the construction area

        Returns:
            bool: True if agent is in construction area, False otherwise
        """
        if self.is_agent_in_construction_flag:
            return True

        return False

    def is_agent_in_storage_area(self) -> bool:
        """
        Check if agent is in the storage area.

        Node Type: Condition
        Description: Checks whether the agent is in the storage zone within the base area. Returns True if the agent is within the storage, and False otherwise.
        Translation: you are in the storage
        Spoon Translation: you are in the storage area

        Returns:
            bool: True if agent is in storage area, False otherwise
        """
        if self.is_agent_in_storage_flag:
            return True

        return False

    def is_agent_in_source_area(self) -> bool:
        """
        Check if agent is in the source area.

        Node Type: Condition
        Description: Checks whether the agent is in the source area. Returns True if the agent is within the source, and False otherwise.
        Translation: you are in the source
        Spoon Translation: you are in the source area

        Returns:
            bool: True if agent is in source area, False otherwise
        """
        if self.is_agent_in_source_flag:
            return True

        return False

    def is_agent_in_waste_area(self) -> bool:
        """
        Check if agent is in the waste area.

        Node Type: Condition
        Description: Checks whether the agent is in the waste area. Returns True if the agent is within the waste, and False otherwise.
        Translation: you are in the waste
        Spoon Translation: you are in the waste area

        Returns:
            bool: True if agent is in waste area, False otherwise
        """
        if self.is_agent_in_waste_flag:
            return True

        return False

    def is_agent_holding_good_part(self) -> bool:
        """
        Check if agent is holding a good part.

        Node Type: Condition
        Description: Checks whether the agent is holding a good part. Returns True if the agent is holding a good part, and False otherwise.
        Translation: you are holding a good part
        Spoon Translation: you are holding a good part

        Returns:
            bool: True if agent is holding a good part, False otherwise
        """
        if self.is_agent_holding_good_part_flag:
            return True

        return False

    def is_agent_holding_scrap_part(self) -> bool:
        """
        Check if agent is holding a scrap part.

        Node Type: Condition
        Description: Checks whether the agent is holding a scrap part. Returns True if the agent is holding a scrap part, and False otherwise.
        Translation: you are holding a scrap part
        Spoon Translation: you are holding a scrap part

        Returns:
            bool: True if agent is holding a scrap part, False otherwise
        """
        if self.is_agent_holding_bad_part_flag:
            return True

        return False

    def is_good_part_detected(self) -> bool:
        """
        Check if a good part is detected nearby.

        Node Type: Condition
        Description: Checks whether the agent detects a good part within range to pick it up. Returns True if the agent is within range of a good part, and False otherwise.
        Translation: you detect a good part
        Spoon Translation: you detect a good part

        Returns:
            bool: True if good part is detected, False otherwise
        """
        nearby_parts = self.in_proximity_performance().filter_kind(Part)
        for part in nearby_parts:
            if part.can_be_picked_up() and part.type == "good":
                return True
        return False

    def is_scrap_part_detected(self) -> bool:
        """
        Check if a scrap part is detected nearby.

        Node Type: Condition
        Description: Checks whether the agent detects a scrap part within range to pick it up. Returns True if the agent is within range of a scrap part, and False otherwise.
        Translation: you detect a scrap part
        Spoon Translation: you detect a scrap part

        Returns:
            bool: True if scrap part is detected, False otherwise
        """
        nearby_parts = self.in_proximity_performance().filter_kind(Part)
        for part in nearby_parts:
            if part.can_be_picked_up() and part.type == "bad":
                return True
        return False

    # ---------------------------------------------------- Action Nodes ----------------------------------------------------

    def pick_up_part(self) -> bool:
        """
        Pick up a nearby part.

        Node Type: ActuatorAction
        Description: Makes the agent pick up a part if its within range and not already holding a part. Returns True if the agent picks up a part, and False otherwise.
        Translation: pick up the part
        Spoon Translation: pick up the part

        Returns:
            bool: True if part was successfully picked up, False otherwise
        """
        if (
            self.is_good_part_detected() or self.is_scrap_part_detected()
        ) and not self.holding_any_part:
            nearby_parts = self.in_proximity_performance().filter_kind(Part)

            # Look for a part that can actually be picked up
            for part in nearby_parts:
                if part.can_be_picked_up() and part.pick_up_by(self):
                    # Set the reference to the held part
                    self.holding_part = part

                    if part.type == "good":
                        self.is_agent_holding_good_part_flag = True
                    elif part.type == "bad":
                        self.is_agent_holding_bad_part_flag = True

                    self.holding_any_part = True
                    return True

            return False

        return False

    def drop_part(self) -> bool:
        """
        Drop the currently held part.

        Node Type: ActuatorAction
        Description: Makes the agent drop a part only if its holding one. Returns True if the agent drops a part, and False otherwise.
        Translation: drop the part
        Spoon Translation: drop the part

        Returns:
            bool: True if part was successfully dropped, False otherwise
        """
        if not self.holding_any_part or self.holding_part is None:
            return False

        # Get location information for metrics
        in_base = self.is_agent_in_base_flag
        in_construction = self.is_agent_in_construction_flag
        in_source = self.is_agent_in_source_flag
        in_waste = self.is_agent_in_waste_flag
        in_storage = self.is_agent_in_storage_flag

        # Drop the part with location context for metrics
        self.holding_part.drop_at_location(
            in_base, in_waste, in_storage, in_construction, in_source
        )

        # Reset agent state
        self.holding_part = None
        self.is_agent_holding_good_part_flag = False
        self.is_agent_holding_bad_part_flag = False
        self.holding_any_part = False

        return True

    def state_seek_base_area(self) -> bool:
        """
        Set state to seek the base area.

        Node Type: StateAction
        Description: Makes the agent move in the direction of the base. Returns True, indicating the action was executed.
        Translation: go to the base
        Spoon Translation: seek the base area

        Returns:
            bool: True if successfully moving towards base area
        """
        self.state = "searching_nest"

        return True

    def state_seek_storage_area(self) -> bool:
        """
        Set state to seek the storage area.

        Node Type: StateAction
        Description: Makes the agent move in the direction of the storage zone within the base. Returns True, indicating the action was executed.
        Translation: go to the storage
        Spoon Translation: seek the storage area

        Returns:
            bool: True if successfully moving towards storage area
        """
        self.state = "searching_storage"

        return True

    def state_seek_construction_area(self) -> bool:
        """
        Set state to seek the construction area.

        Node Type: StateAction
        Description: Makes the agent move in the direction of the construction zone within the base. Returns True, indicating the action was executed.
        Translation: go to the construction
        Spoon Translation: seek the construction area

        Returns:
            bool: True if successfully moving towards construction area
        """
        self.state = "searching_repair"

        return True

    def state_seek_waste_area(self) -> bool:
        """
        Set state to seek the waste area.

        Node Type: StateAction
        Description: Makes the agent move in the opposite direction of light where the waste area is found. Returns True, indicating the action was executed.
        Translation: go to the waste
        Spoon Translation: seek the waste area

        Returns:
            bool: True if successfully moving towards waste area
        """
        self.state = "searching_waste"

        return True

    def state_seek_source_area(self) -> bool:
        """
        Set state to seek the source area.

        Node Type: StateAction
        Description: Makes the agent move in the direction of the light source where the source area is found. Returns True, indicating the action was executed.
        Translation: go to the source
        Spoon Translation: seek the source area

        Returns:
            bool: True if successfully moving towards source area
        """
        self.state = "searching_source"

        return True

    def state_random_walk(self) -> bool:
        """
        Set state to random walk.

        Node Type: StateAction
        Description: Makes the agent move in a random direction. Returns True, indicating the action was executed.
        Translation: search randomly
        Spoon Translation: walk randomly

        Returns:
            bool: True if random walk was performed successfully
        """
        self.state = "wandering"

        return True

    def state_movement_freeze(self) -> bool:
        """
        Set state to freeze movement.

        Node Type: StateAction
        Description: Freeze the agent's and stops its movement. Returns True, indicating the action was executed.
        Translation: stop moving
        Spoon Translation: freeze movement

        Returns:
            bool: True if movement was successfully frozen
        """
        self.state = "static"

        return True

    # ---------------------------------------------------- Helper Functions ----------------------------------------------------

    def helper_update_flags(self) -> None:
        """Update the conditionals of the agent."""
        # Check for agent in base, if it is, check in which zone
        if self.on_site_id() == 0:
            self.is_agent_in_base_flag = True
            if self.pos.x <= 250:
                self.is_agent_in_construction_flag = False
                self.is_agent_in_storage_flag = True
                # print("storage")
            else:
                self.is_agent_in_storage_flag = False
                self.is_agent_in_construction_flag = True
                # print("construction")
        else:
            self.is_agent_in_base_flag = False
            self.is_agent_in_storage_flag = False
            self.is_agent_in_construction_flag = False

        # Check if agent is in the source
        if self.on_site_id() == 1:
            self.is_agent_in_source_flag = True
            # print("source")
        else:
            self.is_agent_in_source_flag = False

        # Check if agent is in the waste
        if self.on_site_id() == 2:
            self.is_agent_in_waste_flag = True
            # print("waste")
        else:
            self.is_agent_in_waste_flag = False

    def update_others(self) -> None:
        """Perform different updates that don't fit into the other helper functions."""
        self.iterations += 1
        self.rand_walk_cooldown += 1

        if self.rand_walk_cooldown >= 20:
            self.rand_walk_cooldown = 0

        if self.iterations % 140 == 0:
            # Safely access nest_integrity if it exists (specific to RobotEnvironment)
            if hasattr(self.env, "nest_integrity"):
                self.env.nest_integrity -= 1

        # Update agent color for style
        if self.is_agent_holding_good_part_flag:
            self.change_image(2)
        elif self.is_agent_holding_bad_part_flag:
            self.change_image(1)
        else:
            self.change_image(0)

    def helper_control_from_state(self) -> None:
        """Produce a control command from the state of the agent defined by the behaviour tree."""
        state = self.state

        try:
            if state == "static":
                self.move = Vector2(0, 0)
                # Safely access stopped_moving if it exists (specific to RobotEnvironment)
                if hasattr(self.env, "stopped_moving"):
                    self.env.stopped_moving = True

            elif state == "searching_source":
                light_dir = self.light_sensor.sense_light()
                direction = light_dir.normalize()
                if direction.length() > 0:
                    direction.scale_to_length(self.config.movement_speed)

                self.move = direction

            elif state == "searching_waste":
                light_dir = self.light_sensor.sense_light()
                direction_x = -light_dir.x
                direction_y = (
                    light_dir.y
                )  # Pygames inverted y-axis sheananigans forces this
                direction = Vector2(direction_x, direction_y).normalize()
                if direction.length() > 0:
                    direction.scale_to_length(self.config.movement_speed)

                self.move = direction

            elif state == "searching_nest":
                base_pos = getattr(self.env, "base_pos", Vector2(250, 325))
                direction = self.helper_direction_to(base_pos)
                self.move = direction

            elif state == "searching_storage":
                storage_pos = getattr(self.env, "storage_pos", Vector2(225, 325))
                direction = self.helper_direction_to(storage_pos)
                self.move = direction

            elif state == "searching_repair":
                construction_pos = getattr(
                    self.env, "construction_pos", Vector2(275, 325)
                )
                direction = self.helper_direction_to(construction_pos)
                self.move = direction

            elif state == "wandering":
                if self.rand_walk_cooldown == 0:
                    # print("-------------------rand walk")
                    angle = self.shared.prng_move.uniform(-math.pi, math.pi)
                    center_dir = self.helper_direction_to(Vector2(250, 250))
                    direction = Vector2(math.cos(angle), math.sin(angle))
                    direction = (
                        center_dir + direction
                    )  # Bias the random walk towards the center
                    if direction.length() > 0:
                        direction.scale_to_length(self.config.movement_speed)

                    self.move = direction
        except ValueError:
            self.move = Vector2(0, 0)

    def helper_direction_to(self, target: Vector2) -> Vector2:
        """
        Calculate direction vector to target position.

        Args:
            target: Target position vector

        Returns:
            Vector2: Normalized direction vector towards target
        """
        try:
            dir = target - self.pos
            direction = dir.normalize()
            if direction.length() > 0:
                direction.scale_to_length(self.config.movement_speed)

            return Vector2(direction.x, direction.y)
        except ValueError:
            return Vector2(0, 0)
