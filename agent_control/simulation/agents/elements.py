"""Part elements for robot simulation environment."""

import random
from typing import TYPE_CHECKING, Optional

from pygame import Vector2
from pygame.surface import Surface
from vi import Agent, HeadlessSimulation

if TYPE_CHECKING:
    from ..envs.robot_env import RobotEnvironment


class Part(Agent):  # type: ignore
    """Part object that can be picked up and moved by robot agents."""

    def __init__(
        self,
        images: list[Surface],
        simulation: HeadlessSimulation,
        type: str,
        pos: Vector2,
        env: "RobotEnvironment",
    ):
        """
        Initialize a part with type, position, and environment reference.

        Args:
            images: List of image surfaces for the part
            simulation: Simulation object this part belongs to
            type: Type of part ('good' or 'bad')
            pos: Initial position of the part
            env: Environment reference for metrics tracking
        """
        super().__init__(images, simulation, pos)
        self.move = Vector2(0, 0)
        self.images = images
        self.simulation = simulation
        self.type = type
        self.pos = pos
        self.owner: Optional[Agent] = None
        self.env = env
        self.is_permanently_placed = False
        self.update_img()

    def update_img(self) -> None:
        """Update the part's visual representation."""
        img = 0 if self.type == "good" else 1
        self.change_image(img)

    def remove_part(self) -> None:
        """Remove this part from the simulation."""
        self.kill()

    def update(self) -> None:
        """Update the part's position and state."""
        if self.owner and not self.is_permanently_placed:
            self.pos = self.owner.pos

    def can_be_picked_up(self) -> bool:
        """
        Check if this part can be picked up by an agent.

        Returns:
            bool: True if part can be picked up, False otherwise
        """
        return not self.is_permanently_placed and self.owner is None

    def pick_up_by(self, agent: Agent) -> bool:
        """
        Attempt to pick up this part by the specified agent.

        Args:
            agent: Agent attempting to pick up the part

        Returns:
            bool: True if pickup was successful, False otherwise
        """
        if self.can_be_picked_up():
            self.owner = agent

            # Handle pickup metrics
            if self.type == "good":
                self.env.good_parts_picked_up += 1
            elif self.type == "bad":
                self.env.bad_parts_picked_up += 1

            return True
        return False

    def drop_at_location(
        self,
        in_base: bool,
        in_waste: bool,
        in_storage: bool,
        in_construction: bool,
        in_source: bool,
    ) -> None:
        """
        Drop the part and update metrics based on location.

        Args:
            in_base: Whether the part is being dropped in base area
            in_waste: Whether the part is being dropped in waste area
            in_storage: Whether the part is being dropped in storage area
            in_construction: Whether the part is being dropped in construction area
            in_source: Whether the part is being dropped in source area
        """
        if self.owner is None:
            return

        noise_x = random.uniform(-10, 10)
        noise_y = random.uniform(-10, 10)

        # Determine part type index (0 for good, 1 for bad)
        part_idx = 0 if self.type == "good" else 1

        # Update metrics based on drop location
        if in_base:
            self.env.parts_dropped_in_base[part_idx] += 1
            if self.type == "good":
                self.env.nest_integrity += 7
            elif self.type == "bad":
                self.env.nest_integrity -= 7

            self.is_permanently_placed = True
            self.owner = self.env
            self.pos = Vector2(
                self.env.base_pos.x + noise_x, self.env.base_pos.y + noise_y
            )

            if in_storage:
                self.env.parts_dropped_in_storage[part_idx] += 1
                self.pos = Vector2(
                    self.env.storage_pos.x + noise_x, self.env.storage_pos.y + noise_y
                )

            elif in_construction:
                self.env.parts_dropped_in_construction[part_idx] += 1
                self.pos = Vector2(
                    self.env.construction_pos.x + noise_x,
                    self.env.construction_pos.y + noise_y,
                )

        elif in_waste:
            self.env.parts_dropped_in_waste[part_idx] += 1
            self.is_permanently_placed = True
            self.pos = Vector2(
                self.env.waste_pos.x + noise_x, self.env.waste_pos.y + noise_y
            )
            self.owner = self.env

        elif in_source:
            self.env.parts_dropped_in_source[part_idx] += 1
            self.is_permanently_placed = True
            self.pos = Vector2(
                self.env.source_pos.x + noise_x, self.env.source_pos.y + noise_y
            )
            self.owner = self.env
        else:
            # Dropped in some other area
            self.owner = None

        # Update total parts placed counter
        self.env.total_parts_placed += 1
