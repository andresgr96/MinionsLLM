import random
from typing import TYPE_CHECKING, Optional

from pygame import Vector2
from pygame.surface import Surface
from vi import Agent, HeadlessSimulation

if TYPE_CHECKING:
    from ..envs.robot_env import RobotEnvironment


class Part(Agent):  # type: ignore
    def __init__(
        self,
        images: list[Surface],
        simulation: HeadlessSimulation,
        type: str,
        pos: Vector2,
        env: "RobotEnvironment",
    ):
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
        img = 0 if self.type == "good" else 1
        self.change_image(img)

    def remove_part(self) -> None:
        self.kill()

    def update(self) -> None:
        if self.owner and not self.is_permanently_placed:
            self.pos = self.owner.pos

    def can_be_picked_up(self) -> bool:
        return not self.is_permanently_placed and self.owner is None

    def pick_up_by(self, agent: Agent) -> bool:
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
        """Drop the part and update metrics based on location"""
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
