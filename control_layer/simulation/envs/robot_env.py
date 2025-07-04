"""Robot simulation environment with parts and areas."""

import random
from typing import Any, Dict

from pygame.math import Vector2
from vi import Config

from ..agents.elements import Part
from ..agents.robot_agent import RobotAgent
from .base_env import SimEnvironment


class RobotEnvironment(SimEnvironment):
    """Robot simulation environment with parts, areas, and metrics tracking."""

    def __init__(
        self,
        config: Config,
        bt_path: str,
        n_agents: int,
        n_parts: int,
        task: str,
        headless: bool,
    ):
        """
        Initialize robot environment with configuration and metrics.

        Args:
            config: Configuration object for the simulation
            bt_path: Path to the behavior tree XML file
            n_agents: Number of agents to spawn
            n_parts: Number of parts to place
            task: Task description for the simulation
            headless: Whether to run without GUI
        """
        super().__init__(config, bt_path, headless)
        self.n_agents = n_agents
        self.n_parts = n_parts
        self.task = task
        self.headless = headless

        # Area Positions
        self.arena_pos = Vector2(250, 240)
        self.base_pos = Vector2(250, 325)
        self.storage_pos = Vector2(225, 325)
        self.construction_pos = Vector2(275, 325)
        self.source_pos = Vector2(75, 250)
        self.waste_pos = Vector2(425, 250)
        self.light_pos = Vector2(75, 250)

        # Metrics
        self.nest_integrity = 100
        self.good_parts_picked_up = 0
        self.bad_parts_picked_up = 0
        self.total_parts_placed = 0
        self.parts_dropped_in_base = [0, 0]  # [good, bad]
        self.parts_dropped_in_construction = [0, 0]  # [good, bad]
        self.parts_dropped_in_storage = [0, 0]  # [good, bad]
        self.parts_dropped_in_source = [0, 0]  # [good, bad]
        self.parts_dropped_in_waste = [0, 0]  # [good, bad]
        self.stopped_moving = False

        # Preloaded Images
        self.loaded_agent_images = self.load_images(
            [
                "./control_layer/simulation/images/white.png",
                "./control_layer/simulation/images/red circle.png",
                "./control_layer/simulation/images/green.png",
            ]
        )
        self.loaded_parts_imgs = self.load_images(
            [
                "./control_layer/simulation/images/part_green.png",
                "./control_layer/simulation/images/part_red.png",
            ]
        )

    def draw_obstacle(self) -> None:
        """Draw obstacle in the simulation environment."""
        x = 350
        y = 100
        self.simulation.spawn_obstacle(
            "./control_layer/simulation/images/rect_obst.png", x, y
        )

    def draw_arena(self) -> None:
        """Draw arena boundaries in the simulation environment."""
        self.simulation.spawn_obstacle(
            "./control_layer/simulation/images/arena_new.png",
            self.arena_pos.x,
            self.arena_pos.y,
        )

    def draw_source(self) -> None:
        """Draw source area where good parts are located."""
        self.simulation.spawn_site(
            "./control_layer/simulation/images/source_green.png",
            self.source_pos.x,
            self.source_pos.y,
        )

    def draw_nest(self) -> None:
        """Draw base/nest area where parts should be delivered."""
        self.simulation.spawn_site(
            "./control_layer/simulation/images/blue_nest.png",
            self.base_pos.x,
            self.base_pos.y,
        )

    def draw_waste(self) -> None:
        """Draw waste area where bad parts should be disposed."""
        self.simulation.spawn_site(
            "./control_layer/simulation/images/waste_red.png",
            self.waste_pos.x,
            self.waste_pos.y,
        )

    def spawn_part(self, type: str, pos: Vector2) -> None:
        """
        Spawn a part of specified type at given position.

        Args:
            type: Type of part to spawn ('good' or 'bad')
            pos: Position vector where to spawn the part
        """
        part = Part(
            images=self.loaded_parts_imgs,
            simulation=self.simulation,
            type=type,
            pos=pos,
            env=self,
        )
        self.simulation._agents.add(part)
        self.simulation._all.add(part)

    def remove_part(self, part: Part) -> None:
        """
        Remove a part from the simulation.

        Args:
            part: Part object to remove from simulation
        """
        part.kill()

    def place_parts(self, num_parts: int) -> None:
        """
        Place specified number of good and bad parts in the environment.

        Args:
            num_parts: Number of parts to place in the environment
        """
        for _ in range(num_parts):
            # Place the good parts
            rand_good_pos = Vector2(
                random.uniform(self.source_pos.x - 15, self.source_pos.x + 15),
                random.uniform(self.source_pos.y - 25, self.source_pos.y + 25),
            )
            good_part = Part(
                images=self.loaded_parts_imgs,
                simulation=self.simulation,
                type="good",
                pos=rand_good_pos,
                env=self,
            )
            self.simulation._agents.add(good_part)
            self.simulation._all.add(good_part)

            # Place the bad parts
            rand_bad_pos = Vector2(
                random.uniform(self.arena_pos.x - 100, self.arena_pos.x + 150),
                random.uniform(self.arena_pos.y - 50, self.arena_pos.y + 50),
            )
            bad_part = Part(
                images=self.loaded_parts_imgs,
                simulation=self.simulation,
                type="bad",
                pos=rand_bad_pos,
                env=self,
            )
            self.simulation._agents.add(bad_part)
            self.simulation._all.add(bad_part)

    def setup(self) -> None:
        """Set up the simulation environment with obstacles, areas, parts, and agents."""
        self.draw_arena()
        self.draw_nest()
        self.draw_source()
        self.draw_waste()
        self.place_parts(num_parts=self.n_parts)
        # self.spawn_part(type="good", pos=Vector2(250, 250))

        for _ in range(self.n_agents):
            noise_x = random.uniform(-10, 10)
            noise_y = random.uniform(-10, 10)
            agent = RobotAgent(
                images=self.loaded_agent_images,
                simulation=self.simulation,
                pos=Vector2(self.base_pos.x + noise_x, self.base_pos.y + noise_y),
                env=self,
                xml_path=self.xml_path,
            )

            self.simulation._agents.add(agent)
            self.simulation._all.add(agent)

    def run(self) -> Dict[str, Any]:
        """
        Run the simulation and return metrics.

        Returns:
            Dict[str, Any]: Dictionary containing simulation metrics
        """
        # Reset metrics at the start of each run for env reusability
        self.good_parts_picked_up = 0
        self.bad_parts_picked_up = 0
        self.total_parts_placed = 0
        self.nest_integrity = 100
        self.parts_dropped_in_base = [0, 0]  # [good, bad]
        self.parts_dropped_in_construction = [0, 0]  # [good, bad]
        self.parts_dropped_in_storage = [0, 0]  # [good, bad]
        self.parts_dropped_in_source = [0, 0]  # [good, bad]
        self.parts_dropped_in_waste = [0, 0]  # [good, bad]

        # Run the simulation
        self.simulation.run()

        # Prepare return metrics
        nest_integrity = self.nest_integrity
        nest_integrity = (
            100 if nest_integrity > 100 else 0 if nest_integrity < 0 else nest_integrity
        )

        return {
            "total_parts_placed": self.total_parts_placed,
            "good_parts_picked_up": self.good_parts_picked_up,
            "bad_parts_picked_up": self.bad_parts_picked_up,
            "nest_integrity": nest_integrity,
            "parts_dropped_in_base": self.parts_dropped_in_base.copy(),
            "parts_dropped_in_construction": self.parts_dropped_in_construction.copy(),
            "parts_dropped_in_storage": self.parts_dropped_in_storage.copy(),
            "parts_dropped_in_source": self.parts_dropped_in_source.copy(),
            "parts_dropped_in_waste": self.parts_dropped_in_waste.copy(),
            "stopped_moving": self.stopped_moving,
        }
