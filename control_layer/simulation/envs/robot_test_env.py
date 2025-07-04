from .base_env import SimEnvironment
from pygame.math import Vector2
from vi import Config
from ..agents.robot_test_agent import RobotTestAgent
from ..agents.elements import Part 
import pygame as pg
import random

class RobotTestEnvironment(SimEnvironment):
    def __init__(self, config: Config, bt_path: str, n_agents: int, task: str, headless: bool):
        super().__init__(config, bt_path, headless)
        self.n_agents = n_agents
        self.arena_pos = Vector2(250, 240)
        self.nest_pos = Vector2(250, 325)
        self.storage_pos = Vector2(225, 325)
        self.repair_pos = Vector2(275, 325)
        self.source_pos = Vector2(75, 250)
        self.waste_pos = Vector2(425, 250)
        self.light_pos = Vector2(75, 250)
        self.task = task
        self.headless = headless
        self.success = False
        self.nest_integrity = 100
        self.num_metric = 0
        self.good_parts_correctly_placed = 0
        self.bad_parts_correctly_placed = 0
        self.total_parts_placed = 0
        
        self.loaded_agent_images = self.load_images(["./images/white.png", "./images/red circle.png", "./images/green.png"])
        self.loaded_parts_imgs = self.load_images(["./images/part_green.png", "./images/part_red.png"])

    def load_images(self, image_paths):
        return [pg.image.load(path).convert_alpha() for path in image_paths] if not self.headless \
            else [pg.image.load(path) for path in image_paths]

    def draw_obstacle(self) -> None:
        x = 350
        y = 100
        self.simulation.spawn_obstacle("./images/rect_obst.png", x, y)

    def draw_arena(self) -> None:
        self.simulation.spawn_obstacle("./images/arena_new.png", self.arena_pos.x, self.arena_pos.y)

    def draw_source(self) -> None:
        self.simulation.spawn_site("./images/source_green.png", self.source_pos.x, self.source_pos.y)

    def draw_nest(self) -> None:
        self.simulation.spawn_site("./images/blue_nest.png", self.nest_pos.x, self.nest_pos.y)

    def draw_waste(self) -> None:
        self.simulation.spawn_site("./images/waste_red.png", self.waste_pos.x, self.waste_pos.y)

    def spawn_part(self, type: str, pos: Vector2) -> None:
        part = Part(images=self.loaded_parts_imgs, simulation=self.simulation, type=type, pos=pos, env=self)
        self.simulation._agents.add(part)
        self.simulation._all.add(part)

    def remove_part(self, part: Part) -> None:
        part.kill()

    def place_parts(self, num_parts: int) -> None:
        for _ in range(num_parts):
            # Place the good parts
            rand_good_pos = Vector2(random.uniform(self.source_pos.x - 15, self.source_pos.x + 15), random.uniform(self.source_pos.y - 25, self.source_pos.y + 25))
            good_part = Part(images=self.loaded_parts_imgs, simulation=self.simulation, type="good", pos=rand_good_pos, env=self)
            self.simulation._agents.add(good_part)
            self.simulation._all.add(good_part)    

            # Place the bad parts
            rand_bad_pos = Vector2(random.uniform(self.arena_pos.x - 165, self.arena_pos.x + 150), random.uniform(self.arena_pos.y - 50, self.arena_pos.y + 50))
            bad_part = Part(images=self.loaded_parts_imgs, simulation=self.simulation, type="bad", pos=rand_bad_pos, env=self)
            self.simulation._agents.add(bad_part)
            self.simulation._all.add(bad_part)


    def setup(self) -> None:
        self.draw_arena()
        self.draw_nest()
        self.draw_source()
        self.draw_waste()
        self.place_parts(num_parts=10)
        # self.spawn_part(type="good", pos=self.nest_pos)
        
        for _ in range(self.n_agents):
            noise_x = random.uniform(-10, 10)
            noise_y = random.uniform(-10, 10)
            agent = RobotTestAgent(images=self.loaded_agent_images, simulation=self.simulation,
                                pos=Vector2(self.nest_pos.x + noise_x, self.nest_pos.y + noise_y), env=self, xml_path=self.xml_path)
            
            self.simulation._agents.add(agent)
            self.simulation._all.add(agent)



    def run(self) -> bool:
        self.simulation.run()
        print(f"Total Parts Placed: {self.total_parts_placed}")
        print(f"Good Parts Placed: {self.good_parts_correctly_placed}")
        print(f"Bad Parts Placed: {self.bad_parts_correctly_placed}")
        return self.success, self.num_metric
