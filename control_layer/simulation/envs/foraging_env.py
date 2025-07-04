from .base_env import SimEnvironment
from pygame.math import Vector2
from vi import Config
from ..agents.forage_agent import ForageAgent
from ..agents.elements import Food
import pygame as pg
import random

class ForagingEnvironment(SimEnvironment):
    def __init__(self, config: Config, bt_path: str, n_agents: int, task: str, headless: bool):
        super().__init__(config, bt_path, headless)
        self.n_agents = n_agents
        self.arena_pos = Vector2(250, 250)
        self.nest_pos = Vector2(250, 400)
        self.source_pos = Vector2(250, 100)
        self.light_pos = Vector2(250, 100)
        self.task = task
        self.headless = headless
        self.success = False
        self.num_metric = 0

        self.loaded_agent_images = self.load_images(["./images/red circle.png", "./images/white.png", "./images/green.png"])
        self.loaded_food_img = self.load_images(["./images/food_med.png"])

    def load_images(self, image_paths):
        return [pg.image.load(path).convert_alpha() for path in image_paths] if not self.headless \
            else [pg.image.load(path) for path in image_paths]

    def draw_obstacle(self) -> None:
        x = 350
        y = 100
        self.simulation.spawn_obstacle("./images/rect_obst.png", x, y)

    def draw_arena(self) -> None:
        self.simulation.spawn_obstacle("./images/arena_large2.png", self.arena_pos.x, self.arena_pos.y)

    def draw_source(self) -> None:
        self.simulation.spawn_site("./images/source_white.png", self.source_pos.x, self.source_pos.y)

    def draw_nest(self) -> None:
        self.simulation.spawn_site("./images/nest_green.png", self.nest_pos.x, self.nest_pos.y)

    def spawn_food(self) -> None:
        food = Food(images=self.loaded_food_img, simulation=self.simulation, pos=self.source_pos, env=self)
        self.simulation._agents.add(food)
        self.simulation._all.add(food)

    def setup(self) -> None:
        self.draw_arena()
        self.draw_nest()
        self.draw_source()
        
        for _ in range(self.n_agents):
            noise_x = random.uniform(-30, 30)
            noise_y = random.uniform(-30, 30)
            noise_vec = Vector2(noise_x, noise_y)
            agent = ForageAgent(images=self.loaded_agent_images, simulation=self.simulation,
                                pos=Vector2(self.nest_pos.x, self.nest_pos.y), env=self, xml_path=self.xml_path)
            
            self.simulation._agents.add(agent)
            self.simulation._all.add(agent)

            food = Food(images=self.loaded_food_img, simulation=self.simulation, pos=self.source_pos + noise_vec, env=self)
            self.simulation._agents.add(food)
            self.simulation._all.add(food)

    def run(self) -> bool:
        self.simulation.run()
        return self.success, self.num_metric
