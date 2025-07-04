from vi import Agent
from pygame.math import Vector2
import random


class LightSensor():
    def __init__(self, agent: Agent):
        self.agent = agent

    def sense_light(self) -> Vector2:
        light_pos = self.agent.env.light_pos
        agent_pos = self.agent.pos
        diff_vec: Vector2 = light_pos - agent_pos

        noise_x = random.uniform(-0.5, 0.5)
        noise_y = random.uniform(-0.5, 0.5)
        noise_vec = Vector2(noise_x, noise_y)

        return diff_vec.normalize() + noise_vec