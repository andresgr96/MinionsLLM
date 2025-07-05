"""Sensor classes for robot agents."""

import random

from pygame.math import Vector2
from vi import Agent


class LightSensor:
    """Light sensor for detecting light sources in the environment."""

    def __init__(self, agent: Agent):
        """
        Initialize light sensor for the given agent.

        Args:
            agent: Agent that owns this sensor
        """
        self.agent = agent

    def sense_light(self) -> Vector2:
        """
        Sense light intensity at the agent's current position.

        Returns:
            Vector2: Direction vector to the nearest light source
        """
        light_pos = self.agent.env.light_pos
        agent_pos = self.agent.pos
        diff_vec: Vector2 = light_pos - agent_pos

        noise_x = random.uniform(-0.5, 0.5)
        noise_y = random.uniform(-0.5, 0.5)
        noise_vec = Vector2(noise_x, noise_y)

        return diff_vec.normalize() + noise_vec
