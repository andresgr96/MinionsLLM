import pygame as pg
from abc import ABC, abstractmethod
from vi import Simulation, Config, HeadlessSimulation
from typing import List, Union, Dict, Any

class SimEnvironment(ABC):
    """
    Abstract base class for all simulation environments.
    Defines the essential interface that all environments must implement.
    """
    
    def __init__(self, config: Config, bt_path: str, headless: bool = False):
        self.config = config
        self.simulation: Union[Simulation, HeadlessSimulation] = (
            Simulation(config) if not headless else HeadlessSimulation(config)
        )
        self.xml_path = bt_path
        self.headless = headless

    @abstractmethod
    def setup(self) -> None:
        """
        Set up the environment by spawning agents, obstacles, etc.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the simulation and return metrics/results.
        Must be implemented by all subclasses. 
        Should ALWAYS include a call to self.simulation.run()
        
        Returns:
            Dict[str, Any]: Dictionary containing simulation results and metrics
        """
        pass

    def load_images(self, image_paths: List[str]) -> List[pg.Surface]:
        """Load images for the environment, handling headless mode."""
        if self.headless:
            return [pg.image.load(path) for path in image_paths]
        else:
            return [pg.image.load(path).convert_alpha() for path in image_paths]