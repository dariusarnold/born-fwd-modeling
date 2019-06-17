import abc
import itertools
from typing import Tuple, Iterator

import numpy as np

from bornfwd.units import MetersPerSecond, KgPerCubicMeter, Meter


class AbstractVelocityModel(abc.ABC):
    """
    Interface expected from a class that can replace the default velocity model
    """
    # These attributes need to be present
    @property
    @abc.abstractmethod
    def background_velocity(self) -> MetersPerSecond:
        """Return homogeneous background velocity"""

    @property
    @abc.abstractmethod
    def fracture_velocity(self) -> MetersPerSecond:
        """Return velocity of fractures"""

    @property
    @abc.abstractmethod
    def density(self) -> KgPerCubicMeter:
        """Return homogeneous density"""

    @property
    @abc.abstractmethod
    def scatterer_radius(self) -> Meter:
        """Return radius of a single scatterer point"""

    @property
    @abc.abstractmethod
    def scatterer_positions(self) -> np.ndarray:
        """Return (N, 3) array of scatterer positions. The first axis contains
        all scatterer points, the second axis contains xyz coordinates of
        a point"""


class VelocityModel(AbstractVelocityModel):
    # Attributes from the ABC need to be listed here if they need to be
    # initialised in the init method
    background_velocity = None
    fracture_velocity = None
    density = None
    scatterer_radius = None
    scatterer_positions = None

    def __init__(self, scatterer_positions: np.ndarray,
                 background_velocity: MetersPerSecond,
                 fracture_velocity: MetersPerSecond, scatterer_radius: Meter,
                 density: KgPerCubicMeter):
        """
        Velocity model with a homogeneous background velocity that represents
        fractures as a collection of close scatterer points with a spherical
        velocity heterogenity around them. For documentation of the required
        parameters, see the AbstractVelocityModel class.
        A new velocity model has to derive from AbstractVelocityModel, this then
        checks if the necessary attributes for born modeling are defined.
        """
        self.background_velocity: MetersPerSecond = background_velocity
        self.fracture_velocity: MetersPerSecond = fracture_velocity
        self.density = density
        self.scatterer_positions = scatterer_positions.astype(np.float32)
        self.scatterer_radius = scatterer_radius
