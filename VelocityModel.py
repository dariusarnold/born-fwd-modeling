import abc
import itertools
from typing import Tuple, Iterator

import numpy as np

from units import MetersPerSecond, KgPerCubicMeter, Meter


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
        velocity heterogenity around them.For documentation of the required
        parameters, see the AbstractVelocityModel class
        """
        self.background_velocity: MetersPerSecond = background_velocity
        self.fracture_velocity: MetersPerSecond = fracture_velocity
        self.density = density
        self.scatterer_positions = scatterer_positions
        self.scatterer_radius = scatterer_radius


def create_scatterers() -> np.ndarray:
    """
    Create a list of scatterer positions for the model described in 3D seismic
    characterization of fractures in a dipping layer using the double-beam
    method (Hao Hu and Yingcai Zheng, 2018)
    """
    model_width = Meter(11200.)
    # values taken from Fig4 in Hu2018a, all units m
    frac_depth = Meter(2400.)
    # fractures are in a square with sides parallel to the axes
    # values taken from Fig 4 Hu2018a
    frac_top_left_x = Meter(600.)
    frac_top_left_y = model_width - Meter(1300)
    frac_bot_right_x = model_width - Meter(1400)
    frac_bot_right_y = Meter(860.)
    frac_spacing = Meter(200.)
    # distance between scatterer points, calculated as 1/8 p wave length
    scatterer_spacing = Meter(20.)
    # + 1 to cover the "last" point
    number_of_fractures = int((frac_top_left_y - frac_bot_right_y) / frac_spacing) + 1
    scatterers_per_fracture = int((frac_bot_right_x - frac_top_left_x) / scatterer_spacing) + 1
    # step parallel to y axis
    scatterer_y_positions = np.linspace(frac_bot_right_y, frac_top_left_y, number_of_fractures)
    # step parallel to x axis
    scatterer_x_positions = np.linspace(frac_top_left_x, frac_bot_right_x, scatterers_per_fracture)
    xy_combinations: Iterator[Tuple[Meter, Meter]] = itertools.product(scatterer_x_positions, scatterer_y_positions)
    scatterers = np.array([(x, y, frac_depth) for x, y in xy_combinations])
    assert len(scatterers) == 21206
    return scatterers


def create_velocity_model() -> VelocityModel:
    background_vel = MetersPerSecond(4800.)
    fracture_vel = MetersPerSecond(4320.)
    scatterer_positions = create_scatterers()
    scatterer_radius = Meter(1.)
    density = KgPerCubicMeter(2550.)
    return VelocityModel(scatterer_positions, background_vel, fracture_vel, scatterer_radius, density)
