import abc
import itertools

import numpy as np
from typing import Tuple, Sequence

import scipy.spatial


class Vector3D:
    """General 3D vector that can be used with numpy operations but makes code
    using it easier to read because it allows attribute access to the xyz
    components."""

    def __init__(self, x: float, y: float, z: float):
        self.data = (x, y, z)

    # implement to support numpy operations on this class
    def __array__(self) ->np.array:
        return np.asarray(self.data)

    @property
    def x(self) -> float:
        return self.data[0]

    @property
    def y(self) -> float:
        return self.data[1]

    @property
    def z(self) -> float:
        return self.data[2]

    # implement len and getitem for the conversion of a sequence of Vector3Ds
    # to a nested numpy array to work
    def __len__(self) -> int:
        return 3

    def __getitem__(self, item: int) -> float:
        return self.data[item]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.z})"

    def __sub__(self, other):
        """Implement subtraction operator"""
        return Vector3D(self.x-other.x, self.y-other.y, self.z-other.z)


class AbstractVelocityModel(abc.ABC):
    """
    Interface expected from a class that can replace the default velocity model
    """
    @abc.abstractmethod
    def eval_at(self, position: Tuple[float, float, float]) -> float:
        """Return velocity (m/s) at coordinate x y z (m)"""
        pass

    # These attributes need to be present
    @property
    @abc.abstractmethod
    def density(self) -> float:
        """Return homogeneous density in kg/m^3"""
        pass

    @property
    @abc.abstractmethod
    def x_width(self) -> float:
        """Return width of model along x axis in m"""
        pass

    @property
    @abc.abstractmethod
    def y_width(self) -> float:
        """Return width of model along y axis in m"""
        pass

    @property
    @abc.abstractmethod
    def scatterer_top(self) -> float:
        """Return depth of uppermost scatterer in m (with the radius) """
        pass

    @property
    @abc.abstractmethod
    def scatterer_bottom(self) -> float:
        """Return depth of bottom most scatterer in m(with the radius)"""
        pass



class VelocityModel(AbstractVelocityModel):
    """
    Velocity model with a homogeneous background velocity that represents fractures
    as a collection of close scatterer points with a circular velocity heterogenity
    around them.
    """
    # Attributes from the ABC need to be listed here if they need to be initialised in the
    # init method
    density = None
    x_width = None
    y_width = None
    scatterer_top = None
    scatterer_bottom = None

    def __init__(self, background_velocity: float, fracture_velocity: float, x_width: float,
                 y_width: float, scatterer_positions: Sequence[Vector3D], scatterer_radius: float,
                 density: float):
        """
        Create new model
        :param background_velocity: unit m/s
        :param fracture_velocity: unit m/s
        :param x_width: unit m
        :param y_width: unit m
        :param scatterer_positions: All scatterer points in the model. A scatterer is a point in the
        model around which there is a velocity anomaly
        :param scatterer_radius: Radius of the velocity anomaly around a scatterer, unit m
        :param density: Homogeneous background density of model in kg/m^3
        """
        self.bg_vel = background_velocity
        self.frac_vel = fracture_velocity
        self.x_width = x_width
        self.y_width = y_width
        self.density = density
        self.scatterer_tree = scipy.spatial.cKDTree(np.asarray(scatterer_positions))
        self.scatterer_positions = scatterer_positions
        self.scatterer_radius = scatterer_radius
        self.scatterer_depth = scatterer_positions[0].z
        self.scatterer_top = self.scatterer_depth - scatterer_radius
        self.scatterer_bottom = self.scatterer_depth + scatterer_radius

    def eval_at(self, position: Vector3D) -> float:
        """Get velocity at position in the model"""
        # see if depth is on scatterer layer, if no we don't need to query the
        # KDTree
        within_scatterer_depth = self.scatterer_top <= position[2] <= self.scatterer_bottom
        if not within_scatterer_depth:
            return self.bg_vel
        # check if the position actually lies within the radius of a scatterer
        scatterer_indices = self.scatterer_tree.query_ball_point(position, self.scatterer_radius)
        if scatterer_indices:
            return self.frac_vel
        else:
            return self.bg_vel


def create_scatterers() -> Sequence[Vector3D]:
    """
    Create a list of scatterer positions for the model described in 3D seismic characterization of
    fractures in a dipping layer using the double-beam method (Hao Hu and Yingcai Zheng)
    :return:
    """
    model_width = 11200.
    # values taken from Fig4 in Hu2018a, all units m
    frac_depth = 2400.
    # fractures are in a square with sides parallel to the axes
    # values taken from Fig 4 Hu2018a
    frac_top_left_x = 600.
    frac_top_left_y = model_width - 1300
    frac_bot_right_x = model_width - 1400
    frac_bot_right_y = 860.
    frac_spacing = 200.
    # distance between scatterer points, calculated as 1/8 p wave length
    scatterer_spacing = 20.
    # + 1 to cover the "last" point
    number_of_fractures = int((frac_top_left_y - frac_bot_right_y) / frac_spacing) + 1
    scatterers_per_fracture = int((frac_bot_right_x - frac_top_left_x) / scatterer_spacing) + 1
    # step parallel to y axis
    scatterer_y_positions = np.linspace(frac_bot_right_y, frac_top_left_y, number_of_fractures)
    # step parallel to x axis
    scatterer_x_positions = np.linspace(frac_top_left_x, frac_bot_right_x, scatterers_per_fracture)
    xy_combinations = itertools.product(scatterer_x_positions, scatterer_y_positions)
    scatterers = [Vector3D(x, y, frac_depth) for x, y in xy_combinations]
    return scatterers


def create_velocity_model() -> VelocityModel:
    background_vel = 4800.
    fracture_vel = 4320.
    x_width = 11200.
    y_width = 11200.
    scatterer_positions = create_scatterers()
    scatterer_radius = 100.
    density = 2550.
    return VelocityModel(background_vel, fracture_vel, x_width, y_width, scatterer_positions,
                         scatterer_radius, density)
