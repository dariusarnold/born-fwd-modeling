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


class VelocityModel:
    """
    Velocity model with a homogeneous background velocity that represents fractures
    as a collection of close scatterer points with a circular velocity heterogenity
    around them.
    """

    def __init__(self, background_velocity: float, fracture_velocity: float,
                 x_width: float, y_width: float,
                 scatterer_positions: Sequence[Vector3D], scatterer_radius: float):
        """
        Create new model
        :param background_velocity: unit m/s
        :param fracture_velocity: unit m/s
        :param x_width: unit m
        :param y_width: unit m
        :param scatterer_positions: All scatterer points in the model. A scatterer is a point in the
        model around which there is a velocity anomaly
        :param scatterer_radius: Radius of the velocity anomaly around a scatterer, unit m
        """
        self.bg_vel = background_velocity
        self.frac_vel = fracture_velocity
        self.x_width = x_width
        self.y_width = y_width
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
        within_scatterer_depth = self.scatterer_top <= position.z <= self.scatterer_bottom
        if within_scatterer_depth:
            # check if the position actually lies within the radius of a scatterer
            scatterer_indices = self.scatterer_tree.query_ball_point(position.data, self.scatterer_radius)
            if scatterer_indices:
                return self.frac_vel
            else:
                return self.bg_vel
        else:
            return self.bg_vel
