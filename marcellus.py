import itertools
from typing import Iterator, Tuple

import numpy as np

from bornfwd.units import Meter, MetersPerSecond, KgPerCubicMeter
from bornfwd.velocity_model import VelocityModel

"""
This file creates a velocity model for the marcellus shale. This model can be 
used to recreate data from the paper
3D seismic characterization of fractures in a dipping layer using the 
double-beam method (Hao Hu and Yingcai Zheng, 2018)
"""


def create_scatterers() -> np.ndarray:
    """
    Create a list of scatterer positions for the model
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
    """
    Create a complete velocity model
    """
    background_vel = MetersPerSecond(4800.)
    fracture_vel = MetersPerSecond(4320.)
    scatterer_positions = create_scatterers()
    scatterer_radius = Meter(1.)
    density = KgPerCubicMeter(2550.)
    return VelocityModel(scatterer_positions, background_vel, fracture_vel, scatterer_radius, density)
