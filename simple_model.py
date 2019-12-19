from pathlib import Path

import numpy as np

from bornfwd.velocity_model import VelocityModel
from bornfwd.units import Meter, MetersPerSecond, KgPerCubicMeter
from bornfwd.io import save_receiver_file, save_source_file

"""
This file contains a simple model with only two fracture planes.
"""


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def generate_grid(top_left: Point, bottom_right: Point, depth: float,
                  num_x: int, num_y: int) -> np.ndarray:
    x_coords = np.linspace(top_left.x, bottom_right.x, num_x)
    y_coords = np.linspace(top_left.y, bottom_right.y, num_y)
    points = np.empty((num_x*num_y, 3))
    # set x coord
    points[:, 0] = np.concatenate((x_coords,)*num_y)
    # set y coord
    points[:, 1] = np.concatenate((y_coords,)*num_x)
    # set z coord
    points[:, 2] = depth
    return points


def create_scatterers() -> np.ndarray:
    """
    Create two fractures, spaced 200 m apart, with a length of 600 m, parallel
    to the y axis at a depth of 500 m.
    """
    scatterer_distance = 5
    fracture_length = 600
    scatterers_per_fracture = int(fracture_length / scatterer_distance) + 1
    fracture_y_bottom = 200
    fracture_y_top = 800
    y_positions = np.linspace(fracture_y_bottom, fracture_y_top,
                              scatterers_per_fracture)
    fracture_1_x = 400
    fracture_2_x = 800
    fracture_depth = 500
    scatterers = np.empty((2*scatterers_per_fracture, 3))
    # set x values of first fracture
    scatterers[0:scatterers_per_fracture, 0] = fracture_1_x
    # set x values of second fracture
    scatterers[scatterers_per_fracture:2*scatterers_per_fracture+1, 0] = fracture_2_x
    # set y values
    scatterers[:, 1] = np.concatenate((y_positions, y_positions))
    # set z values
    scatterers[:, 2] = fracture_depth
    return scatterers[0:scatterers_per_fracture, :]


def create_velocity_model() -> VelocityModel:
    background_velocity = MetersPerSecond(3000)
    scatterer_velocity = MetersPerSecond(2600)
    density = KgPerCubicMeter(2700)
    scatterer_radius = Meter(0.1)
    scatterer_positions = create_scatterers()
    model = VelocityModel(scatterer_positions, background_velocity,
                          scatterer_velocity, scatterer_radius, density)
    return model


def create_receivers():
    receivers = generate_grid(Point(50, 50), Point(950, 950), 0, 20, 20)
    save_receiver_file("simple_receivers.txt", receivers)


def create_source_file():
    sources = generate_grid(Point(100, 100), Point(900, 900), 0, 25, 25)
    save_source_file(Path("simple_sources.txt"), sources)


if __name__ == '__main__':
    create_receivers()
    create_source_file()