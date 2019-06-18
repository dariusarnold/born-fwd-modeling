from pathlib import Path

import numpy as np

from bornfwd.plotting import plot_recording_geometry
from bornfwd.io import save_receiver_file, save_source_file

"""
Generate source- and receiverfile for marcellus shale recording geometry
described by Fracture parameter inversion for Marcellus Shale
(Mehdi E. Far, Bob Hardage, and Don Wagner)
"""


def generate_points(start_point: np.ndarray, end_point: np.ndarray,
                    num_of_points: int) -> np.ndarray:
    points_x = np.linspace(start_point[0], end_point[0], num_of_points)
    points_y = np.linspace(start_point[1], end_point[1], num_of_points)
    points_z = np.linspace(start_point[2], end_point[2], num_of_points)
    return np.vstack((points_x, points_y, points_z)).T


def generate_receiver_positions():

    # Coordinates specified in meters
    num_of_receiver_lines = 13
    receivers_per_line = 97
    # coordinates of leftmost receiver in Hu2018a fig 4a)
    first_line_start = np.array((3200, 5600, 0))  # left most geophone
    first_line_end = np.array((5400, 3400, 0))  # bottom geophone
    # coordinates of rightmost receiver line in Hu2018a fig4a)
    last_line_start = np.array((5400, 7900, 0))  # top geophone
    last_line_end = np.array((7560, 5600, 0))  # right most geophone

    startpoints = generate_points(first_line_start, last_line_start, num_of_receiver_lines)
    endpoints = generate_points(first_line_end, last_line_end, num_of_receiver_lines)
    lines = np.array([generate_points(startpoint, endpoint, receivers_per_line) for (startpoint, endpoint) in zip(startpoints, endpoints)])
    receivers = lines.reshape(num_of_receiver_lines*receivers_per_line, 3)
    return receivers


def generate_source_positions():
    # Coordinates specified in meters
    num_of_source_lines = 41
    sources_per_line = 60
    source_depth = 10
    first_line_start = np.array((0, 5200, source_depth))  # left most source
    first_line_end = np.array((5600, 11200, source_depth))  # top source
    last_line_start = np.array((5600, 0, source_depth))  # bottom source
    last_line_end = np.array((11200, 5600, source_depth))  # right source
    startpoints = generate_points(first_line_start, last_line_start, num_of_source_lines)
    endpoints = generate_points(first_line_end, last_line_end, num_of_source_lines)
    lines = np.array([generate_points(startpoint, endpoint, sources_per_line) for (startpoint, endpoint) in zip(startpoints, endpoints)])
    sources = lines.reshape(num_of_source_lines*sources_per_line, 3)
    return sources


if __name__ == '__main__':
    receivers = generate_receiver_positions()
    sources = generate_source_positions()
    plot_recording_geometry(sources, receivers)
    save_receiver_file(Path("receivers.txt"), receivers)
    save_source_file(Path("sources.txt"), sources)