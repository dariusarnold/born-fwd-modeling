from pathlib import Path

import numpy as np


def read_stations(filepath: Path) -> np.ndarray:
    """
    Read file defining receiver positions.
    :param filepath:
    :return: Array containing all receiver positions with shape (N, 3).
    N is the number of receivers defined in the file (first line), 3 are the
    x, y, z coordinates of the individual receivers.
    """
    stations = np.genfromtxt(filepath, skip_header=1, usecols=(1, 2, 3))
    return stations
