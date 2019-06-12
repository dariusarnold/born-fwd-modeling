from pathlib import Path
from typing import List, Tuple

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

def save_seismogram(seismogram: np.ndarray, time_steps: np.ndarray, header: str,
                    filename: str) -> None:
    # transpose stacked arrays to save them as columns instead of rows
    np.savetxt(filename, np.vstack((time_steps, seismogram)).T, header=header)


def create_header(source_pos: np.ndarray, receiver_pos: np.ndarray) -> str:
    """
    Create header string containing information about the seismogram from the
    arguments used to create it. This information will be saved as a header in
    the seismogram file.
    """
    h = f"source: {source_pos}\nreceiver: {receiver_pos}"
    return h


def read_header_from_file(filepath: Path) -> List[str]:
    """Read first two lines from a file saved by save_seismogram function and
    return them as a list of strings, stripped from newline characters."""
    with open(filepath, "r") as f:
        # read first 2 lines
        header = [next(f).rstrip("\n") for _ in range(2)]
    return header


def parse_header(header: List[str]) -> List[np.ndarray]:
    """Parse header from seismogram file to Vectors of source and receiver
    position"""
    vectors = []
    for line in header:
        data = line.split(":")[-1]
        # remove outer square brackets
        data = data[2:-1]
        vec = np.fromstring(data, dtype=float, sep=" ")
        vectors.append(vec)
    return vectors


def load_seismogram(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load seismogram from file
    :return: two numpy arrays: First is time data, second is seismic data
    """
    times, seismic_data = np.loadtxt(filepath, unpack=True)
    return times, seismic_data


def load_seismograms(seismograms_path: Path, seismogram_filename_template: str)\
        -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Load seismograms from the given path.
    This loads all seismograms from the path and returns them as well as
    additional information.
    :param seismograms_path: Path to seismogram files
    :return Numpy array consisting of all loaded seismograms, numpy array
    containing the common timesteps for all seismograms, a list of receiver
    positions for these seismograms, and the source position.
    """
    seismograms = []
    receiver_positions = []
    seismogram_filenames = seismograms_path.glob(seismogram_filename_template)
    for fname in sorted(seismogram_filenames):
        header = read_header_from_file(fname)
        source_pos, receiver_pos = parse_header(header)
        receiver_positions.append(receiver_pos)
        times, seismic_data = load_seismogram(fname)
        seismograms.append(seismic_data)
    return np.array(seismograms), times, receiver_positions, source_pos