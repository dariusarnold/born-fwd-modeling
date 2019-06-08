import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from VelocityModel import Vector3D, create_velocity_model
from main import angular, frequency_samples, born, time_samples, create_header, \
    save_seismogram
from units import Seconds, RadiansPerSecond

"""
This script generates data to recreate fig. 5b) from the paper 3D seismic 
characterization of fractures in a dipping layer using the double-beam method 
by Hao Hu and Yingcai Zheng. The data is generated using the same born modeling 
method as mentioned in the paper. 
"""

#
# parameters
#
start_receiver = Vector3D(5272., 3090., 0.)
end_receiver = Vector3D(3090., 5430., 0.)
num_of_receivers = 100

#
# constant parameters, dont change these
#
source_pos = Vector3D(11200., 5600., 10.)
vm = create_velocity_model()
omega_central: RadiansPerSecond = angular(30.)
length: Seconds = 4
sample_period: Seconds = 0.004
# format id leftpadded with zeros
output_filename = "receiver_{id:03d}.txt"
f_samples = frequency_samples(length, sample_period)


def generate_seismograms_for_receivers():
    receivers_x = np.linspace(start_receiver.x, end_receiver.x, num_of_receivers)
    receivers_y = np.linspace(start_receiver.y, end_receiver.y, num_of_receivers)
    with open("timing.txt", "w", buffering=1) as f:
        for index, (x, y) in enumerate(zip(receivers_x, receivers_y)):
            before = time.time()
            current_receiver = Vector3D(x, y, 0.)
            seismogram = born(source_pos, current_receiver, vm, omega_central,
                              f_samples)
            t_samples = time_samples(length, sample_period)
            header = create_header(current_receiver, source_pos)
            fname = output_filename.format(id=index)
            save_seismogram(seismogram, t_samples, header, fname)
            after = time.time()
            runtime = after - before
            f.write(f"Iteration {index:03d}: {runtime} s\n")


def read_header_from_file(filepath: Path) -> List[str]:
    """Read first two lines from a file saved by save_seismogram function and
    return them as a list of strings, stripped from newline characters."""
    with open(filepath, "r") as f:
        # read first 2 lines
        header = [next(f).rstrip("\n") for _ in range(2)]
    return header


def parse_header(header: List[str]) -> List[Vector3D]:
    """Parse header from seismogram file to Vectors of source and receiver
    position"""
    vectors = []
    for line in header:
        data = line.split(":")[-1]
        vec = eval(data)
        vectors.append(vec)
    return vectors


def load_seismogram(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load seismogram from file
    :return: two numpy arrays: First is time data, second is seismic data
    """
    times, seismic_data = np.loadtxt(filepath, unpack=True)
    return times, seismic_data


def load_seismograms(seismograms_path: Path) -> Tuple[np.ndarray, np.ndarray,
                                                      List[Vector3D], Vector3D]:
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
    for index in range(num_of_receivers):
        fname = output_filename.format(id=index)
        fpath = seismograms_path / fname
        header = read_header_from_file(fpath)
        source_pos, receiver_pos = parse_header(header)
        receiver_positions.append(receiver_pos)
        times, seismic_data = load_seismogram(fpath)
        seismograms.append(seismic_data)
    return np.array(seismograms), times, receiver_positions, source_pos


if __name__ == '__main__':
    #generate_seismograms_for_receivers()
    seismos, timesteps, receiver_positions, source_pos = load_seismograms(Path("output"))
