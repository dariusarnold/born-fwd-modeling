import time
from pathlib import Path

import numpy as np

from bornfwd.marcellus import create_velocity_model
from bornfwd.functions import born, time_samples, angular, frequency_samples
from bornfwd.io import save_seismogram, create_header, load_seismograms
from bornfwd.plotting import plot_seismogram_gather
from bornfwd.units import Seconds, RadiansPerSecond, KgPerCubicMeter, MetersPerSecond, Meter

"""
This script generates data to recreate fig. 5b) from the paper 3D seismic 
characterization of fractures in a dipping layer using the double-beam method 
by Hao Hu and Yingcai Zheng. The data is generated using the same born modeling 
method as mentioned in the paper. 
"""


def generate_seismograms():
    t_samples = time_samples(length, sample_period)
    vm = create_velocity_model()
    with open("timing.txt", "w", buffering=1) as f:
        for index, receiver_pos in enumerate(receivers):
            before = time.time()
            seismogram = born(source_pos, receiver_pos, vm, omega_central, f_samples)
            header = create_header(receiver_pos, source_pos)
            fname = output_filename.format(id=index)
            save_seismogram(seismogram, t_samples, header, fname)
            after = time.time()
            runtime = after - before
            f.write(f"Iteration {index:03d}: {runtime} s\n")


if __name__ == '__main__':
    #
    # parameters set so that same data as in paper is generated
    #
    start_receiver = np.array((5272., 3090., 0.))
    end_receiver = np.array((3090., 5430., 0.))
    num_of_receivers = 100
    source_pos = np.array((11200., 5600., 10.))
    omega_central: RadiansPerSecond = angular(30.)
    length: Seconds = 4
    sample_period: Seconds = 0.004

    # format id leftpadded with zeros
    output_filename = "output/receiver_{id:03d}.txt"
    f_samples = frequency_samples(length, sample_period)

    # generate line of receivers with similar geometry as the one in the paper
    receivers_x = np.linspace(start_receiver[0], end_receiver[0], num_of_receivers)
    receivers_y = np.linspace(start_receiver[1], end_receiver[1], num_of_receivers)
    receivers = np.array([(x, y, 0.) for x, y in zip(receivers_x, receivers_y)])

    seismograms = generate_seismograms()
    seismos, *_ = load_seismograms(Path(output_filename).parent, "receiver_*.txt")
    plot_seismogram_gather(seismos)
