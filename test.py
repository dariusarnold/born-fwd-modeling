import time
from pathlib import Path

import numpy as np

from marcellus import create_velocity_model
from bornfwd.functions import born, time_samples, angular, frequency_samples
from bornfwd.io import save_seismogram, create_header, load_seismograms
from scripts.create_sample_receiverfile import create_receivers
from bornfwd.plotting import plot_seismogram_gather
from bornfwd.units import Seconds, RadiansPerSecond, KgPerCubicMeter, MetersPerSecond, Meter

"""
This script generates data to recreate fig. 5b) from the paper 3D seismic 
characterization of fractures in a dipping layer using the double-beam method 
by Hao Hu and Yingcai Zheng. The data is generated using the same born modeling 
method as mentioned in the paper. 
"""


def generate_seismograms():
    #
    # parameters set so that same data as in paper is generated
    #
    source_positions = np.random.rand(4, 3) * 11200
    receiver_positions = create_receivers()[:3]
    omega_central: RadiansPerSecond = angular(30.)
    length: Seconds = 4
    sample_period: Seconds = 0.004

    # format id leftpadded with zeros
    output_folder = "output/source_{idsource:03d}"
    output_filename = "receiver_{idreceiver:03d}.txt"
    f_samples = frequency_samples(length, sample_period)
    t_samples = time_samples(length, sample_period)
    vm = create_velocity_model()
    # create all source directories
    for i in range(len(source_positions)):
        path = Path(output_folder.format(idsource=i))
        path.mkdir(parents=True, exist_ok=True)
    with open("timing.txt", "w", buffering=1) as f:
        # use symmetry of equation: a seismogram calculated from source to
        # receiver is the same as the one calculated from receiver to source
        # this can save almost 50% of calculations if an equal number of
        # sources and receivers is used
        receiver_indices, source_indices = np.triu_indices(len(receiver_positions), 0, len(source_positions))
        # TODO tqdm um diesen Iterator
        for index_source, index_receiver in zip(source_indices, receiver_indices):
            before = time.time()
            seismogram = born(source_positions[index_source], receiver_positions[index_receiver], vm, omega_central, f_samples)
            after = time.time()
            runtime = after - before
            print(runtime)
            f.write(f"Runtime: {runtime} s\n")
            # save seismogram
            header = create_header(source_positions[index_source], receiver_positions[index_receiver])
            fpath = Path(output_folder.format(idsource=index_source))
            fname = Path(output_filename.format(idreceiver=index_receiver))
            save_seismogram(seismogram, t_samples, header, fpath/fname)
            if index_source != index_receiver:
                # this means we are not on the diagonal of the source/receiver
                # matrix and have to save the other source/receiver permutation
                header = create_header(source_positions[index_source], receiver_positions[index_source])
                fname = Path(output_filename.format(idreceiver=index_source))
                save_seismogram(seismogram, t_samples, header, fpath/fname)


if __name__ == '__main__':
    generate_seismograms()
