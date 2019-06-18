from argparse import Namespace
from pathlib import Path

from bornfwd.functions import angular
from bornfwd.io import load_seismograms
from bornfwd.plotting import plot_seismogram_gather
from bornfwd.units import Seconds, Hertz
from main import fullmodel
from marcellus import create_velocity_model

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
    parameters = {
        "timeseries_length": Seconds(4.),
        "sample_period": Seconds(0.004),
        "receiverfile": "sample_receiverfile",
        "sourcefile": "reproduce_fig_sourcefile",
        "model": create_velocity_model(),
        "omega_central": angular(Hertz(30)),
        "chunksize": 12
    }

    args = Namespace(**parameters)
    fullmodel(args)


if __name__ == '__main__':
    generate_seismograms()
    seismos, *_ = load_seismograms(Path("output_fig5b/source_001"), "receiver_*.txt")
    plot_seismogram_gather(seismos)
