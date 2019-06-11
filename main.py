import argparse
import importlib

import numpy as np

from VelocityModel import VelocityModel, AbstractVelocityModel
from functions import angular, born, time_samples, create_header, save_seismogram, frequency_samples
from units import Hertz


def setup_parser() -> argparse.ArgumentParser:
    def velocity_model(fname: str) -> VelocityModel:
        """Create VelocityModel from given VelocityModel file name.
        This function imports the file as a module and looks for a
        create_velocity_model function. If found it is called and
        checked if the return value is a subclass of the AbstractVelocityModel.
        The created velocity model is returned."""
        try:
            # try to import module from file
            module = importlib.import_module(fname.rstrip(".py"))
        except ModuleNotFoundError:
            raise argparse.ArgumentTypeError(f"The file {fname} does not exist")
        try:
            # try to create velocity model
            vm = module.create_velocity_model()
        except TypeError as e:
            msg = ("Cant instantiate VelocityModel since the provided class "
                   "doesn't implement all abstract functions. "
                   f"Error encountered during instantiation: {e}")
            raise argparse.ArgumentTypeError(msg)
        except AttributeError:
            msg = (f"The provided module {fname} does not contain a "
                   "create_velocity_model function.")
            raise argparse.ArgumentTypeError(msg)
        if not issubclass(type(vm), AbstractVelocityModel):
            msg = ("The VelocityModel is not registered as a subclass of the "
                   "AbstractVelocityModel. Derive the velocity"
                   " model class from AbstractVelocityModel")
            raise argparse.ArgumentTypeError(msg)
        return vm

    class ConvertToNumpyArray(argparse.Action):
        """Convert tuple of 3 values to a numpy array during argument parsing"""

        def __call__(self, parser, namespace, values, option_string=None):
            vector = np.array(values)
            setattr(namespace, self.dest, vector)

    class AddNargsAsAttributesAction(argparse.Action):
        """Instead of gathering nargs into a list, add them as attributes to the
        namespace using dest as the attribute names. dest should be a string of
        names split by whitespace."""

        def __call__(self, parser, namespace, values, option_string=None):
            names = self.dest.split()
            for name, value in zip(names, values):
                setattr(namespace, name, value)

    program_description = ("Use born modelling to generate seismic recordings from fracture "
                           "scattered waves.")
    parser = argparse.ArgumentParser(description=program_description,
                                     fromfile_prefix_chars="@")
    # Even though making optional arguments required is against command line
    # conventions, this is the only way to use metavars to specify order of
    # coordinates
    # see https://bugs.python.org/issue14074
    parser.add_argument("-s", "--source_pos", nargs=3, type=float, action=ConvertToNumpyArray,
                        metavar=("XS", "YS", "ZS"), required=True,
                        help="coordinates of source (shot position) in m")
    parser.add_argument("-r", "--receiver_pos", nargs=3, type=float, action=ConvertToNumpyArray,
                        metavar=("XR", "YR", "ZR"), required=True,
                        help="coordinates of receiver (geophone position) in m")
    parser.add_argument("filename", type=str, metavar="output_filename",
                        help="Filename in which results will be saved")
    parser.add_argument("-w", "--omega_central", type=angular,
                        metavar="HZ", default=angular(Hertz(30.)),
                        help="Central frequency of Ricker source wavelet in Hz")
    parser.add_argument("-t", "--time", nargs=2, type=float, metavar=("LENGTH", "SAMPLEPERIOD"),
                        help="Length of output time series in s and sample rate in s.",
                        action=AddNargsAsAttributesAction, dest="timeseries_length sample_period")
    parser.add_argument("-c", "--cores", type=int, help=("Number of cores for parallelization. "
                        "If not specified, numpys default value will be kept."))
    parser.add_argument("-m", "--model", type=velocity_model, default="VelocityModel.py",
                        help=("Specify file from which the velocity model is created. The file "
                              "should contain a create_velocity_model function that returns a "
                              "VelocityModel object, which contains model data such as density and "
                              "velocity as well as scatterer positions in a (N, 3) shape array "
                              "where N are the number of scatterer points and the second axis "
                              "contains the (x, y, z) sequence of coordinates for every point. "
                              "An abstract base class that has to be overridden is provided in "
                              "VelocityModel.py"))
    return parser


def main() -> None:
    """
    Read command line arguments, create a seismogram by born modeling and save
    it to a file.
    """
    parser = setup_parser()
    args = parser.parse_args()

    omega_samples = frequency_samples(args.timeseries_length, args.sample_period)
    seismogram = born(args.source_pos, args.receiver_pos, args.model, args.omega_central,
                      omega_samples)
    t_samples = time_samples(args.timeseries_length, args.sample_period)
    header = create_header(args.source_pos, args.receiver_pos)
    save_seismogram(seismogram, t_samples, header, args.filename)


if __name__ == '__main__':
    main()
