import argparse
import importlib

import numpy as np

from bornfwd.velocity_model import VelocityModel, AbstractVelocityModel
from bornfwd.functions import angular, born, time_samples, create_header, \
    save_seismogram, frequency_samples
from bornfwd.plotting import plot_time_series
from bornfwd.units import Hertz


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

    program_description = ("Use born modelling to generate seismic recordings "
                           "from fracture scattered waves.")
    p = argparse.ArgumentParser(description=program_description,
                                fromfile_prefix_chars="@")
    # Even though making optional arguments required is against command line
    # conventions, this is the only way to use metavars to specify order of
    # coordinates
    # see https://bugs.python.org/issue14074
    p.add_argument("-s", "--source_pos", nargs=3, type=float, required=True,
                   action=ConvertToNumpyArray, metavar=("XS", "YS", "ZS"),
                   help="coordinates of source (shot position) in m")
    p.add_argument("-r", "--receiver_pos", nargs=3, type=float, required=True,
                   action=ConvertToNumpyArray, metavar=("XR", "YR", "ZR"),
                   help="coordinates of receiver (geophone position) in m")
    p.add_argument("filename", type=str, metavar="output_filename",
                   help="Filename in which results will be saved")
    p.add_argument("-w", "--omega_central", type=angular,
                   metavar="HZ", default=angular(Hertz(30.)),
                   help="Central frequency of Ricker source wavelet in Hz")
    p.add_argument("-t", "--time", nargs=2, type=float,
                   metavar=("LENGTH", "SAMPLEPERIOD"),
                   help="Length of output time series (s) and sample rate (s).",
                   action=AddNargsAsAttributesAction,
                   dest="timeseries_length sample_period")
    p.add_argument("-c", "--cores", type=int, help=("Number of cores for "
                   "parallelization. If not specified, numpys default value "
                   "will be kept."))
    p.add_argument("-m", "--model", type=velocity_model, default="marcellus.py",
                   help=("Specify file from which the velocity model is created."
                         "The file should contain a create_velocity_model "
                         "function that returns a VelocityModel object, which "
                         "contains model data such as density and velocity as "
                         "well as scatterer positions in a (N, 3) shape array."
                         "N are the number of scatterer points and the second "
                         "axis contains the (x, y, z) sequence of coordinates "
                         "for every point. An abstract base class for your "
                         "own models to derive from is provided in "
                         "VelocityModel.py. Deriving from this class ensures "
                         "that the model has defined all required attributes."))
    p.add_argument("-q", "--quiet", action="store_true", help=("Flag to disable "
                   "performance output (iterations per second)."))
    return p


def main() -> None:
    """
    Read command line arguments, create a seismogram by born modeling and save
    it to a file.
    """
    parser = setup_parser()
    args = parser.parse_args()

    omega_samples = frequency_samples(args.timeseries_length, args.sample_period)
    seismogram = born(args.source_pos, args.receiver_pos, args.model, args.omega_central,
                      omega_samples, args.quiet)
    t_samples = time_samples(args.timeseries_length, args.sample_period)
    header = create_header(args.source_pos, args.receiver_pos)
    plot_time_series(seismogram, t_samples)
    save_seismogram(seismogram, t_samples, header, args.filename)


if __name__ == '__main__':
    main()
