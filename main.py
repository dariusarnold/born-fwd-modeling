import argparse
import importlib
import itertools
from pathlib import Path

import numpy as np

from bornfwd.velocity_model import VelocityModel, AbstractVelocityModel
from bornfwd.functions import angular, born, time_samples, frequency_samples
from bornfwd.io import save_seismogram, create_header, read_stations, read_sources
from bornfwd.plotting import plot_time_series
from bornfwd.units import Hertz


def _setup_parser() -> argparse.ArgumentParser:
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

    # add general options valid for both commands
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
    p.add_argument("-q", "--quiet", action="store_true", help=("Flag to disable"
                   " performance output (iterations per second)."))

    # add subparsers
    subparsers_descr = ("These two subcommands differ in the way that source "
                        "and receiver positions are specified. For a oneshot, "
                        "a single source and receiver are specified on the "
                        "command line (or in an options file, see the README). "
                        "fullmodel requires all source and receiver positions "
                        "in two seperate files. Their format is again described"
                        " in the README.")
    subparsers = p.add_subparsers(description=subparsers_descr)

    # add arguments for oneshot parser
    oneshot_p = subparsers.add_parser("oneshot", help="Create data for single "
                                      "source-receiver combination.")
    oneshot_p.set_defaults(func=oneshot)
    # Even though making optional arguments required is against command line
    # conventions, this is the only way to use metavars to specify order of
    # coordinates
    # see https://bugs.python.org/issue14074
    # TODO rewrite to required named arguments group
    oneshot_p.add_argument("-s", "--source_pos", nargs=3, type=float, required=True,
                   action=ConvertToNumpyArray, metavar=("XS", "YS", "ZS"),
                   help="coordinates of source (shot position) in m")
    oneshot_p.add_argument("-r", "--receiver_pos", nargs=3, type=float, required=True,
                   action=ConvertToNumpyArray, metavar=("XR", "YR", "ZR"),
                   help="coordinates of receiver (geophone position) in m")
    oneshot_p.add_argument("filename", type=str, metavar="output_filename", nargs="?",
                   help="Filename in which results will be saved. If no name is "
                   "specified, the output won't be saved to a file.")
    oneshot_p.add_argument("-p", "--plot", action="store_true", help=("If flag "
                           "is specified, plot the trace and show plot."))

    # add arguments for fullmodel parser
    fullmodel_p = subparsers.add_parser("fullmodel", help="Create data multiple"
                                        " source-receiver combinations.")
    fullmodel_p.add_argument("-s", "--sourcefile", required=True,
                             help="Specify file from which source coordinates "
                                  "are read.")
    fullmodel_p.add_argument("-r", "--receiverfile", required=True,
                             help="Specify file from which receiver coordinates"
                                  "are read.")
    fullmodel_p.set_defaults(func=fullmodel)
    return p


def main() -> None:
    """
    Parse arguments and call the function depending on if oneshot or fullmodel
    subcommand was used, passing the parsed arguments.
    """
    parser = _setup_parser()
    args = parser.parse_args()
    args.func(args)


def oneshot(args) -> None:
    """
    Create a seismogram by born modeling and save it to a file or plot it.
    """
    omega_samples = frequency_samples(args.timeseries_length, args.sample_period)
    # transpose positions from (N, 3) to (3, N) for broadcasting
    seismogram = born(np.reshape(args.source_pos, (3, 1)), np.reshape(args.receiver_pos, (3, 1)), args.model, args.omega_central,
                      omega_samples, args.quiet)
    t_samples = time_samples(args.timeseries_length, args.sample_period)
    header = create_header(args.source_pos, args.receiver_pos)
    if args.plot is True:
        plot_time_series(np.squeeze(seismogram), t_samples)
    if args.filename is not None:
        save_seismogram(seismogram, t_samples, header, args.filename)


def fullmodel(args) -> None:
    omega_samples = frequency_samples(args.timeseries_length, args.sample_period)
    t_samples = time_samples(args.timeseries_length, args.sample_period)
    receivers = read_stations(Path(args.receiverfile))
    sources = read_sources(Path(args.sourcefile))
    output_folder = Path("output")
    for source_index, source_pos in enumerate(sources, start=1):
        sourcepath = output_folder / f"source_{source_index:03d}"
        try:
            sourcepath.mkdir()
        except FileExistsError:
            pass
        for receiver_index, receiver_pos in enumerate(receivers):
            seismogram = born(source_pos, receiver_pos, args.model,
                              args.omega_central, omega_samples, args.quiet)
            header = create_header(source_pos, receiver_pos)
            seismopath = sourcepath / f"receiver_{receiver_index:03d}.txt"
            save_seismogram(seismogram, t_samples, header, seismopath)



if __name__ == '__main__':
    main()
