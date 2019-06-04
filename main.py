import argparse
import importlib
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from VelocityModel import VelocityModel, AbstractVelocityModel, Vector3D
from functions import born_modeling
from units import Hertz, RadiansPerSecond, Seconds


def plot_fractures(velocity_model: VelocityModel):
    x = list(point.x for point in velocity_model.scatterer_positions)
    y = list(point.y for point in velocity_model.scatterer_positions)
    plt.scatter(x, y, marker=",", c="g", s=1)
    plt.title("Scatterer points in model")
    plt.xlabel("x axis (West-East, m)")
    plt.ylabel("y axis (South-North, m)")
    plt.tight_layout()
    plt.axis("equal")
    plt.xlim((0, 11200))
    plt.ylim((0, 11200))
    plt.show()


def plot_time_series(time_series: np.ndarray):
    plt.plot(np.real(time_series))
    plt.title("Time series from Born scattering")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()


def angular(f: Hertz) -> RadiansPerSecond:
    return RadiansPerSecond(2. * math.pi * f)


def frequency_samples(timeseries_length: Seconds, sample_period: Seconds) -> np.array:
    """Calculate frequency samples required to reach the given length and sample period after
    the inverse Fourier transform."""
    num_of_samples = int(timeseries_length / sample_period)
    delta_omega = 2*math.pi / timeseries_length
    omega_max = num_of_samples * delta_omega
    f_samples = np.linspace(0, omega_max, num_of_samples)
    return f_samples


def time_samples(timeseries_length: Seconds, sample_period: Seconds) -> np.array:
    """Calculate all time points between 0 and timeseries_length when the timeseries is sampled
    with the given period."""
    num_of_samples = int(timeseries_length / sample_period)
    return np.linspace(0, timeseries_length, num_of_samples)


def save_seismogram(seismogram: np.ndarray, time_steps: np.ndarray, header: str, filename: str):
    # transpose stacked arrays to save them as columns instead of rows
    np.savetxt(filename, np.vstack((time_steps, seismogram)).T, header=header)


def create_header(args: argparse.Namespace) -> str:
    """Create header string containing information about the seismogram from the arguments used to
    create it. This information will be saved as a header in the seismogram file."""
    h = f"source: {args.source_pos}\nreceiver: {args.receiver_pos}"
    return h


def setup_parser():
    def velocity_model(fname: str) -> VelocityModel:
        """Create VelocityModel from given VelocityModel file name.
        This function imports the file as a module and looks for a
        create_velocity_model function. If found it is called and
        checked if the return value is a subclass of the AbstractVelocityModel.
        The created velocity model is returned."""
        try:
            # try to import module from file
            vm_module = importlib.import_module(fname.rstrip(".py"))
        except ModuleNotFoundError:
            raise argparse.ArgumentTypeError(f"The file {fname} does not exist")
        try:
            # try to create velocity model
            vm = vm_module.create_velocity_model()
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

    class ConvertToVector3DAction(argparse.Action):
        """Convert tuple of 3 values to a Vector3D during argument parsing"""

        def __call__(self, parser, namespace, values, option_string=None):
            vector = Vector3D(*values)
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
    parser.add_argument("-s", "--source_pos", nargs=3, type=float, action=ConvertToVector3DAction,
                        metavar=("XS", "YS", "ZS"),required=True,
                        help="coordinates of source (shot position) in m")
    parser.add_argument("-r", "--receiver_pos", nargs=3, type=float, action=ConvertToVector3DAction,
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
    parser.add_argument("--serial", action="store_const", dest="processing", const="serial",
                        default="parallel", help="This flag activates serial processing of the "
                        "frequency samples instead of parallel (default)")
    parser.add_argument("-c", "--cores", type=int, help=("Number of cores for multiprocessing. "
                        "If not specified, all available cores will be used"))
    parser.add_argument("-v", "--velocity_model", type=velocity_model, default="VelocityModel.py",
                        help=("Specify file from which the VelocityModel is created. The file "
                              "should contain a create_velocity_model function that returns a "
                              "VelocityModel object, which has an eval_at method that takes a "
                              "(x, y, z) sequence of coordinates and returns the velocity at that "
                              "point. An abstract base class that has to be overridden is provided "
                              "in VelocityModel.py"))
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    f_samples = frequency_samples(args.timeseries_length, args.sample_period)
    seismogram = born(args.source_pos, args.receiver_pos, args.velocity_model, args.omega_central,
                      f_samples, args.processing, args.cores)
    t_samples = time_samples(args.timeseries_length, args.sample_period)
    header = create_header(args)
    save_seismogram(seismogram, t_samples, header, args.filename)


def born(source_pos: Vector3D, receiver_pos: Vector3D, velocity_model: AbstractVelocityModel,
         omega_central: RadiansPerSecond, frequency_samples: Sequence[RadiansPerSecond],
         processing: str, num_cores: int) -> np.ndarray:
    p_wave_spectrum: List[Tuple[RadiansPerSecond, complex]] = []
    futures = []
    fut_freq_mapping = {}
    density = velocity_model.density
    if processing == "serial":
        for frequency in tqdm(frequency_samples, desc="Born modeling", total=len(frequency_samples),
                              unit="frequency samples"):
            res = born_modeling(source_pos, receiver_pos, frequency, omega_central, density=density,
                                velocity_model=velocity_model)
            p_wave_spectrum.append((frequency, res))
    elif processing == "parallel":
        with ProcessPoolExecutor(num_cores) as process_pool:
            for frequency in frequency_samples:
                future = process_pool.submit(born_modeling, source_pos, receiver_pos, frequency,
                                             omega_central, density=density, velocity_model=velocity_model)
                futures.append(future)
                fut_freq_mapping[future] = frequency
            for future in tqdm(as_completed(futures), desc="Born modeling",
                               total=len(frequency_samples), unit="frequency samples"):
                res = future.result()
                frequency = fut_freq_mapping[future]
                p_wave_spectrum.append((frequency, res))

    p_wave_spectrum = sorted(p_wave_spectrum, key=lambda x: x[0])
    freq_domain = np.array([amplitude for freq, amplitude in p_wave_spectrum])
    time_domain = np.real(np.fft.ifft(freq_domain))
    return time_domain


if __name__ == '__main__':
    main()
