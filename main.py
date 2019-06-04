import argparse
import importlib
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from VelocityModel import VelocityModel, AbstractVelocityModel, Vector3D
from functions import born_modeling
from units import Hertz, RadiansPerSecond


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


def angular(f: Hertz) -> RadiansPerSecond:
    return RadiansPerSecond(2. * math.pi * f)


def main():

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

    program_description = ("Use born modelling to generate seismic recordings "
                           "from fracture scattered waves.")
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
    parser.add_argument("output_filename", type=argparse.FileType("w"),
                        help="Filename in which results will be saved")
    parser.add_argument("-w", "--omega_central", type=angular,
                        metavar="HZ", default=angular(Hertz(30.)),
                        help="Central frequency of Ricker source wavelet in Hz")
    parser.add_argument("-n", "--num_of_frequency_steps", type=int, default=16,
                        help="# of evenly spaced frequency samples to take between [fmin, fmax]")
    parser.add_argument("--fmin", type=float, default=1.,
                        help="Minimal frequency for which u_scattering is calculated")
    parser.add_argument("--fmax", type=float, default=100.,
                        help="Maximum frequency for which u_scattering is calculated")
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

    args = parser.parse_args()
    print(args)

    frequency_min: RadiansPerSecond = angular(args.fmin)
    frequency_max: RadiansPerSecond = angular(args.fmax)
    frequency_samples: Sequence[RadiansPerSecond] = np.linspace(frequency_min, frequency_max,
                                                                args.num_of_frequency_steps)
    born(args.source_pos, args.receiver_pos, args.velocity_model, args.omega_central,
         frequency_samples, args.processing, args.cores)


def born(source_pos: Vector3D, receiver_pos: Vector3D,
         velocity_model: AbstractVelocityModel, omega_central: RadiansPerSecond,
         frequency_samples: Sequence[RadiansPerSecond], processing: str, num_cores: int):
    p_wave_spectrum = []
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
    print(p_wave_spectrum)
    freq_domain = np.array([amplitude for freq, amplitude in p_wave_spectrum])
    time_domain = np.fft.ifft(freq_domain)
    # throws ComplexWarning
    plt.plot(time_domain)
    plt.show()


if __name__ == '__main__':
    main()
