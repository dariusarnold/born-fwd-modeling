import math
import os
from typing import List, Sequence

import numpy as np
import quadpy
from tqdm import tqdm

from units import Hertz, RadiansPerSecond, Seconds
from velocity_model import AbstractVelocityModel


def set_number_numpy_threads(threads: int):
    """
    Set number of threads numpy will use for parallel processing.
    This has to be called before numpy is imported!
    """
    if threads <= 0:
        # this will leave the default value unchanged
        return
    threads = str(threads)
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["OMP_NUM_THREADS"] = threads
# TODO handle this
#set_number_numpy_threads(1)


def born_all_scatterers(xs: np.ndarray, xr: np.ndarray, velocity_model: AbstractVelocityModel,
                        omega: RadiansPerSecond, omega_central: RadiansPerSecond) -> complex:
    """
    Calculate born scattering for all scatterer positions
    This implements eq. (1) from
    3D seismic characterization of fractures in a dipping layer using the double-beam method
    Hao Hu and Yingcai Zheng
    :param xs: Source position
    :param xr: Receiver position
    :param velocity_model: Model containing geometry of scatterers, velocities and density
    :param omega: Angular frequency of this sample
    :param omega_central: Central frequency of ricker source wavelet
    :return: Value of single frequency bin of the scattered P wave
    """

    def complex_exp(exp_term: np.array) -> np.array:
        """
        Calculate complex exp by Eulers formula using cos(x) + i sin(x).
        Interestingly in numpy a complex exp takes more time to compute than the
        expanded version from eulers formula see:
        https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/758148
        This version is taken from above link and is even faster than a simple cos+isin
        since
        :param exp_term: The argument of the complex exp without imaginary i
        :return: Numpy array of complex values
        """
        df_exp = np.empty(exp_term.shape, dtype=np.csingle)
        trig_buf = np.cos(exp_term)
        df_exp.real[:] = trig_buf
        np.sin(exp_term, out=trig_buf)
        df_exp.imag[:] = trig_buf
        return df_exp

    def greens_function_vectorized(x: np.array, x_prime: np.array) -> np.array:
        """
        Vectorized version of the greens function using numpy.
        """
        # this is an optimized version of linalg.norm
        subtraction = x - x_prime
        lengths = np.sqrt(np.einsum("ijk,ijk->jk", subtraction, subtraction))
        # minus sign in exp term is required since it was exp(-ix) before, which
        # transforms to cos(-x) + i * sin(-x)
        # arguments of exp are reordered resulting in only one multiplication of lengths
        return 1./lengths * complex_exp(-omega * (1. / velocity_model.fracture_velocity) * lengths)

    def integral(x):
        """
        x is a np array containing all points to be evaluated.
        Its shape is (3, M, N) where M is the number of scatterer midpoints
        returned by create_scatterers and N is the number of evaluation points
        chosen by quadpy. The first axis is fixed (3). It represents the x, y, z
        values, eg. x[0] contains all x values of all points.
        """
        epsilon = scattering_potential(velocity_model.fracture_velocity, velocity_model.background_velocity)
        # extend the 3D vector from shape (3,) to (3, 1, 1) so numpy
        # broadcasting works as expected
        G0_left = greens_function_vectorized(xs[:, None, None], x)
        G0_right = greens_function_vectorized(x, xr[:, None, None])
        return G0_left * epsilon * G0_right

    scatterer_radii = np.full(len(velocity_model.scatterer_positions), velocity_model.scatterer_radius)
    integration_scheme = quadpy.ball.Stroud("S3 3-1")
    # sum over the result from all scatterer points
    res = np.sum(quadpy.ball.integrate(integral, velocity_model.scatterer_positions, scatterer_radii,
                                       integration_scheme))
    res *= ricker_frequency_domain(omega, omega_central) * omega**2
    res *= 1 / (4. * np.pi * velocity_model.density * velocity_model.background_velocity**2)
    return res

def born(source_pos: np.ndarray, receiver_pos: np.ndarray, velocity_model: AbstractVelocityModel,
         omega_central: RadiansPerSecond, omega_samples: Sequence[RadiansPerSecond],
         quiet: bool = False) -> np.ndarray:
    """
    Loop over frequency samples to create frequency spectrum of scattered P wave and backtransform
    into a time domain signal.
    """
    p_wave_spectrum: List[complex] = []
    for omega in tqdm(omega_samples, desc="Born modeling", total=len(omega_samples), unit="frequency samples", disable=quiet):
        u_scattering = born_all_scatterers(source_pos, receiver_pos, velocity_model, omega, omega_central)
        p_wave_spectrum.append(u_scattering)
    time_domain = np.real(np.fft.ifft(p_wave_spectrum))
    return time_domain


def angular(f: Hertz) -> RadiansPerSecond:
    return RadiansPerSecond(2. * math.pi * f)


def frequency_samples(timeseries_length: Seconds, sample_period: Seconds) -> np.ndarray:
    """
    Calculate frequency samples required to reach the given length and sample period after
    the inverse Fourier transform.
    """
    num_of_samples = int(timeseries_length / sample_period)
    delta_omega = 2*math.pi / timeseries_length
    omega_max = num_of_samples * delta_omega
    f_samples = np.linspace(0, omega_max, num_of_samples)
    return f_samples


def time_samples(timeseries_length: Seconds, sample_period: Seconds) -> np.ndarray:
    """Calculate all time points between 0 and time series_length when the time series is sampled
    with the given period."""
    num_of_samples = int(timeseries_length / sample_period)
    return np.linspace(0, timeseries_length, num_of_samples)


def save_seismogram(seismogram: np.ndarray, time_steps: np.ndarray, header: str, filename: str):
    # transpose stacked arrays to save them as columns instead of rows
    np.savetxt(filename, np.vstack((time_steps, seismogram)).T, header=header)


def create_header(source_pos: np.ndarray, receiver_pos: np.ndarray) -> str:
    """
    Create header string containing information about the seismogram from the arguments used to
    create it. This information will be saved as a header in the seismogram file.
    """
    h = f"source: {source_pos}\nreceiver: {receiver_pos}"
    return h


def ricker_frequency_domain(omega: float, omega_central: float) -> float:
    """
    Taken from Frequencies of the Ricker wavelet by Yanghua Wang (eq. 7)
    :param omega: Frequency at which to evaluate the spectrum
    :param omega_central: Central or dominant frequency
    :return: Value of frequency spectrum of a Ricker wavelet
    """
    return 2 * omega**2 / (math.sqrt(math.pi) * omega_central**3) * math.exp(- omega**2 / omega_central**2)


def scattering_potential(v: float, v0: float) -> float:
    """
    Scattering potential with respect to the homogeneous medium
    Eq. 3 from Hu2018a
    :param v: velocity at position x prime
    :param v0: homogeneous background velocity
    """
    return 1 / v0**2 - 1 / v**2
