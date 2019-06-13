import math
import os
from typing import List, Sequence

import numpy as np
from quadpy.ball import integrate, Stroud
from tqdm import tqdm

from bornfwd.units import Hertz, RadiansPerSecond, Seconds
from bornfwd.velocity_model import AbstractVelocityModel


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


def born_all_scatterers(xs: np.ndarray, xr: np.ndarray,
                        velocity_model: AbstractVelocityModel,
                        omega: RadiansPerSecond,
                        omega_central: RadiansPerSecond) -> np.ndarray:
    """
    Calculate born scattering for all scatterer positions
    This implements eq. (1) from
    3D seismic characterization of fractures in a dipping layer using the
    double-beam method
    Hao Hu and Yingcai Zheng
    :param xs: Source position
    :param xr: Receiver position
    :param velocity_model: Model containing geometry of scatterers, velocities
    and density
    :param omega: Angular frequency of this sample
    :param omega_central: Central frequency of ricker source wavelet
    :return: Value of single frequency bin of the scattered P wave
    """

    def complex_exp(exp_term: np.array) -> np.array:
        """
        Calculate complex exp by Eulers formula using cos(x) + i sin(x).
        Interestingly in numpy a complex exp takes more time to compute than the
        expanded version from eulers formula, see:
        https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/758148
        This version is taken from above link and is even faster than a simple
        cos+isin since you avoid intermediate temporary arrays and needless
        copying.
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
        lengths = np.sqrt(np.einsum("ijkl,ijkl->jkl", subtraction, subtraction))
        # minus sign in exp term is required since it was exp(-ix) before, which
        # transforms to cos(-x) + i * sin(-x)
        # arguments of exp are reordered compared to the formula fromt the paper,
        # resulting in only one multiplication of lengths by a scalar
        return 1./lengths * complex_exp(-omega * (1. / bg_vel) * lengths)

    def integral(x):
        """
        Inner function of the integral equation (1) from Hu2018.
        Constant factors are moved out of the integral and are multiplied later.
        x is a np array containing all points to be evaluated.
        Its shape is (3, M, N) where M is the number of scatterer midpoints
        returned by create_scatterers and N is the number of evaluation points
        chosen by quadpy. The first axis is fixed (3). It represents the x, y, z
        values, eg. x[0] contains all x values of all points.
        """
        # add empty last axis to x. This axis will be broadcasted to the number
        # of source points
        # extend the 3D vector (source and receiver positions) from shape (3, A)
        # or (3, B) to (3, 1, 1, A) or (3, 1, 1, B) so numpy broadcasting works
        # as expected
        G0_left = greens_function_vectorized(xs[:, None, None, :], x[..., None])
        G0_right = greens_function_vectorized(x[..., None], xr[:, None, None, :])
        res = G0_left[..., None] * np.expand_dims(G0_right, -2)
        res = np.moveaxis(res, -1, 0)
        res = np.moveaxis(res, -1, 0)
        return res

    frac_vel = velocity_model.fracture_velocity
    bg_vel = velocity_model.background_velocity
    epsilon = scattering_potential(frac_vel, bg_vel)
    scatterer_radii = np.full(len(velocity_model.scatterer_positions),
                              velocity_model.scatterer_radius)
    integration_scheme = Stroud("S3 3-1")
    # sum over the result from all scatterer points
    res = integrate(integral, velocity_model.scatterer_positions,
                    scatterer_radii, integration_scheme)
    res = np.sum(res, axis=-1)
    res *= ricker_frequency_domain(omega, omega_central) * omega**2 * epsilon
    res *= 1 / (4. * np.pi * velocity_model.density * bg_vel**2)
    return res


def born(source_pos: np.ndarray, receiver_pos: np.ndarray,
         velocity_model: AbstractVelocityModel,
         omega_central: RadiansPerSecond,
         omega_samples: Sequence[RadiansPerSecond],
         quiet: bool = False) -> np.ndarray:
    """
    Loop over frequency samples to create frequency spectrum of scattered P wave
    and backtransform into a time domain signal.
    :param source_pos: (A, 3) array of A (x y z) coordinates of A source
    positions
    :param receiver_pos: (B, 3) array of B (x y z) coordinates of B receiver
    positions
    :param velocity_model: Velocity model instance containg scatterer positions
    and other data
    :param omega_central: Central frequency for Ricker source wavelet
    :param omega_samples: Sequence of angular frequencies
    :param quiet: If True, progress bar output is disabled
    :return: Time domain signal (seismograms) for all source/receiver
    combinations
    """
    # TODO  add checking for correct dimensionality of source/receiver arrays with documentation and exceptions
    p_wave_spectrum: List[complex] = []
    omegas_iterator = tqdm(omega_samples, desc="Born modeling",
                           unit="frequency samples",
                           disable=quiet, leave=False)
    for omega in omegas_iterator:
        u_scattering = born_all_scatterers(source_pos, receiver_pos,
                                           velocity_model, omega, omega_central)
        p_wave_spectrum.append(u_scattering)
    p_wave_spectrum: np.ndarray = np.array(p_wave_spectrum)
    # axis 0 contains the time samples
    time_domain = np.real(np.fft.ifft(p_wave_spectrum, axis=0))
    # shape is (N, A, B) where N is the number of time samples, A the number of
    # sources and B the number of receivers. Move N to the back to be able to
    # select seismograms for source/receiver combinations
    time_domain = np.moveaxis(time_domain, 0, -1)
    return time_domain


def angular(f: Hertz) -> RadiansPerSecond:
    return RadiansPerSecond(2. * math.pi * f)


def frequency_samples(timeseries_length: Seconds, sample_period: Seconds) -> np.ndarray:
    """
    Calculate frequency samples required to reach the given length and sample
    period after the inverse Fourier transform.
    """
    num_of_samples = int(timeseries_length / sample_period)
    delta_omega = 2*math.pi / timeseries_length
    omega_max = num_of_samples * delta_omega
    f_samples = np.linspace(0, omega_max, num_of_samples)
    return f_samples


def time_samples(timeseries_length: Seconds, sample_period: Seconds) -> np.ndarray:
    """
    Calculate all time points between 0 and time series_length when the time
    series is sampled with the given period.
    """
    num_of_samples = int(timeseries_length / sample_period)
    return np.linspace(0, timeseries_length, num_of_samples)


def ricker_frequency_domain(omega: float, omega_central: float) -> float:
    """
    Taken from Frequencies of the Ricker wavelet by Yanghua Wang (eq. 7)
    :param omega: Frequency at which to evaluate the spectrum
    :param omega_central: Central or dominant frequency
    :return: Value of frequency spectrum of a Ricker wavelet
    """
    return 2 * omega**2 / (math.sqrt(math.pi) * omega_central**3) \
           * math.exp(- omega**2 / omega_central**2)


def scattering_potential(v: float, v0: float) -> float:
    """
    Scattering potential with respect to the homogeneous medium
    Eq. 3 from Hu2018a
    :param v: velocity at position x prime
    :param v0: homogeneous background velocity
    """
    return 1 / v0**2 - 1 / v**2
