import math

import numpy as np

from units import Hertz, RadiansPerSecond, Seconds
from typing import List, Sequence
from VelocityModel import AbstractVelocityModel
from tqdm import tqdm
from quadpy_integration import born_all_scatterers


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
    :return:
    """
    return 1 / v0**2 - 1 / v**2
