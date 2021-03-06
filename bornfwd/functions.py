import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import quadpy
import numexpr
from tqdm import tqdm

from bornfwd.io import save_seismogram, create_header
from bornfwd.units import Hertz, RadiansPerSecond, Seconds
from bornfwd.velocity_model import AbstractVelocityModel


def _born(xs: np.ndarray, xr: np.ndarray, velocity_model: AbstractVelocityModel,
          omega: np.ndarray, omega_central: RadiansPerSecond,
          ricker_amplitude: float) -> np.ndarray:
    """
    Calculate born scattering for all scatterers over a range of frequencies.
    This implements eq. (1) from
    3D seismic characterization of fractures in a dipping layer using the
    double-beam method
    Hao Hu and Yingcai Zheng
    :param xs: Source position array, shape (3, 1).
    :param xr: Receiver position array, shape (M, 3).
    :param velocity_model: Model containing geometry of scatterers, velocities
    and density.
    :param omega: Angular frequencies in (K,) array where K is the number of
    samples.
    :param omega_central: Central frequency of ricker source wavelet.
    :param ricker_amplitude: Amplitude of the Ricker source wavelet in the
    frequency domain.
    :return: Frequency spectrum of the scattered P wave for all omega sample
    points. If M receiver positions were specified, the returned array will be
    of shape (M, K).
    """
    def greens_function_vectorized(x: np.array, x_prime: np.array) -> np.array:
        """
        Vectorized version (over the frequencies in omega) of the greens
        function.
        """
        # Calculate magnitude of vectors
        subtraction = x - x_prime
        lengths = np.sqrt(np.einsum("ijkl, ijkl -> jkl", subtraction, subtraction,
                                    optimize=True))
        # capture variables in closure since I can't find a way to access them
        # in the numexpr expression otherwise
        bg_vel_ = bg_vel
        omega_ = omega[None, :, None, None]
        lengths_ = lengths[:, None, ...]
        return numexpr.evaluate("exp(-omega_ * (1j / bg_vel_) * lengths_) / lengths_")

    def integral(x):
        """
        Modified version of the integrand from equation (1), Hu2018.
        Constant factors are moved out of the integral and are multiplied later.
        x is a np array containing all points to be evaluated.
        Its shape is (3, M, N) where M is the number of scatterer midpoints
        returned by create_scatterers and N is the number of evaluation points
        chosen by quadpy. The first axis is fixed (3). It represents the x, y, z
        values, eg. x[0] contains all x values of all points.
        """
        # xs, xr need to be extended  from (N, 3) to (N, 3, 1, 1) for numpy
        # broadcasting to work.
        G0_left = greens_function_vectorized(xs[..., None, None], x[:, None, ...])
        G0_right = greens_function_vectorized(x[:, None, ...], xr[..., None, None])
        return numexpr.evaluate("G0_left * G0_right")

    frac_vel = velocity_model.fracture_velocity
    bg_vel = velocity_model.background_velocity
    epsilon = scattering_potential(frac_vel, bg_vel)
    scatterer_radii = np.full(len(velocity_model.scatterer_positions),
                              velocity_model.scatterer_radius, dtype=np.float32)
    scheme = quadpy.ball.stroud_3_1()
    # hack to compute everything with 32 bit floats
    scheme.points = scheme.points.astype(np.float32)
    scheme.weights = scheme.weights.astype(np.float32)
    res = scheme.integrate(integral, velocity_model.scatterer_positions,
                           scatterer_radii,
                           dot=lambda x, y: np.einsum("ijkl, l", x, y, optimize=True))
    # sum over the result from all scatterer points
    res = np.sum(res, axis=-1)
    res *= ricker_frequency_domain(omega, omega_central, ricker_amplitude) \
           * omega**2 * epsilon
    # multiply by constant factor pulled from Greens function so only
    # one multiplication is done per frequency bin.
    res *= 1 / (4. * np.pi * velocity_model.density * bg_vel**2)
    return res


def born_single(source_position: np.ndarray, receiver_position: np.ndarray,
                velocity_model: AbstractVelocityModel,
                omega_central: RadiansPerSecond, ricker_amplitude: float,
                timeseries_length: Seconds,
                sample_period: Seconds) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate seismogram for shot and receiver position.
    :param source_position: (x y z) coordinates of source
    :param receiver_position: (x y z) coordinates of receivers
    :param velocity_model: VelocityModel instance
    :param omega_central: Central frequency of Ricker source wavelet
    :param ricker_amplitude: Amplitude of the Ricker source wavelet in the
    frequency domain.
    :param timeseries_length: Desired length of seismograms in seconds
    :param sample_period: Desired samplerate of seismograms in seconds
    :return: Tuple of two arrays: First array is timesteps where the seismogram
    was calculated, second array is the seismogram
    """
    omega_samples = frequency_samples(timeseries_length, sample_period)
    t_samples = time_samples(timeseries_length, sample_period)
    u_scattering = _born(source_position.reshape(3, 1),
                         receiver_position.reshape(3, 1), velocity_model,
                         omega_samples, omega_central, ricker_amplitude)
    time_domain = np.real(np.fft.ifft(np.squeeze(u_scattering)))
    return t_samples, time_domain


def born_multi(source_positions: np.ndarray, receiver_positions: np.ndarray,
               velocity_model: AbstractVelocityModel, omega_central: RadiansPerSecond,
                ricker_amplitude: float, timeseries_length: Seconds,
               sample_period: Seconds, chunksize: int) -> None:
    """
    Generate scattered P wave for multiple source/receiver locations and save
    them as files.
    :param source_positions: (N, 3) array of coordinates of N source positions.
    Second axis with length 3 contains (x y z) points.
    :param receiver_positions: (M, 3) array of coordinates of M receiver positions.
    :param velocity_model: Velocity model instance containing scatterer positions
    and other data.
    :param omega_central: Central frequency for Ricker source wavelet
    :param ricker_amplitude: Amplitude of Ricker source wavelet in frequency
    domain.
    :param timeseries_length: Desired length of seismograms in seconds
    :param sample_period: Desired samplerate of seismograms in seconds
    :param chunksize: Number of receivers for which seismograms are calculated
    in parallel
    """
    if len(source_positions.shape) != 2 or source_positions.shape[1] != 3:
        raise ValueError("Shape mismatch for source position: Got "
                         f"{source_positions.shape}, expected (N, 3).")
    # add axes so that indexing or iterating an array of shape (N, 3, 1)
    # gives (3, 1) array instead of (3,), as expected by _born function
    source_positions = source_positions[..., None]
    # assert that receiver positions have shape (M, 3)
    if len(receiver_positions.shape) > 2 or receiver_positions.shape[1] != 3:
        raise ValueError("Shape mismatch for receiver position: Got"
                         f"{receiver_positions.shape}, expected (M, 3).")

    # TODO replace hardcoded paths with command line options
    source_folder_name = "source_{{id:0{number_of_digits}d}}".format(number_of_digits=len(str(source_positions.shape[0])))
    output_folder = os.path.join("output", source_folder_name)
    output_filename = "receiver_{{id:0{number_of_digits}d}}.txt".format(number_of_digits=len(str(receiver_positions.shape[0])))

    omega_samples = frequency_samples(timeseries_length, sample_period)
    t_samples = time_samples(timeseries_length, sample_period)

    split_positions = range(chunksize, len(receiver_positions), chunksize)
    receiver_chunks = np.array_split(receiver_positions, split_positions)
    for index_source, source_position in enumerate(source_positions):
        # generate source folder
        fpath = Path(output_folder.format(id=index_source+1))
        fpath.mkdir(parents=True, exist_ok=True)
        progress_bar = tqdm(receiver_chunks, desc=f"Source {index_source+1:03d}/{len(source_positions):03d}",
                            unit="chunk", leave=True)
        for index_chunk, receiver_chunk in enumerate(progress_bar):
            # calculate seismograms
            u_scattering = _born(source_position, receiver_chunk.T,
                                 velocity_model, omega_samples,
                                 omega_central, ricker_amplitude)
            u_scattering = np.real(np.fft.ifft(u_scattering, axis=-1))
            # save seismograms
            for seismogram_index, seismogram in enumerate(u_scattering):
                header = create_header(source_position, receiver_chunk[seismogram_index])
                receiver_id = index_chunk * chunksize + seismogram_index + 1
                fname = Path(output_filename.format(id=receiver_id))
                save_seismogram(seismogram, t_samples, header, fpath/fname)


def angular(f: Hertz) -> RadiansPerSecond:
    return RadiansPerSecond(2. * math.pi * float(f))


def frequency_samples(timeseries_length: Seconds, sample_period: Seconds) -> np.ndarray:
    """
    Calculate frequency samples required to reach the given length and sample
    period after the inverse Fourier transform.
    """
    num_of_samples = int(timeseries_length / sample_period)
    delta_omega = 2*math.pi / timeseries_length
    omega_max = num_of_samples * delta_omega
    f_samples = np.linspace(0, omega_max, num_of_samples, dtype=np.float32)
    return f_samples


def time_samples(timeseries_length: Seconds, sample_period: Seconds) -> np.ndarray:
    """
    Calculate all time points between 0 and time series_length when the time
    series is sampled with the given period.
    """
    num_of_samples = int(timeseries_length / sample_period)
    return np.linspace(0, timeseries_length, num_of_samples)


def ricker_frequency_domain(omega: np.ndarray, omega_central: float,
                            amplitude: float) -> np.ndarray:
    """
    Taken from Frequencies of the Ricker wavelet by Yanghua Wang (eq. 7)
    :param omega: Frequency at which to evaluate the spectrum
    :param omega_central: Central or dominant frequency
    :return: Value of frequency spectrum of a Ricker wavelet
    """
    return amplitude * 2 * omega**2 / (math.sqrt(math.pi) * omega_central**3) \
        * np.exp(- omega**2 / omega_central**2)


def scattering_potential(v: float, v0: float) -> float:
    """
    Scattering potential with respect to the homogeneous medium
    Eq. 3 from Hu2018a
    :param v: velocity at position x prime
    :param v0: homogeneous background velocity
    """
    return 1 / v0**2 - 1 / v**2
