import time
from typing import Sequence

import fastfunctions as ff
from tqdm import tqdm

import functions as f
import numpy as np
import quadpy
from VelocityModel import Vector3D, create_scatterers
from main import angular, frequency_samples
from plotting import plot_time_series
from units import RadiansPerSecond, KgPerCubicMeter, MetersPerSecond, Meter, Hertz


def born_all_scatterers(xs: Vector3D, xr: Vector3D, scatterer_pos: np.ndarray,
                        omega: RadiansPerSecond, omega_central: RadiansPerSecond,
                        density: KgPerCubicMeter, bg_vel: MetersPerSecond,
                        frac_vel: MetersPerSecond, scatterer_radius: Meter) -> complex:
    """
    Calculate born scattering for all scatterer positions
    This implements eq. (1) from
    3D seismic characterization of fractures in a dipping layer using the double-beam method
    Hao Hu and Yingcai Zheng
    :param xs: Source position
    :param yr: Receiver position
    :param omega: Angular frequency of this sample
    :param omega_central: Central frequency of ricker source wavelet
    :param density: Homogeneous background density of model
    :param bg_vel: Homogeneous background velocity
    :param frac_vel: Velocity of the fractures, which are represented by the
    scatterers
    :param scatterer_radius: Radius of a scatterer
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
        return 1./lengths * complex_exp(-omega * (1. / bg_vel) * lengths)

    def integral(x):
        """
        x is a np array containing all points to be evaluated.
        Its shape is (3, M, N) where M is the number of scatterer midpoints
        returned by create_scatterers and N is the number of evaluation points
        chosen by quadpy. The first axis is fixed (3). It represents the x, y, z
        values, eg. x[0] contains all x values of all points.
        """
        epsilon = ff.scattering_potential(frac_vel, bg_vel)
        # extend the 3D vector from shape (3,) to (3, 1, 1) so numpy
        # broadcasting works as expected
        G0_left = greens_function_vectorized(xs[:, None, None], x)
        G0_right = greens_function_vectorized(x, xr[:, None, None])
        return G0_left * epsilon * G0_right

    scatterer_radii = np.full(len(scatterer_pos), scatterer_radius)
    integration_scheme = quadpy.ball.HammerStroud("14-3a")
    # sum over the result from all scatterer points
    res = np.sum(quadpy.ball.integrate(integral, scatterer_pos, scatterer_radii, integration_scheme))
    res *= ff.ricker_frequency_domain(omega, omega_central) * omega**2
    res *= 1 / (4. * np.pi * density * bg_vel**2)
    return res


def born(source_pos: np.array, receiver_pos: np.array, scatterer_positions: np.array,
         omega_central: RadiansPerSecond, omega_samples: Sequence[RadiansPerSecond],
         density: KgPerCubicMeter, background_vel: MetersPerSecond, fracture_vel:
         MetersPerSecond, scatterer_radius: Meter, quiet: bool = False) -> np.array:
    """Calculate complete seismogram"""
    spectrum = []
    for omega in tqdm(omega_samples, desc="Born modeling", total=len(omega_samples), unit="frequency samples", disable=quiet):
        u_scattering = born_all_scatterers(source_pos, receiver_pos, scatterer_positions,
                                           omega, omega_central, density,
                                           background_vel, fracture_vel, scatterer_radius)
        spectrum.append(u_scattering)
    time_domain = np.real(np.fft.ifft(spectrum))
    return time_domain


if __name__ == '__main__':

    xs = np.array((11200.0, 5600.0, 10.0))
    xr = np.array((5272.0, 3090.0, 0.0))
    scatterer_pos = np.array((5000., 5000., 2400.))
    omega_central = angular(Hertz(30.))
    density = KgPerCubicMeter(2550.)
    bg_vel = MetersPerSecond(4800.)
    frac_vel = MetersPerSecond(4320.)
    scatterer_radius = Meter(1.)

    scatterers = np.array(create_scatterers())
    omega_samples = frequency_samples(4, 0.004)
    time_domain = born(xs, xr, scatterers, omega_central, omega_samples, density,
                       bg_vel, frac_vel, scatterer_radius)
    plot_time_series(time_domain)