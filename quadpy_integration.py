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
    Calculate born scattering for a single scatterer position
    :param xs:
    :param yr:
    :param omega:
    :param omega_central:
    :param density:
    :param bg_vel:
    :param frac_vel:
    :param scatterer_radius:
    :return:
    """

    def exp(exp_term):
        """Interestingly in numpy a complex exp takes more time to compute
        than the expanded version from eulers formula
        see: https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/758148"""
        return np.cos(exp_term) + 1j * np.sin(exp_term)

    def greens_function_vectorized(x, x_prime):
        """
        Vectorized version of the greens function using numpy.
        """
        subtraction = x - x_prime
        lengths = np.sqrt(np.einsum("ijk,ijk->jk", subtraction, subtraction))
        # minus sign in exp term is required since it was exp(-ix) before, which
        # transforms to cos(-x) + i * sin(-x)
        return 1/lengths * exp(-omega * lengths / bg_vel)

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

    res = np.sum(quadpy.ball.integrate(integral, scatterer_pos, np.full(len(scatterer_pos), scatterer_radius),
                                       quadpy.ball.HammerStroud("14-3a")))
    res *= ff.ricker_frequency_domain(omega, omega_central) * omega**2
    res *= 1 / (4. * np.pi * density * bg_vel**2)
    return res


def born(source_pos: np.array, receiver_pos: np.array, scatterer_positions: np.array,
         omega_central: RadiansPerSecond, omega_samples: Sequence[RadiansPerSecond],
         density: KgPerCubicMeter, background_vel: MetersPerSecond, fracture_vel:
         MetersPerSecond, scatterer_radius: Meter) -> np.array:
    """Calculate complete seismogram"""
    spectrum = []
    for omega in tqdm(omega_samples, desc="Born modeling", total=len(omega_samples), unit="frequency samples"):
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