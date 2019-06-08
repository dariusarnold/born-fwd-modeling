import time

import fastfunctions as ff
import functions as f
import numpy as np
import quadpy
from VelocityModel import Vector3D, create_scatterers
from main import angular, frequency_samples
from units import RadiansPerSecond, KgPerCubicMeter, MetersPerSecond, Meter


from alternative import integral


def born_single_scatterer(xs: Vector3D, xr: Vector3D, scatterer_pos: np.ndarray,
                          omega: RadiansPerSecond, omega_central: RadiansPerSecond,
                          density: KgPerCubicMeter, bg_vel: MetersPerSecond,
                          scatterer_radius: Meter) -> complex:
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

    def greens_function_vectorized(x, x_prime):
        """
        Vectorized version of the greens function using numpy.
        """
        lengths = np.linalg.norm(x - x_prime, axis=1)
        return 1. / lengths * np.exp(-1j * omega * lengths / bg_vel)

    G0_left = lambda xs, x_prime: greens_function_vectorized(xs, x_prime.T)
    G0_right = lambda x_prime, xr: greens_function_vectorized(x_prime.T, xr)

    def integral(x):
        """x is a np array containing all points to be evaluated as columns"""
        return G0_left(xs, x) * G0_right(x, xr)

    res = quadpy.ball.integrate(integral, scatterer_pos, scatterer_radius,
                                quadpy.ball.HammerStroud("14-3a"))
    res *= ff.ricker_frequency_domain(omega, omega_central) * omega**2
    res *= 1 / (4. * np.pi * density * bg_vel**2)
    return res


if __name__ == '__main__':

    xs = np.array((11200.0, 5600.0, 10.0))
    xr = np.array((5272.0, 3090.0, 0.0))
    scatterer_pos = np.array((5000., 5000., 2400.))
    omega = angular(10.)
    omega_central = angular(30.)
    density = 2550.
    bg_vel = 4800.
    scatterer_radius = 1.

    scatterers = np.array(create_scatterers())
    #res = sum(born_single_scatterer(xs, xr, scatterer, omega, omega_central,
    #                                 density, bg_vel, scatterer_radius) for scatterer in scatterers)
    f_samples = frequency_samples(4, 0.004)
    spectrum = []
    for f in f_samples:
        before = time.time()
        res = sum(born_single_scatterer(xs, xr, scatterer, omega, omega_central,
                                        density, bg_vel, scatterer_radius) for scatterer in scatterers)
        spectrum.append(res)
        after = time.time()
        print(after-before, " s")
    print(res)
