import time

import fastfunctions as ff
import functions as f
import numpy as np
import quadpy
from VelocityModel import Vector3D, create_scatterers
from main import angular, frequency_samples
from plotting import plot_time_series
from units import RadiansPerSecond, KgPerCubicMeter, MetersPerSecond, Meter, Hertz

from alternative import integral


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

    def greens_function_vectorized(x, x_prime):
        """
        Vectorized version of the greens function using numpy.
        """
        subtraction = x - x_prime
        lengths = np.sqrt(np.einsum("ijk,ijk->jk", subtraction, subtraction))
        return 1. / lengths * np.exp(-1j * omega * lengths / bg_vel)

    def G0_left(xs, x_prime):
        """The left Greens function has xs as a 3D vector, while x_prime
        contains all evaluation coordinates"""
        # add empty axes to 3D vector so subtraction with full x_prime works
        return greens_function_vectorized(xs[:, None, None], x_prime)

    def G0_right(x_prime, xr):
        """The right greens function has xr as a 3D vector and x_prime as the
        array containging all evaluation coordinates"""
        # add empty axes to 3D vector fur subtraction
        return greens_function_vectorized(x_prime, xr[:, None, None])

    def integral(x):
        """x is a np array containing all points to be evaluated.
        Its shape is (3, M, N) where M is the number of scatterer midpoints
        returned by create_scatterers and N is the number of evaluation points
        chosen by quadpy. The first axis is fixed (3). It represents the x, y, z
        values, eg. x[0] contains all x values of all points."""
        epsilon = ff.scattering_potential(frac_vel, bg_vel)
        return G0_left(xs, x) * epsilon * G0_right(x, xr)

    res = np.sum(quadpy.ball.integrate(integral, scatterer_pos, np.full(len(scatterer_pos), scatterer_radius),
                                quadpy.ball.HammerStroud("14-3a")))
    res *= ff.ricker_frequency_domain(omega, omega_central) * omega**2
    res *= 1 / (4. * np.pi * density * bg_vel**2)
    return res


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
    f_samples = frequency_samples(4, 0.004)
    spectrum = []
    for omega in f_samples:
        before = time.time()
        res = born_all_scatterers(xs, xr, scatterers, omega, omega_central,
                                  density, bg_vel, frac_vel, scatterer_radius)
        spectrum.append(res)
        after = time.time()
        print(after-before, " s")
    print(spectrum)
    time_domain = np.real(np.fft.ifft(spectrum))
    plot_time_series(time_domain)