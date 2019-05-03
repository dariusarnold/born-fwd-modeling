import cmath
import functools
import math
from typing import Tuple

import scipy
from scipy import integrate

from VelocityModel import Vector3D, VelocityModel


def ricker_frequency_domain(omega: float, omega_central: float) -> float:
    """
    Taken from Frequencies of the Ricker wavelet by Yanghua Wang (eq. 7)
    :param omega: Frequency at which to evaluate the spectrum
    :param omega_central: Central or dominant frequency
    :return: Value of frequency spectrum of a Ricker wavelet
    """
    return 2 * omega ** 2 / (math.sqrt(math.pi) * omega_central ** 3) * math.exp(- omega ** 2 / omega_central ** 2)


def scattering_potential(v: float, v0: float) -> float:
    """
    Scattering potential with respect to the homogeneous medium
    Eq. 3 from Hu2018a
    :param v: velocity at position x prime
    :param v0: homogeneous background velocity
    :return:
    """
    return 1 / v0**2 - 1/v**2


def length(vector: Vector3D) -> float:
    """
    Calculate length of vector
    """
    return math.sqrt(sum(component**2 for component in vector))


def greens_function(density: float, v0: float, x: Vector3D, x_prime: Vector3D,
                    omega: float) -> complex:
    """
    Measures P wave response at x due to a source at x_prime in the homogeneous
    background medium
    :param density: density of homogeneous background medium in kg/m^3
    :param v0: P wave velocity in the homogeneous medium
    :param x: Position of response
    :param x_prime: Position of source
    :param omega: Frequency
    :return:
    """
    # split formula for better readability
    a = 1 / (4*math.pi * density * v0**2 * length(x - x_prime))
    b = cmath.exp(-1j * omega * length(x - x_prime) / v0)
    return a * b


def born_modeling(xs: Vector3D, xr: Vector3D, omega: float, w_central: float, density: float,
                  velocity_model: VelocityModel) -> complex:
    """

    :param xs:
    :param xr:
    :param omega:
    :param w_central:
    :param density:
    :param velocity_model:
    :return:
    """
    v0 = velocity_model.bg_vel
    W = functools.partial(ricker_frequency_domain, omega_central=w_central)
    G0 = functools.partial(greens_function, density, v0)
    epsilon = functools.partial(scattering_potential, v0=v0)

    x_start = 0.
    x_end = velocity_model.x_width
    y_start = 0.
    y_end = velocity_model.y_width
    z_start = 0
    z_end = velocity_model.scatterer_max_depth

    def integral(z, y, x):
        x_prime = Vector3D(x, y, z)
        return G0(xs, x_prime, omega) * epsilon(velocity_model.eval_at(x_prime)) * G0(x_prime, xr, omega)

    # scipy cant integrate complex functions, so we split integration into real
    # and imaginary part
    def real_func(z, y, x):
        return scipy.real(integral(z, y, x))

    def imag_func(z, y, x):
        return scipy.imag(integral(z, y, x))

    y_real, abserr_real = integrate.tplquad(real_func, x_start, x_end,
                                  y_start, y_end,
                                  z_start, z_end)
    y_imag, abserr_imag = integrate.tplquad(imag_func, x_start, x_end,
                                            y_start, y_end,
                                            z_start, z_end)

    y_real *= W(omega) * omega ** 2
    y_imag *= W(omega) * omega ** 2
    return complex(y_real, y_imag)
