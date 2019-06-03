import cmath
import math
from typing import Tuple

import fastfunctions
from scipy import integrate

from VelocityModel import Vector3D, AbstractVelocityModel
from units import RadiansPerSecond, KgPerCubicMeter, MetersPerSecond


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


def length(vector: Vector3D) -> float:
    """Calculate length of vector"""
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
    # pull out length from formual to calculate it only once
    l = length(x - x_prime)
    # split formula for better readability
    a = 1. / (4. * math.pi * density * v0**2 * l)
    b = cmath.exp(-1j * omega * l / v0)
    return a * b


def integral(x: float, y: float, z: float, additional_params: Tuple):
    """
    Triple integral from Hu2018a eq. 1.
    :param x: x coordinate of x_prime
    :param y: y coordinate of x_prime
    :param z: z_coordinate of x_prime
    :param additional_params:
    :return:
    """
    x_prime = (x, y, z)
    density, v0, omega, velocity_model, xs, xr = additional_params
    v = velocity_model.eval_at(x_prime)
    if v == v0:
        # early exit to save function evaluations in inner loop
        # scattering potential epsilon is 0 when v = v0 and the full integral term
        # is a product with that 0
        return 0.
    epsilon = fastfunctions.scattering_potential(v, v0)
    G0_left = fastfunctions.greens_function(density, v0, xs, x_prime, omega)
    G0_right = fastfunctions.greens_function(density, v0, x_prime, xr, omega)
    return G0_left * epsilon * G0_right


def born_modeling(xs: Vector3D, xr: Vector3D, omega: RadiansPerSecond, omega_central: RadiansPerSecond,
                  density: KgPerCubicMeter, velocity_model: AbstractVelocityModel, epsabs: float = 1E-16,
                  epsrel: float = 1E-16, limit: int = 50) -> complex:
    """
    Calculate scattered P wave in the frequency domain (eq. 1 from Hu2018a) using
    born approximation. Evaluate for a frequency range to get the P wave spectrum.
    Source wavelet is a Ricker wavelet.
    :param xs: Source position
    :param xr: Receiver position
    :param omega: Frequency for which the P wave spectrum should be calculated.
    :param omega_central: Central frequency of Ricker source wavelet
    :param density: Density in kg/m^3 of Model
    :param velocity_model: Used to evaluate velocity at integration point
    :param epsabs: Absolute error tolerance of integration, passed on to scipy.integrate.nquad
    :param epsrel: Relative error tolerance of integration, passed on to scipy.integrate.nquad
    :param limit: An upper bound on the number of subintervals used in the adaptive algorithm,
    passed on to scipy.integrate.nquad
    :return:
    """
    v0 = velocity_model.bg_vel  # type: MetersPerSecond

    # integration limits. Since integral is zero everywhere outside of the scatterers,
    # we can limit the integration area
    x_start = 0.
    x_end = velocity_model.x_width
    y_start = 0.
    y_end = velocity_model.y_width
    z_start = velocity_model.scatterer_top
    z_end = velocity_model.scatterer_bottom
    integration_limits = [(x_start, x_end), (y_start, y_end), (z_start, z_end)]

    # scipy cant integrate complex functions, so we split integration into real
    # and imaginary part
    def real_func(x, y, z, *args):
        return integral(x, y, z, args).real

    def imag_func(x, y, z, *args):
        return integral(x, y, z, args).imag

    # additional arguments for integration that are passed to the integral function
    args = (density, v0, omega, velocity_model, xs.data, xr.data)

    # integration options
    opts = [{"epsabs": epsabs, "epsrel": epsrel, "limit": limit} for _ in range(3)]

    # Calculate triple integral
    # integration is carried out in order, x is the innermost integral, z the outermost
    y_real, *_ = integrate.nquad(real_func, integration_limits, args=args, opts=opts)
    y_imag, *_ = integrate.nquad(imag_func, integration_limits, args=args, opts=opts)

    # Add factor
    factor = fastfunctions.ricker_frequency_domain(omega, omega_central) * omega**2
    y_real *= factor
    y_imag *= factor
    return complex(y_real, y_imag)
