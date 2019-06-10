import itertools
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import sqrt

import fastfunctions as ff
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import nquad
from tqdm import tqdm

import functions as f
from VelocityModel import create_scatterers


def integral(x, y, z, additional_params):
    density, v0, xs, xr, omega = additional_params
    x_prime = (x, y, z)
    G0_left = ff.greens_function(density, v0, xs, x_prime, omega)
    G0_right = ff.greens_function(density, v0, x_prime, xr, omega)
    return G0_left * G0_right


def born_modeling(xs, xr, omega: float, omega_central: float, density, vel_bg, vel_frac, scatterer_radius):

    xs = np.asarray(xs)
    xr = np.asarray(xr)

    # scipy cant integrate complex functions, so we split integration into real
    # and imaginary part
    def real_func(x, y, z, *args):
        return ff.integral_sphere(x, y, z, args).real

    def imag_func(x, y, z, *args):
        return ff.integral_sphere(x, y, z, args).imag


    # additional arguments for integration that are passed to the integral function
    args = (density, vel_bg, xs, xr, omega)

    # integration options
    #opts = [{"epsabs": epsabs, "epsrel": epsrel, "limit": limit} for _ in range(3)]
    y_real = 0
    y_imag = complex()
    scatterers = create_scatterers()
    for i, scatterer_pos in enumerate(scatterers):
        print(i)
        # this is the center of the sphere about which we wish to integrate
        x0, y0, z0 = scatterer_pos
        # set integration borders for integration over a sphere
        r = scatterer_radius
        # *args are the additional arguments given to the integration containing density etc.
        circle_boundaries = [
            lambda y, z, *args: [-math.sqrt(r**2 - (y-y0)**2 - (z-z0)**2) + x0, math.sqrt(r**2 - (y-y0)**2 - (z-z0)**2) + x0],
            lambda z, *args: [-sqrt(r**2 - (z-z0)**2) + y0, sqrt(r**2 - (z-z0)**2) + y0],
            lambda *args: [-r+z0, r+z0]]
        # option for integration
        opts = [{"limit": 1} for _ in range(3)]
        y_real += nquad(real_func, circle_boundaries, args=args, opts=opts)[0]
        y_imag += nquad(imag_func, circle_boundaries, args=args, opts=opts)[0]

    factor = ff.ricker_frequency_domain(omega, omega_central) * omega**2
    factor *= ff.scattering_potential(vel_frac, vel_bg)  # constant within scatterers
    y_real *= factor
    y_imag *= factor
    return complex(y_real, y_imag)


def main(processing="serial", num_cores=None):
    num_frequency_steps = 1
    p_wave_spectrum = []
    omega_central = 2 * math.pi * 30
    xs = (11200, 11200/2, 10.)
    xr = (5272., 3090., 0.)
    scatterer_radius = 1.
    frequency_samples = np.linspace(1, 100, num_frequency_steps) * 2 * math.pi
    if processing == "serial":
        for omega in frequency_samples:
            res = born_modeling(xs, xr, omega, omega_central, 2550., 4800., 4320., scatterer_radius)
            p_wave_spectrum.append((omega, res))
    elif processing == "parallel":
        with ProcessPoolExecutor(num_cores) as process_pool:
            futures = []
            fut_freq_mapping = {}
            for omega in frequency_samples:
                future = process_pool.submit(born_modeling, xs, xr, omega, omega_central, 2550., 4800., 4320., scatterer_radius)
                futures.append(future)
                fut_freq_mapping[future] = omega
            for future in tqdm(as_completed(futures), desc="Born modeling", total=num_frequency_steps, unit="frequency samples"):
                res = future.result()
                omega = fut_freq_mapping[future]
                p_wave_spectrum.append((omega, res))
    # sort spectrum by frequency
    p_wave_spectrum = sorted(p_wave_spectrum, key=lambda x: x[0])
    time_domain = np.fft.ifft(p_wave_spectrum)
    # throws ComplexWarning
    plt.plot(time_domain)
    plt.show()


if __name__ == '__main__':
    main()