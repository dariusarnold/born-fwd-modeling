import itertools
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import fastfunctions as ff
from tqdm import tqdm

import functions as f
import matplotlib.pyplot as plt
import numpy as np

from main import create_scatterers


def create_scatterer_point_cloud(scatterer_edge_length_m: float,  points_per_edge: int):
    points_per_edge = np.linspace(0, scatterer_edge_length_m, points_per_edge)
    # get (x, y, z) tuples representing sample points in the scatterer
    scatterer_pointcloud = np.array(list(itertools.product(*(points_per_edge for _ in range(3)))))
    # center the point cloud around (0, 0, 0)
    scatterer_pointcloud -= scatterer_edge_length_m/2
    return scatterer_pointcloud


def born_modeling(xs, xr, omega: float, omega_central: float, density, vel_bg, vel_frac):
    xs = np.asarray(xs)
    xr = np.asarray(xr)
    factor = ff.ricker_frequency_domain(omega, omega_central) * omega**2
    scatterer_pointcloud = create_scatterer_point_cloud(10, 3)
    scatterers = create_scatterers()
    scatterers = np.array([pos.data for pos in scatterers])
    eps = ff.scattering_potential(vel_frac, vel_bg)  # constant within a scatterer, else 0 (outside)
    for scatterer_position in scatterers:
        # iterate over scatterer positions
        integral = 0
        for x_prime in scatterer_pointcloud:
            # iterate over the sample grid points
            x_prime += scatterer_position
            G0_left = f.greens_function(density, vel_bg, xs, x_prime, omega)
            G0_right = f.greens_function(density, vel_bg, x_prime, xr, omega)
            integral += G0_left * eps * G0_right
    return factor * integral


def main(processing="parallel", num_cores=None):
    num_frequency_steps = 16
    p_wave_spectrum = []
    omega_central = 2 * math.pi * 30
    xs = (11200, 11200/2, 10.)
    xr = (5272., 3090., 0.)
    frequency_samples = np.linspace(1, 100, num_frequency_steps) * 2 * math.pi
    if processing == "serial":
        for omega in frequency_samples:
            res = born_modeling(xs, xr, omega, omega_central, 2550, 4800, 4320)
            p_wave_spectrum.append((omega, res))
    elif processing == "parallel":
        with ProcessPoolExecutor(num_cores) as process_pool:
            futures = []
            fut_freq_mapping = {}
            for omega in frequency_samples:
                future = process_pool.submit(born_modeling, xs, xr, omega, omega_central, 2550, 4800, 4320)
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