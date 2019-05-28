import itertools
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from VelocityModel import VelocityModel, Vector3D
from functions import born_modeling


def create_scatterers():
    model_width = 11200.
    # values taken from Fig4 in Hu2018a, all units m
    frac_depth = 2400.
    # fractures are in a square with sides parallel to the axes
    # values taken from Fig 4 Hu2018a
    frac_top_left_x = 600.
    frac_top_left_y = model_width - 1300
    frac_bot_right_x = model_width - 1400
    frac_bot_right_y = 860.
    frac_spacing = 200.
    # distance between scatterer points, calculated as 1/8 p wave length
    scatterer_spacing = 20.
    # + 1 to cover the "last" point
    number_of_fractures = int((frac_top_left_y - frac_bot_right_y) / frac_spacing) + 1
    scatterers_per_fracture = int((frac_bot_right_x - frac_top_left_x) / scatterer_spacing) + 1
    # step parallel to y axis
    scatterer_y_positions = np.linspace(frac_bot_right_y, frac_top_left_y, number_of_fractures)
    # step parallel to x axis
    scatterer_x_positions = np.linspace(frac_top_left_x, frac_bot_right_x, scatterers_per_fracture)
    xy_combinations = itertools.product(scatterer_x_positions, scatterer_y_positions)
    scatterers = [Vector3D(x, y, frac_depth) for x, y in xy_combinations]
    return scatterers


def create_velocity_model():
    background_vel = 4800.
    fracture_vel = 4320.
    x_width = 11200
    y_width = 11200
    scatterer_positions = create_scatterers()
    scatterer_radius = 100
    return VelocityModel(background_vel, fracture_vel, x_width, y_width, scatterer_positions, scatterer_radius)


def plot_fractures(velocity_model: VelocityModel):
    x = list(point.x for point in velocity_model.scatterer_positions)
    y = list(point.y for point in velocity_model.scatterer_positions)
    plt.scatter(x, y, marker=",", c="g", s=1)
    plt.title("Scatterer points in model")
    plt.xlabel("x axis (West-East, m)")
    plt.ylabel("y axis (South-North, m)")
    plt.tight_layout()
    plt.axis("equal")
    plt.xlim((0, 11200))
    plt.ylim((0, 11200))
    plt.show()


def angular(f):
    return 2.*math.pi*f


def main():
    model = create_velocity_model()

    # parameters
    source_pos = Vector3D(model.x_width, model.y_width/2, 10.)
    receiver_pos = Vector3D(5272., 3090., 0.)
    # angular frequencies in rad/s
    omega_central = angular(30)
    num_of_frequency_steps = 16
    frequency_min = angular(1)
    frequency_max = angular(100)
    processing = "parallel"
    # Move density into velocity model
    density = 2550.  # kg/m^3

    frequency_samples = np.linspace(frequency_min, frequency_max, num_of_frequency_steps)
    p_wave_spectrum = []
    futures = []
    fut_freq_mapping = {}
    if processing == "serial":
        for frequency in tqdm(frequency_samples, desc="Born modeling", total=num_of_frequency_steps, unit="frequency samples"):
            res = born_modeling(source_pos, receiver_pos, frequency, omega_central, density=density, velocity_model=model)
            p_wave_spectrum.append((frequency, res))
    elif processing == "parallel":
        with ProcessPoolExecutor() as process_pool:
            for frequency in frequency_samples:
                future = process_pool.submit(born_modeling, source_pos, receiver_pos, frequency, omega_central, density=density, velocity_model=model)
                futures.append(future)
                fut_freq_mapping[future] = frequency
            for future in tqdm(as_completed(futures), desc="Born modeling", total=num_of_frequency_steps, unit="frequency samples"):
                res = future.result()
                frequency = fut_freq_mapping[future]
                p_wave_spectrum.append((frequency, res))

    p_wave_spectrum = sorted(p_wave_spectrum, key=lambda x: x[0])
    print(p_wave_spectrum)
    freq_domain = np.array([amplitude for freq, amplitude in p_wave_spectrum])
    time_domain = np.fft.ifft(freq_domain)
    # throws ComplexWarning
    plt.plot(time_domain)
    plt.show()


if __name__ == '__main__':
    main()
