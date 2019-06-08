import matplotlib.pyplot as plt
import numpy as np

from VelocityModel import VelocityModel


def plot_fractures(velocity_model: VelocityModel) -> None:
    """Plot the fractures (scatterer points) in the VelocityModel"""
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


def plot_time_series(time_series: np.ndarray) -> None:
    """Plot the seismogram generated from Born modeling as a time series"""
    plt.plot(np.real(time_series))
    plt.title("Time series from Born scattering")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()