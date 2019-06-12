import matplotlib.pyplot as plt
import numpy as np

from velocity_model import VelocityModel


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


def plot_time_series(data: np.ndarray, timesteps: np.ndarray,
                     time_unit: str = "s") -> None:
    """Plot the seismogram generated from Born modeling as a time series"""
    plt.plot(timesteps, np.real(data))
    plt.title("Time series from Born scattering")
    plt.ylabel("Amplitude")
    plt.xlabel(f"t ({time_unit})")
    plt.show()


def plot_seismogram_gather(seismograms: np.ndarray) -> None:
    """Plot all seismograms from a shot to recreate fig 5b) from Hu2018a"""
    plt.pcolormesh((seismograms.T), cmap="RdGy", antialiased=True)
    cb = plt.colorbar()
    # invert y axis so origin is in top left
    plt.ylim(plt.ylim()[::-1])
    # ugly and overly specific way to limit the plotting to 1.5-3 secs
    # this is valid for 4 secs trace length with dt = 0.004 s
    plt.ylim(ymin=750, ymax=375)
    # label y axis with seconds
    plt.yticks(np.linspace(375, 750, 4), [f"{x:.2f}" for x in np.linspace(1.5, 3, 4)])
    cb.set_label("Amplitude")
    plt.xlabel("Trace #")
    plt.ylabel("Time (s)")
    plt.show()
