import matplotlib.pyplot as plt
import numpy as np

from bornfwd.utils import RecordingGeometryInfo
from bornfwd.velocity_model import VelocityModel


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


def plot_recording_geometry(sources: np.ndarray, receivers: np.ndarray,
                            velocity_model: VelocityModel, plot_info: RecordingGeometryInfo) -> None:
    """
    Plot recording geometry (positions of sources and receivers)
    :param sources: array of shape (N, 3)
    :param receivers: array of shape (M, 3)
    """
    sources_x = sources.T[0]
    sources_y = sources.T[1]
    receivers_x = receivers.T[0]
    receivers_y = receivers.T[1]
    scatterers_x = velocity_model.scatterer_positions.T[0]
    scatterers_y = velocity_model.scatterer_positions.T[1]
    plt.scatter(scatterers_x, scatterers_y, marker="_", c="silver", s=1, label="Scatterers")
    plt.scatter(sources_x, sources_y, marker="*", color="orange", s=2,
                label="Sources")
    plt.scatter(receivers_x, receivers_y, marker="v", color="b", s=2,
                label="Receivers")
    plt.title("Source/Receiver geometry\n\n"
              f"{plot_info.receivers_per_line} receivers/line, {plot_info.num_of_receiver_lines} lines "
              f"= {plot_info.num_of_receiver_lines * plot_info.receivers_per_line} receivers\n"
              f"{plot_info.sources_per_line} sources/line, {plot_info.num_of_source_lines} lines "
              f"= {plot_info.num_of_source_lines * plot_info.sources_per_line} sources")
    plt.legend()
    plt.xlabel("x axis (West-East, m)")
    plt.ylabel("y axis (South-North, m)")
    plt.xlim((0, 11200))
    plt.ylim((0, 11200))
    plt.axis("equal")
    plt.tight_layout()
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
