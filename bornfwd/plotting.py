import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np

from bornfwd.utils import RecordingGeometryInfo
from bornfwd.velocity_model import VelocityModel


def plot_fractures(velocity_model: VelocityModel) -> None:
    """Plot the fractures (scatterer points) in the VelocityModel"""
    x = velocity_model.scatterer_positions.T[0]
    y = velocity_model.scatterer_positions.T[1]
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
    plt.rcParams["font.size"] = 12
    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.scatter(scatterers_x, scatterers_y, marker="_", c="dimgray", s=1, label="Scatterers")
    plt.scatter(sources_x, sources_y, marker="*", color="r", s=2,
                label="Sources")
    plt.scatter(receivers_x, receivers_y, marker="v", color="b", s=2,
                label="Receivers")
    plt.title("Source/Receiver geometry\n\n"
              f"{plot_info.receivers_per_line} receivers/line, {plot_info.num_of_receiver_lines} lines "
              f"= {plot_info.num_of_receiver_lines * plot_info.receivers_per_line} receivers\n"
              f"{plot_info.sources_per_line} sources/line, {plot_info.num_of_source_lines} lines "
              f"= {plot_info.num_of_source_lines * plot_info.sources_per_line} sources")
    legend = plt.legend()
    # change markersize for all legend items
    for item in legend.legendHandles:
        item._sizes = [30]
    plt.xlabel("x axis (West-East, m)")
    plt.ylabel("y axis (South-North, m)")
    plt.axis("equal")

    plt.xlim((0, 11200))
    plt.ylim((0, 11200))
    plt.tight_layout()
    plt.show()


def plot_recording_geometry_3D(sources: np.ndarray, receivers: np.ndarray,
                               velocity_model: VelocityModel, plot_info: RecordingGeometryInfo,
                               three_dimensional: bool = False) -> None:
    """
    Plot recording geometry (positions of sources and receivers)
    :param sources: array of shape (N, 3)
    :param receivers: array of shape (M, 3)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*velocity_model.scatterer_positions.T, marker="_", c="green", s=1, label="Scatterers")
    ax.scatter(*sources.T, marker="*", color="orange", s=2,
               label="Sources")
    ax.scatter(*receivers.T, marker="v", color="b", s=2,
               label="Receivers")
    ax.set_title("Source/Receiver geometry\n\n"
                 f"{plot_info.receivers_per_line} receivers/line, {plot_info.num_of_receiver_lines} lines "
                 f"= {plot_info.num_of_receiver_lines * plot_info.receivers_per_line} receivers\n"
                 f"{plot_info.sources_per_line} sources/line, {plot_info.num_of_source_lines} lines "
                 f"= {plot_info.num_of_source_lines * plot_info.sources_per_line} sources")
    ax.legend()
    ax.set_xlabel("x axis (West-East, m)")
    ax.set_ylabel("y axis (South-North, m)")
    ax.set_xlim((0, 11200))
    ax.set_ylim((0, 11200))
    ax.invert_zaxis()
    fig.tight_layout()
    plt.show()


def plot_time_series(data: np.ndarray, timesteps: np.ndarray,
                     time_unit: str = "s") -> None:
    """Plot the seismogram generated from Born modeling as a time series"""
    plt.plot(timesteps, np.real(data))
    plt.title("Time series from Born scattering")
    plt.ylabel("Amplitude")
    plt.xlabel(f"t ({time_unit})")
    plt.show()

def make_colormap(seq):
    """
    Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def plot_seismogram_gather(seismograms: np.ndarray) -> None:
    """Plot all seismograms from a shot to recreate fig 5b) from Hu2018a"""
    fig, ax = plt.subplots(figsize=(8,6), dpi=244)

    c = mcolors.ColorConverter().to_rgb
    rvn = make_colormap([c("black"), c("white"), 0.49, c("white"), 0.51, c("white"), c("red")])

    plot = ax.pcolormesh(seismograms[480:,...].T, cmap=rvn, antialiased=True)
    cb = fig.colorbar(plot)
    cb.set_label("Amplitude")
    # invert y axis so origin is in top left
    ax.set_ylim(plt.ylim()[::-1])
    # ugly and overly specific way to limit the plotting to 1.5-3.0 secs
    # this is valid for 4 secs trace length with dt = 0.004 s
    start_time = 1.5
    end_time = 3.0
    total_time = 4
    num_samples = seismograms.shape[1]
    start_sample = int(num_samples*start_time/total_time)
    end_sample =int(num_samples*end_time/total_time)
    ax.set_ylim(ymin=end_sample, ymax=start_sample)
    # label y axis with seconds at the same position and values as the original plot
    ax.set_yticks(np.linspace(start_sample, int(num_samples*(end_time-0.1)/total_time), total_time))
    ax.set_yticklabels([f"{x:.2f}" for x in np.linspace(start_time, end_time-0.1, total_time)])
    ax.set_xlabel("Trace #")
    ax.set_ylabel("Time (s)")
    #plt.show()
    plt.savefig("shot_gather.png", dpi=300)