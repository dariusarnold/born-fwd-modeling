from pathlib import Path

from bornfwd.plotting import plot_recording_geometry_3D, plot_recording_geometry, RecordingGeometryInfo
from bornfwd.io import read_sources, read_stations
from marcellus import create_velocity_model


def main(sourcefile: Path, receiverfile: Path):
    vm = create_velocity_model()
    sources = read_sources(sourcefile)
    receivers = read_stations(receiverfile)
    plot_recording_geometry(sources, receivers, vm,
                               RecordingGeometryInfo(num_of_receiver_lines=13, receivers_per_line=50,
                                                     num_of_source_lines=25, sources_per_line=40))


if __name__ == '__main__':
    main("/home/darius/masterarbeit/output_0degrees/sources.txt", "/home/darius/masterarbeit/output_0degrees/receivers.txt")
