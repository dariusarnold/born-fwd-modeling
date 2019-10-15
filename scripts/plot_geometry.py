from bornfwd.plotting import plot_recording_geometry_3D, RecordingGeometryInfo
from bornfwd.io import read_sources, read_stations
from marcellus_multiple_fractures import create_velocity_model


def main():
    vm = create_velocity_model()
    sources = read_sources("../sources.txt")
    receivers = read_stations("../receivers.txt")
    plot_recording_geometry_3D(sources, receivers, vm,
                               RecordingGeometryInfo(num_of_receiver_lines=13, receivers_per_line=50,
                                                     num_of_source_lines=25, sources_per_line=40))


if __name__ == '__main__':
    main()
