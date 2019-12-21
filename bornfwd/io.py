import ast
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import re


def read_stations(filepath: Path) -> np.ndarray:
    """
    Read file defining receiver positions.
    :param filepath:
    :return: Array containing all receiver positions with shape (N, 3).
    N is the number of receivers defined in the file (first line), 3 are the
    x, y, z coordinates of the individual receivers.
    """
    stations = np.genfromtxt(filepath, skip_header=1, usecols=(1, 2, 3),
                             dtype=np.float32)
    return stations


def read_sources(filepath: Path) -> np.ndarray:
    """
    Read sources from text file and return (N, 3) array of source coordinates.
    """
    int_regex = r"(\d+)"
    float_regex = r"(?:\d+(?:\.\d*)?|\.\d+)"
    number_of_sources_regex = fr"nsrc\s+=\s+(?P<nsrc>{int_regex})"
    x_regex = fr"xsource\s+=\s+(?P<xsource>{float_regex})"
    y_regex = fr"ysource\s+=\s+(?P<ysource>{float_regex})"
    z_regex = fr"zsource\s+=\s+(?P<zsource>{float_regex})"
    # save name of regex group as key, which can later be used to retrieve the
    # matched value
    regexes = {"nsrc": number_of_sources_regex,
               "xsource": x_regex,
               "ysource": y_regex,
               "zsource": z_regex}

    def match_line(line: str) -> Union[Tuple[str, Union[int, float]],
                                       Tuple[None, None]]:
        """
        Match all regexes against a line and return either the first matching
        one or a tuple(None, None) if no regexes matched the line.
        :return: A tuple of the key to the regex that matched and an interpreted
        value (int or float)
        """
        for key, regex in regexes.items():
            match = re.search(regex, line)
            if match:
                value = ast.literal_eval(match.group(key))
                return key, value
        return None, None

    x_values, y_values, z_values = [], [], []
    nsrc = None
    with open(filepath, "r") as f:
        for line in f:
            key, value = match_line(line)
            if key is None:
                continue
            elif key == "nsrc" and nsrc is None:
                nsrc = value
            elif key == "xsource":
                x_values.append(value)
            elif key == "ysource":
                y_values.append(value)
            elif key == "zsource":
                z_values.append(value)

        sources = np.array([(x, y, z) for x, y, z in
                            zip(x_values, y_values, z_values)], dtype=np.float32)
        if nsrc is None:
            raise ValueError(f"Number of sources (nsrc) not specified in file"
                             f"{filepath}")
        if len(sources) != nsrc:
            raise ValueError(f"Expected {nsrc} stations in file {filepath} but "
                             f"got only {len(sources)}.")
        return sources


def save_seismogram(seismogram: np.ndarray, time_steps: np.ndarray, header: str,
                    filename: str, format_string: str = "%1.3f % e") -> None:
    # transpose stacked arrays to save them as columns instead of rows
    np.savetxt(filename, np.vstack((time_steps, seismogram)).T, header=header,
               fmt=format_string)


def create_header(source_pos: np.ndarray, receiver_pos: np.ndarray) -> str:
    """
    Create header string containing information about the seismogram from the
    arguments used to create it. This information will be saved as a header in
    the seismogram file.
    """
    # reshape so the output has the expected shape for parsing
    source_pos = source_pos.reshape(3,)
    receiver_pos = receiver_pos.reshape(3,)
    h = f"source: {source_pos}\nreceiver: {receiver_pos}"
    return h



def read_header_from_file(filepath: Path) -> List[np.ndarray]:
    """
    Parse header from seismogram file and return source and receiver position.
    :param filepath: Full filename
    :return: Tuple of (source_position, receiver_position)
    """
    with open(filepath, "r") as f:
        # read first 2 lines
        header: List[str] = [next(f).rstrip("\n") for _ in range(2)]
    positions = []
    for line in header:
        # throw away source/receiver, only keep digits
        data = line.split(":")[-1]
        # remove outer square brackets
        data = data[2:-1]
        vec = np.fromstring(data, dtype=float, sep=" ")
        positions.append(vec)
    return positions


def load_seismogram(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load seismogram from file
    :return: two numpy arrays: First is time data, second is seismic data
    """
    times, seismic_data = np.loadtxt(filepath, unpack=True)
    return times, seismic_data


def load_seismograms(seismograms_path: Path, seismogram_filename_template: str)\
        -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Load seismograms from the given path.
    This loads all seismograms from the path and returns them as well as
    additional information.
    :param seismograms_path: Path to seismogram files
    :param seismogram_filename_template: String where the numeric part of the
    filename is replaced by a star. Eg. if the seismograms where saved as
    receiver_001.txt, receiver_002.txt, ..., pass "receiver_*.txt".
    :return Numpy array consisting of all loaded seismograms, numpy array
    containing the common timesteps for all seismograms, a list of receiver
    positions for these seismograms, and the source position.
    """
    seismograms = []
    receiver_positions = []
    seismogram_filenames = seismograms_path.glob(seismogram_filename_template)
    for fname in sorted(seismogram_filenames):
        source_pos, receiver_pos = read_header_from_file(fname)
        receiver_positions.append(receiver_pos)
        time, seismic_data = load_seismogram(fname)
        seismograms.append(seismic_data)
    return np.array(seismograms), time, receiver_positions, source_pos


def save_receiver_file(filepath: Path, receivers: np.ndarray) -> None:
    """
    Save receivers to file with correct formatting so the file can be read
    :param filepath: Where file will be saved
    :param receivers: (N, 3) array of N receiver coordinates
    """
    # header contains number of receiver positions
    header = str(len(receivers))
    # indices start at 1
    indices = np.array(range(1, len(receivers)+1))
    # reshape to the same dimension as receivers so hstack works
    indices = indices.reshape(len(receivers), 1)
    # append indices to the left
    data = np.hstack((indices, receivers))
    # save index as int while coordinates are formatted as float
    format_str = "%d %f %f %f"
    # make comments empty string so header isn't prepended with default #
    np.savetxt(str(filepath), data, fmt=format_str, header=header, comments=" ")


def save_source_file(filepath: Path, sources: np.ndarray) -> None:
    """
    Save source locations to file with correct formatting.
    :param filepath: Where file will be saved
    :param sources: (N, 3) array of N source coordinates
    :return:
    """
    with filepath.open("w") as f:
        f.write(f"nsrc = {len(sources)}\n")
        for source_index, source_position in enumerate(sources):
            f.write(f"source = {source_index + 1}\n"
                    f"xsource = {source_position[0]}\n"
                    f"ysource = {source_position[1]}\n"
                    f"zsource = {source_position[2]}\n\n")
