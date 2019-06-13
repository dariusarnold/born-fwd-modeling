from pathlib import Path

import numpy as np


def create_receivers() -> np.ndarray:
    """
    Create sample receiver geometry use in Hu2018a fig5a, black line
    """
    start_receiver = np.array((5272., 3090., 0.))
    end_receiver = np.array((3090., 5430., 0.))
    num_of_receivers = 100
    receivers_x = np.linspace(start_receiver[0], end_receiver[0], num_of_receivers)
    receivers_y = np.linspace(start_receiver[1], end_receiver[1], num_of_receivers)
    receivers = np.array([(x, y, 0.) for x, y in zip(receivers_x, receivers_y)])
    return receivers


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
