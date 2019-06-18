import numpy as np

from bornfwd.io import save_receiver_file


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


if __name__ == '__main__':
    receivers = create_receivers()
    save_receiver_file("receivers_fig5.txt", receivers)
