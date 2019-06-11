import numpy as np

from VelocityModel import create_velocity_model
from main import angular, frequency_samples
from main import born
from plotting import plot_time_series
from units import *

if __name__ == '__main__':

    xs = np.array((11200.0, 5600.0, 10.0))
    xr = np.array((5272.0, 3090.0, 0.0))
    omega_central = angular(Hertz(30.))

    omega_samples = frequency_samples(4, 0.004)
    vm = create_velocity_model()
    time_domain = born(xs, xr, vm, omega_central, omega_samples)
    plot_time_series(time_domain)
