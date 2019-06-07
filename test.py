import numpy as np
import time

from VelocityModel import Vector3D, create_velocity_model
from main import angular, frequency_samples, born, time_samples, create_header, \
    save_seismogram
from units import Seconds, RadiansPerSecond

"""
This script generates data to recreate fig. 5b) from the paper 3D seismic 
characterization of fractures in a dipping layer using the double-beam method 
by Hao Hu and Yingcai Zheng. The data is generated using the same born modeling 
method as mentioned in the paper. 
"""

#
# parameters
#
start_receiver = Vector3D(5272., 3090., 0.)
end_receiver = Vector3D(3090., 5430., 0.)
num_of_receivers = 100

#
# constant parameters, dont change these
#
source_pos = Vector3D(11200., 5600., 10.)
vm = create_velocity_model()
omega_central: RadiansPerSecond = angular(30.)
length: Seconds = 4
sample_period: Seconds = 0.004
# format id leftpadded with zeros
output_filename = "receiver_{id:03d}.txt"
f_samples = frequency_samples(length, sample_period)


def generate_seismograms_for_receivers():
    receivers_x = np.linspace(start_receiver.x, end_receiver.x, num_of_receivers)
    receivers_y = np.linspace(start_receiver.y, end_receiver.y, num_of_receivers)
    with open("timing.txt", "w", buffering=1) as f:
        for index, (x, y) in enumerate(zip(receivers_x, receivers_y)):
            before = time.time()
            current_receiver = Vector3D(x, y, 0.)
            seismogram = born(source_pos, current_receiver, vm, omega_central,
                              f_samples)
            t_samples = time_samples(length, sample_period)
            header = create_header(current_receiver, source_pos)
            fname = output_filename.format(id=index)
            save_seismogram(seismogram, t_samples, header, fname)
            after = time.time()
            runtime = after - before
            f.write(f"Iteration {index:03d}: {runtime} s\n")



if __name__ == '__main__':
    generate_seismograms_for_receivers()