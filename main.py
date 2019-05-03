import itertools

import numpy as np

from VelocityModel import VelocityModel, Vector3D
import matplotlib.pyplot as plt


def create_scatterers():
    model_width = 11200
    # values taken from Fig4 in Hu2018a, all units m
    frac_depth = 2400.
    # fractures are in a square with sides parallel to the axes
    # values taken from Fig 4 Hu2018a
    frac_top_left_x = 600
    frac_top_left_y = model_width - 1300
    frac_bot_right_x = model_width - 1400
    frac_bot_right_y = 860
    frac_spacing = 200
    # distance between scatterer points, calculated as 1/8 p wave length
    scatterer_spacing = 20
    # + 1 to cover the "last" point
    number_of_fractures = int((frac_top_left_y - frac_bot_right_y) / frac_spacing) + 1
    scatterers_per_fracture = int((frac_bot_right_x - frac_top_left_x) / scatterer_spacing) + 1
    # step parallel to y axis
    scatterer_y_positions = np.linspace(frac_bot_right_y, frac_top_left_y, number_of_fractures)
    # step parallel to x axis
    scatterer_x_positions = np.linspace(frac_top_left_x, frac_bot_right_x, scatterers_per_fracture)
    xy_combinations = itertools.product(scatterer_x_positions, scatterer_y_positions)
    scatterers = [Vector3D(x, y, frac_depth) for x, y in xy_combinations]
    return scatterers


def create_velocity_model():
    background_vel = 4800
    fracture_vel = 4320
    x_width = 11200
    y_width = 11200
    scatterer_positions = create_scatterers()
    scatterer_radius = 1
    return VelocityModel(background_vel, fracture_vel, x_width, y_width, scatterer_positions, scatterer_radius)


def plot_fractures(velocity_model: VelocityModel):
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


def main():
    model = create_velocity_model()

    #plot_fractures(model)

    


if __name__ == '__main__':
    main()



