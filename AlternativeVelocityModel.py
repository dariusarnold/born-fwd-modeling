from VelocityModel import VelocityModel, Vector3D
from units import MetersPerSecond, Meter, KgPerCubicMeter
from functions import length


class AlternativeVelocityModel(VelocityModel):

    def __init__(self, bg_vel, frac_vel, x_width, y_width, scatterer_x_distance,
                 scatterer_y_distance, scatterer_bot_left_x, scatterer_bot_left_y,
                 scatterer_top_right_x, scatterer_top_right_y,
                 scatterer_depth, scatterer_radius, density):
        self.bg_vel = bg_vel
        self.frac_vel = frac_vel
        self.x_width = x_width
        self.y_width = y_width
        self.scatterer_radius = scatterer_radius
        self.density = density
        self.scatterer_x_distance = scatterer_x_distance
        self.scatterer_y_distance = scatterer_y_distance
        self.scatterer_bot_left_x = scatterer_bot_left_x
        self.scatterer_bot_left_y = scatterer_bot_left_y
        self.scatterer_top_right_x = scatterer_top_right_x
        self.scatterer_top_right_y = scatterer_top_right_y
        self.grid_origin = Vector3D(scatterer_bot_left_x, scatterer_bot_left_y, 0.)
        self.xmin = scatterer_bot_left_x - scatterer_radius
        self.xmax = scatterer_top_right_x + scatterer_radius
        self.ymin = scatterer_bot_left_y - scatterer_radius
        self.ymax = scatterer_top_right_x + scatterer_radius
        self.scatterer_depth = scatterer_depth
        self.scatterer_top = scatterer_depth - scatterer_radius
        self.scatterer_bottom = scatterer_depth + scatterer_radius

    def closest_point_on_grid(self, position: Vector3D) -> Vector3D:
        # use floored division by int to get closest point in grid
        # shift grid origin to x:0, y:0, this will be backshifted later
        position -= self.grid_origin
        correct_x = round(position.x / self.scatterer_x_distance) * self.scatterer_x_distance
        correct_y = round(position.y / self.scatterer_y_distance) * self.scatterer_y_distance
        return Vector3D(correct_x+self.grid_origin.x, correct_y+self.grid_origin.y, self.scatterer_depth)

    def eval_at(self, position: Vector3D) -> MetersPerSecond:
        position = Vector3D(*position)
        within_scatterer_depth = self.scatterer_top <= position.z <= self.scatterer_bottom
        if not within_scatterer_depth:
            return self.bg_vel
        closest_scatterer = self.closest_point_on_grid(position)
        # check if the found closest scatterer even exist, e.g. if it is within the grid boundaries
        if not self.xmin <= closest_scatterer.x <= self.xmax:
            return self.bg_vel
        if not self.ymin <= closest_scatterer.y <= self.ymax:
            return self.bg_vel
        distance = length(closest_scatterer - position)
        if distance > self.scatterer_radius:
            return self.bg_vel
        return self.frac_vel


def create_velocity_model():
    background_vel = MetersPerSecond(4800.)
    fracture_vel = MetersPerSecond(4320.)
    x_width = Meter(11200.)
    y_width = Meter(11200.)
    frac_depth = Meter(2400.)
    # fractures are in a square with sides parallel to the axes
    # values taken from Fig 4 Hu2018a
    frac_bot_left_x = Meter(600.)
    frac_bot_left_y = Meter(900)
    frac_top_right_x = x_width - 1400.
    frac_top_right_y = y_width - 1300.
    frac_spacing = Meter(200.)
    # distance between scatterer points, calculated as 1/8 p wave length
    scatterer_spacing = Meter(20.)
    scatterer_radius = Meter(100.)
    density = KgPerCubicMeter(2550.)
    return AlternativeVelocityModel(background_vel, fracture_vel, x_width, y_width, scatterer_spacing,
                                    frac_spacing, frac_bot_left_x, frac_bot_left_y, frac_top_right_x,
                                    frac_top_right_y, frac_depth, scatterer_radius, density)
