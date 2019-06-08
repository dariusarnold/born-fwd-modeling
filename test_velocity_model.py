import unittest
from VelocityModel import VelocityModel, Vector3D
from AlternativeVelocityModel import create_velocity_model, AlternativeVelocityModel


class TestVelocityModel(unittest.TestCase):

    def setUp(self):
        scatterer_positions = [Vector3D(500, 500, 501)]
        self.vel_bg = 4800
        self.vel_frac = 4320
        self.model = VelocityModel(self.vel_bg, self.vel_frac, 1000, 1000, scatterer_positions, 0.5)

    def test_position_on_scatter_point(self):
        scatterer_pos = Vector3D(500, 500, 501)
        self.assertEqual(self.model.eval_at(scatterer_pos), self.vel_frac)

    def test_position_within_scatterer_radius(self):
        close_to_scatterer = Vector3D(500.3, 500, 501)
        self.assertEqual(self.model.eval_at(close_to_scatterer), self.vel_frac)

    def test_position_outside_of_scatterer(self):
        outside_of_scatterer_radius = Vector3D(400, 400, 400)
        self.assertEqual(self.model.eval_at(outside_of_scatterer_radius), self.vel_bg)

    def test_close_but_outside_scatterer_radius(self):
        close_but_not_in = Vector3D(500, 500, 501.50000001)
        self.assertEqual(self.model.eval_at(close_but_not_in), self.vel_bg)

    def test_fracture_depth(self):
        self.assertEqual(self.model.scatterer_bottom, 501.5)
        self.assertEqual(self.model.scatterer_top, 500.5)


class TestAlternativeVelocityModelEasyGrid(unittest.TestCase):
    def setUp(self) -> None:
        self.model = AlternativeVelocityModel(0, 1, 10, 10, 2, 2, 1, 1, 9, 9, 5, 1, 2)

    def test_grid_on_point(self):
        pos = Vector3D(1., 1., 0.)
        gridpoint = self.model.closest_point_on_grid(pos)
        self.assertEqual(Vector3D(1., 1., 5.), gridpoint)

    def test_not_on_point1(self):
        pos = Vector3D(2.2, 1.1, 0.)
        gridpoint = self.model.closest_point_on_grid(pos)
        self.assertEqual(Vector3D(3., 1., 5.), gridpoint)

    def test_not_on_point2(self):
        pos = Vector3D(1.1, 2.5, 0.)
        gridpoint = self.model.closest_point_on_grid(pos)
        self.assertEqual(Vector3D(1., 3., 5.), gridpoint)

    def test_outside_of_grid(self):
        pos = Vector3D(-1, -1, 0)
        gridpoint = self.model.closest_point_on_grid(pos)
        self.assertEqual(Vector3D(-1., -1., 5.,), gridpoint)

    def test_between_points(self):
        pos = Vector3D(2, 2, 0.)
        gridpoint = self.model.closest_point_on_grid(pos)
        self.assertEqual(Vector3D(1., 1., 5.), gridpoint)


class TestAlternativeVelocityModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = create_velocity_model()

    def test_position_on_scatterer_point(self):
        # this is the top left scatterer point
        scatterer_pos = Vector3D(600.0, 9900.0, 2400.0)
        self.assertEqual(self.model.eval_at(scatterer_pos), self.model.frac_vel)

    def test_position_outside_of_heightrange(self):
        pos_under = Vector3D(600, 900, 2501)
        pos_over = Vector3D(600, 900, 2299)
        self.assertEqual(self.model.eval_at(pos_over), self.model.bg_vel)
        self.assertEqual(self.model.eval_at(pos_under), self.model.bg_vel)

    def test_position_near_scatterer_point(self):
        scatterer_pos = Vector3D(600.0, 9901.0, 2400.0)
        self.assertEqual(self.model.eval_at(scatterer_pos), self.model.frac_vel)

    def test_bottom_left_scatterer(self):
        pos = Vector3D(600., 900., 2400.)
        self.assertEqual(self.model.eval_at(pos), self.model.frac_vel)

    def test_scatterer_outside_grid(self):
        outside_left_bottom = Vector3D(580., 400., 2400.)
        outside_right_top = Vector3D(9820., 11000., 2400.)
        self.assertEqual(self.model.eval_at(outside_left_bottom), self.model.bg_vel)
        self.assertEqual(self.model.eval_at(outside_right_top), self.model.bg_vel)


if __name__ == '__main__':
    unittest.main()
