import unittest
from VelocityModel import VelocityModel, Vector3D


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
        self.assertEqual(self.model.scatterer_max_depth, 501)


if __name__ == '__main__':
    unittest.main()
