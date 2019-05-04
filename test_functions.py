import math
import unittest


from functions import length
from VelocityModel import Vector3D


class TestLength(unittest.TestCase):

    def test_length_zero(self):
        v = Vector3D(0, 0, 0)
        self.assertEqual(length(v), 0)

    def test_unit_vectors_length1(self):
        vx = Vector3D(1, 0, 0)
        vy = Vector3D(0, 1, 0)
        vz = Vector3D(0, 0, 1)
        for v in (vx, vy, vz):
            with self.subTest(v=v):
                self.assertEqual(length(v), 1)

    def test_length(self):
        v = Vector3D(1, 1, 1)
        self.assertEqual(length(v), math.sqrt(3))

    def test_length_negative(self):
        v = Vector3D(-1, 0, -1)
        self.assertEqual(length(v), math.sqrt(2))
