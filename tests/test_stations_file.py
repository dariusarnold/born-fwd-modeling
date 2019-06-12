import unittest
from pathlib import Path

import numpy as np

from bornfwd.io import read_stations


class TestReadingStationsFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filepath = Path("test_inputs/stations")
        cls.stations = read_stations(cls.filepath)
        # this is the data contained in the file
        cls.number_of_stations = 5
        cls.dimension_of_point = 3
        cls.expected_dimensions = 2
        cls.expected_stations = np.array([(0., y, 0.) for
                                          y in np.linspace(0, 200, 5)])

    def test_parsing_len(self):
        """
        Test if the correct amount of values is returned
        """
        self.assertEqual(self.number_of_stations, len(self.stations),
                         msg="Read incorrect number of stations from file.")

    def test_parsing_shape(self):
        """
        Test if and array with the correct shape (N, 3) is returned.
        """
        self.assertEqual(len(self.stations.shape), self.expected_dimensions,
                         msg="Got wrong shape of stations array.")

    def test_parsing_receiver_coordinates_dimension(self):
        self.assertEqual(self.stations.shape[-1], self.dimension_of_point,
                         msg="Wrong dimension for receiver coordinates.")


    def test_parsing_stations_values(self):
        """
        Test if the correct values are returned from the function
        """
        stations = read_stations(self.filepath)
        np.testing.assert_allclose(stations, self.expected_stations,
                                   err_msg="Got wrong values from stations file.")
