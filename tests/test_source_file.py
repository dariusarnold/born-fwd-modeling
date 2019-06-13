import numpy as np
import unittest
from pathlib import Path

from bornfwd.io import read_sources


class TestReadingSourceFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filepath = Path("test_inputs/source")
        cls.sources = read_sources(cls.filepath)
        # this is data contained in the file
        cls.number_of_sources = 3
        cls.source_coordinates = np.array([[25.408407, 21.647630, 60.],
                                           [24.434352, 21.811987, 60.],
                                           [24.992942, 20.803909, 60.]])

    def test_number_of_sources_read(self):
        self.assertEqual(len(self.sources), self.number_of_sources,
                         msg="Got wrong number of sources from file.")

    def test_source_coordinates(self):
        for source_read, source_expected in zip(self.sources, self.source_coordinates):
            with self.subTest(source_read=source_read):
                np.testing.assert_allclose(source_read, source_expected,
                    err_msg="Wrong source coordinates read.")
