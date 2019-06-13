import unittest
from pathlib import Path

from bornfwd.functions import born, frequency_samples
from bornfwd.io import load_seismogram
from main import _setup_parser
import numpy as np


class TestOneShotResult(unittest.TestCase):

    def test_born_output(self):
        parser = _setup_parser()
        with open("test_inputs/options_oneshot.txt", "r") as f:
            args_text = f.read().replace("\n", " ")
        args = parser.parse_args(args_text.split())
        omega_samples = frequency_samples(args.timeseries_length, args.sample_period)
        seismogram = born(args.source_pos, args.receiver_pos, args.model, args.omega_central,
                          omega_samples, quiet=True)
        expected = load_seismogram(Path("test_inputs/output_oneshot.txt"))[1]
        np.testing.assert_allclose(seismogram, expected)
