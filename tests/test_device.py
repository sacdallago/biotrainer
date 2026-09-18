import torch
import unittest

from pathlib import Path
from unittest.mock import patch
from biotrainer_core.data_classes import BiotrainerModelResult

from biotrainer.shared import get_device
from biotrainer.training.output_files import InferenceOutputManager


class GetDeviceTests(unittest.TestCase):

    def test_auto_selection_falls_back_to_cpu(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=False):
            self.assertEqual(get_device().type, "cpu")

    def test_available_device_is_honoured(self):
        self.assertEqual(get_device("cpu").type, "cpu")
        self.assertEqual(get_device(torch.device("cpu")).type, "cpu")
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(get_device("cuda:1"), torch.device("cuda:1"))

    def test_unavailable_device_raises_instead_of_falling_back(self):
        """ Silently downgrading to CPU turns a misconfigured GPU run into an hours-long one """
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=False):
            for requested in ["cuda", "cuda:0", torch.device("cuda"), "mps", torch.device("mps")]:
                with self.assertRaises(ValueError):
                    get_device(requested)

    def test_unknown_device_string_raises(self):
        with self.assertRaises(ValueError):
            get_device("not-a-device")


class InferenceOutputManagerDeviceTests(unittest.TestCase):
    """ A stored config records the device the model was *trained* on, so - unlike a device the caller asks for -
        it must not make the model unloadable on a machine that lacks it. """

    _out_file = Path("test_input_files/test_models/rs2c/out.yml")

    def _iom(self, stored_device: str) -> InferenceOutputManager:
        iom = InferenceOutputManager(training_result=BiotrainerModelResult.from_file(self._out_file),
                                     output_file_path=self._out_file)
        iom._input_config["device"] = stored_device
        return iom

    def test_stored_device_is_honoured_when_available(self):
        self.assertEqual(self._iom("cpu").device().type, "cpu")

    def test_unavailable_stored_device_falls_back_to_best_available(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=True):
            self.assertEqual(self._iom("cuda").device().type, "mps")

    def test_unavailable_stored_device_falls_back_to_cpu_without_accelerator(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=False):
            self.assertEqual(self._iom("cuda").device().type, "cpu")


if __name__ == "__main__":
    unittest.main()
