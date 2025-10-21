# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import torch
from qlib.contrib.model.pytorch_utils import get_device
class TestMPS(unittest.TestCase):
    def test_get_device(self):
        device = get_device("auto")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.assertEqual(device.type, "mps")
        elif hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
        else:
            self.assertEqual(device.type, "cpu")
if __name__ == "__main__":
    unittest.main()
