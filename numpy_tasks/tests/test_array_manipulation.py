import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.array_manipulation import (  # noqa: E402
    transpose_array,
    reshape_array,
    split_array,
    combine_arrays,
)


class TestArrayFunctions(unittest.TestCase):

    def setUp(self):
        """Set up a random array for testing."""
        min_val, max_val = 1, 9999
        rng = np.random.default_rng(41)
        self.array = rng.integers(min_val, max_val, (6, 6))

    def tearDown(self):
        """Automatically validate the array's integrity and dimensions after each test."""
        self.validate_array(self.array)

    def validate_array(self, arr: np.ndarray):
        """Helper function to validate the integrity and dimensions of the array."""
        self.assertIsInstance(arr, np.ndarray, "Output must be a NumPy array.")
        self.assertTrue(np.issubdtype(arr.dtype, np.number), "Array must contain numeric values.")
        self.assertGreater(arr.size, 0, "Array must contain elements.")
        self.assertTrue(all(dim > 0 for dim in arr.shape), "Array dimensions must be greater than zero.")

    def test_transpose_array(self):
        """Test transposing the array."""
        transposed = transpose_array(self.array)
        self.validate_array(transposed)
        expected_shape = (self.array.shape[1], self.array.shape[0])
        self.assertEqual(transposed.shape, expected_shape, "Transposed array shape should match the expected shape.")

    def test_reshape_array(self):
        """Test reshaping the array."""
        reshaped = reshape_array(self.array, (3, 12))
        self.validate_array(reshaped)
        self.assertEqual(reshaped.shape, (3, 12), "Reshaped array shape should be (3, 12).")
        self.assertEqual(reshaped.size, self.array.size, "Reshaped array must have the same number of elements.")

    def test_split_array(self):
        """Test splitting the array."""
        splitted = split_array(self.array, axis=1, indices=(1, 3))
        for split in splitted:
            self.validate_array(split)
        self.assertEqual(len(splitted), 3, "Array should be split into 3 parts.")

    def test_combine_arrays(self):
        """Test combining arrays."""
        splitted = split_array(self.array, axis=1, indices=(1, 3))
        combined = combine_arrays(splitted, axis=1)
        self.validate_array(combined)
        self.assertTrue(np.array_equal(combined, self.array), "Combined array should match the original array.")


if __name__ == "__main__":
    unittest.main()
