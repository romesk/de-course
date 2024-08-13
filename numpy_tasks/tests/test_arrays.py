import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.arrays import generate_array  # noqa: E402


class TestGenerateArray(unittest.TestCase):

    def test_generate_array_basic(self):
        """Test basic functionality of generate_array."""
        result = generate_array(1, 10, (10,))
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_array_equal(result, expected)

    def test_generate_array_two_dimensional(self):
        """Test generation of a 2D array."""
        result = generate_array(1, 9, (3, 3))
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(result, expected)

    def test_generate_array_float_values(self):
        """Test generation of an array with float values."""
        result = generate_array(1.5, 3.5, (4,))
        np.testing.assert_equal(result.shape, (4,))

    def test_generate_array_edge_case(self):
        """Test edge case of a single-element array."""
        result = generate_array(1, 1, (1,))
        expected = np.array([1])
        np.testing.assert_array_equal(result, expected)

    def test_generate_array_zero_elements(self):
        """Test edge case with zero elements in the shape."""
        result = generate_array(1, 10, (0,))
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_generate_array_non_integer_elements(self):
        """Test that non-integer shapes raise an error."""
        with self.assertRaises(TypeError):
            generate_array(1, 10, (3.5, 4))  # Non-integer value in shape

    def test_generate_array_negative_shape(self):
        """Test that negative dimensions in shape raise an error."""
        with self.assertRaises(ValueError):
            generate_array(1, 10, (-3, 4))  # Negative dimension in shape


if __name__ == "__main__":
    unittest.main()
