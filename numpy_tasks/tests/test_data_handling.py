import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_handling import (  # noqa: E402
    save_array,
    load_array,
    calculate_mean,
    calculate_median,
    calculate_sumation,
    calculate_standard_deviation,
)


class TestArrayFileOperations(unittest.TestCase):

    def setUp(self):
        """Set up a random array and file paths for testing."""
        min_val, max_val = 1, 9999
        rng = np.random.default_rng(41)
        self.array = rng.integers(min_val, max_val, (10, 10))
        self.filename = "test_array"

    def tearDown(self):
        """Automatically validate the array's integrity and dimensions after each test and clean up files."""
        self.validate_array(self.array)
        self.cleanup_files()

    def validate_array(self, arr: np.ndarray):
        """Helper function to validate the integrity and dimensions of the array."""
        self.assertIsInstance(arr, np.ndarray, "Output must be a NumPy array.")
        self.assertTrue(np.issubdtype(arr.dtype, np.number), "Array must contain numeric values.")
        self.assertGreater(arr.size, 0, "Array must contain elements.")
        self.assertTrue(all(dim > 0 for dim in arr.shape), "Array dimensions must be greater than zero.")

    def cleanup_files(self):
        """Helper function to remove generated files after tests."""
        extensions = ["txt", "npy", "csv", "npz"]
        for ext in extensions:
            filepath = os.path.join("output", f"{self.filename}.{ext}")
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_save_array_txt(self):
        """Test saving the array as a text file."""
        generated_files = save_array(
            self.array, self.filename, save_txt=True, save_npy=False, save_csv=False, save_npz=False
        )
        self.assertIn(f"output/{self.filename}.txt", generated_files)
        self.assertTrue(os.path.exists(f"output/{self.filename}.txt"))

    def test_save_array_npy(self):
        """Test saving the array as a NumPy binary file."""
        generated_files = save_array(
            self.array, self.filename, save_txt=False, save_npy=True, save_csv=False, save_npz=False
        )
        self.assertIn(f"output/{self.filename}.npy", generated_files)
        self.assertTrue(os.path.exists(f"output/{self.filename}.npy"))

    def test_save_array_csv(self):
        """Test saving the array as a CSV file."""
        generated_files = save_array(
            self.array, self.filename, save_txt=False, save_npy=False, save_csv=True, save_npz=False
        )
        self.assertIn(f"output/{self.filename}.csv", generated_files)
        self.assertTrue(os.path.exists(f"output/{self.filename}.csv"))

    def test_save_array_npz(self):
        """Test saving the array as a compressed archive."""
        generated_files = save_array(
            self.array, self.filename, save_txt=False, save_npy=False, save_csv=False, save_npz=True
        )
        self.assertIn(f"output/{self.filename}.npz", generated_files)
        self.assertTrue(os.path.exists(f"output/{self.filename}.npz"))

    def test_load_array_txt(self):
        """Test loading the array from a text file."""
        save_array(self.array, self.filename, save_txt=True, save_npy=False, save_csv=False, save_npz=False)
        loaded_array = load_array(f"output/{self.filename}.txt")
        self.validate_array(loaded_array)
        self.assertTrue(np.array_equal(loaded_array, self.array), "Loaded array should match the original array.")

    def test_load_array_npy(self):
        """Test loading the array from a NumPy binary file."""
        save_array(self.array, self.filename, save_txt=False, save_npy=True, save_csv=False, save_npz=False)
        loaded_array = load_array(f"output/{self.filename}.npy")
        self.validate_array(loaded_array)
        self.assertTrue(np.array_equal(loaded_array, self.array), "Loaded array should match the original array.")

    def test_load_array_csv(self):
        """Test loading the array from a CSV file."""
        save_array(self.array, self.filename, save_txt=False, save_npy=False, save_csv=True, save_npz=False)
        loaded_array = load_array(f"output/{self.filename}.csv")
        self.validate_array(loaded_array)
        self.assertTrue(np.array_equal(loaded_array, self.array), "Loaded array should match the original array.")

    def test_load_array_npz(self):
        """Test loading the array from a compressed archive."""
        save_array(self.array, self.filename, save_txt=False, save_npy=False, save_csv=False, save_npz=True)
        loaded_array = load_array(f"output/{self.filename}.npz")
        self.validate_array(loaded_array)
        self.assertTrue(np.array_equal(loaded_array, self.array), "Loaded array should match the original array.")

    def test_calculate_sumation(self):
        """Test calculating the summation of the array."""
        result = calculate_sumation(self.array)
        self.validate_array(np.array(result))
        self.assertEqual(result, np.sum(self.array).round(2))

    def test_calculate_mean(self):
        """Test calculating the mean of the array."""
        result = calculate_mean(self.array)
        self.validate_array(np.array(result))
        self.assertEqual(result, np.mean(self.array).round(2))

    def test_calculate_median(self):
        """Test calculating the median of the array."""
        result = calculate_median(self.array)
        self.validate_array(np.array(result))
        self.assertEqual(result, np.median(self.array).round(2))

    def test_calculate_standard_deviation(self):
        """Test calculating the standard deviation of the array."""
        result = calculate_standard_deviation(self.array)
        self.validate_array(np.array(result))
        self.assertEqual(result, np.std(self.array).round(2))


if __name__ == "__main__":
    unittest.main()
