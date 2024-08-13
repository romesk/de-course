# Practical Task 4: Comprehensive Data Handling and Analysis with NumPy

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import print_array  # noqa: E402


def save_array(
    arr: np.ndarray,
    filename: str,
    save_txt: bool = True,
    save_npy: bool = True,
    save_csv: bool = False,
    save_npz: bool = False,
):
    """
    Save a NumPy array to a file in multiple formats.

    Parameters:
    arr (numpy.ndarray): The NumPy array to save.
    filename (str): The name of the file to save the array to (without extension).
    save_txt (bool): Whether to save the array as a text file (default is True).
    save_npy (bool): Whether to save the array as a NumPy binary file (default is True).
    save_csv (bool): Whether to save the array as a CSV file (default is False).
    save_npz (bool): Whether to save the array as a NumPy compressed archive (default is False).
    """

    generated_files = []

    # create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    filename = os.path.join("output", filename)

    if save_txt:
        np.savetxt(f"{filename}.txt", arr, fmt="%d")
        generated_files.append(f"{filename}.txt")
        print(f"Array saved as {filename}.txt")

    if save_npy:
        np.save(f"{filename}.npy", arr)
        generated_files.append(f"{filename}.npy")
        print(f"Array saved as {filename}.npy")

    if save_csv:
        np.savetxt(f"{filename}.csv", arr, delimiter=",", fmt="%d")
        generated_files.append(f"{filename}.csv")
        print(f"Array saved as {filename}.csv")

    if save_npz:
        np.savez(f"{filename}.npz", array=arr)
        generated_files.append(f"{filename}.npz")
        print(f"Array saved as {filename}.npz")

    return generated_files


def load_array(filename: str) -> np.ndarray:
    """
    Load a NumPy array from a file of any format (supported .txt, .csv, .npy, .npz).
    """

    print(f"Loading array from {filename}")

    if filename.endswith(".txt"):
        return np.loadtxt(filename)
    elif filename.endswith(".csv"):
        return np.loadtxt(filename, delimiter=",")
    elif filename.endswith(".npy"):
        return np.load(filename)
    elif filename.endswith(".npz"):
        return np.load(filename)["array"]
    else:
        raise ValueError("Unsupported file format.")


def calculate_sumation(arr: np.ndarray, axis=None) -> np.floating | np.ndarray:
    """Calculate the sum of all elements in the array."""
    sum = np.sum(arr, axis=axis)
    return sum.round(2)


def calculate_mean(arr: np.ndarray, axis=None) -> np.floating | np.ndarray:
    """Calculate the mean of all elements in the array."""
    mean = np.mean(arr, axis=axis)
    return mean.round(2)


def calculate_median(arr: np.ndarray, axis=None) -> np.floating | np.ndarray:
    """Calculate the median of all elements in the array."""
    median = np.median(arr, axis=axis)
    return median.round(2)


def calculate_standard_deviation(arr: np.ndarray, axis=None) -> np.floating | np.ndarray:
    """Calculate the standard deviation of all elements in the array."""
    std = np.std(arr, axis=axis)
    return std.round(2)


def main():
    min, max = 1, 9999
    rng = np.random.default_rng(41)
    arr = rng.integers(min, max, (10, 10))
    print_array(arr, "[Task 1.1] Random Array:")

    # save the array in multiple formats
    print("[Task 2.1] Saving Array in Multiple Formats:")
    generated = save_array(arr, "random_array", save_csv=True, save_npz=True)

    # read from random file
    rand_int = np.random.randint(0, len(generated))
    read_array = load_array(generated[rand_int])
    print_array(read_array, f"[Task 2.2] Loaded Array from {generated[rand_int]}:")

    # calculate statistics
    sumation = calculate_sumation(arr)
    print(f"[Task 3.1] Sum of all elements in the array: {sumation})")

    mean = calculate_mean(arr)
    print(f"[Task 3.2] Mean of all elements in the array: {mean}")

    median = calculate_median(arr)
    print(f"[Task 3.3] Median of all elements in the array: {median}")

    std_dev = calculate_standard_deviation(arr)
    print(f"[Task 3.4] Standard deviation of all elements in the array: {std_dev}")

    # calculate statistics along the rows
    sumation_rows = calculate_sumation(arr, axis=1)
    print_array(sumation_rows, "[Task 3.5] Sum of elements along the rows:")

    mean_rows = calculate_mean(arr, axis=1)
    print_array(mean_rows, "[Task 3.6] Mean of elements along the rows:")

    # calculate statistics along the columns
    median_cols = calculate_median(arr, axis=0)
    print_array(median_cols, "[Task 3.7] Median of elements along the columns:")

    std_dev_cols = calculate_standard_deviation(arr, axis=0)
    print_array(std_dev_cols, "[Task 3.8] Standard deviation of elements along the columns:")

    print_array(arr, "Array after manipulations:")


if __name__ == "__main__":
    main()
