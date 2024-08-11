import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from numpy_tasks.utils import print_array  # noqa: E402


def generate_array(start: int, end: int, shape: tuple) -> np.ndarray:
    """
    Generate a NumPy array with values ranging from start to end, reshaped to the given shape.

    Parameters:
    start (float): The starting value of the array.
    end (float): The ending value of the array.
    shape (tuple): The desired shape of the array (e.g., (3, 4) for a 3x4 array).

    Returns:
    numpy.ndarray: A NumPy array of the specified shape with values linearly spaced between start and end.
    """
    num_elements = np.prod(shape)
    array = np.linspace(start, end, num=num_elements)
    return array.reshape(shape)


def main():
    # Create a one-dimensional NumPy array with values ranging from 1 to 10.
    array_1d = generate_array(1, 10, (10,))
    print_array(array_1d, "[Task 1.1] One-dimensional array:", precision=0)

    # Create a two-dimensional NumPy array (matrix) with shape (3, 3) containing values from 1 to 9.
    array_2d = generate_array(1, 9, (3, 3))
    print_array(array_2d, "[Task 1.2] Two-dimensional array:", precision=0)

    # Access and print the third element of the one-dimensional array.
    print("[Task 2.1] Third element of the one-dimensional array:", array_1d[2])

    # Slice and print the first two rows and columns of the two-dimensional array.
    print_array(array_2d[:2, :2], "[Task 2.2] First two rows and columns of the two-dimensional array:")

    # Add 5 to each element of the one-dimensional array and print the result.
    updated_array_1d = array_1d + 5
    print_array(updated_array_1d, "[Task 2.3] Updated one-dimensional array:")

    # Multiply each element of the two-dimensional array by 2 and print the result.
    updated_array_2d = array_2d * 2
    print_array(updated_array_2d, "[Task 2.4] Updated two-dimensional array:")


if __name__ == "__main__":
    main()
