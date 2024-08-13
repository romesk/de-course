import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.arrays import generate_array  # noqa: E402
from src.utils import print_array  # noqa: E402


def transpose_array(arr: np.ndarray) -> np.ndarray:
    return arr.T


def reshape_array(arr: np.ndarray, new_shape: tuple) -> np.ndarray:
    return arr.reshape(new_shape)


def split_array(arr: np.ndarray, axis: int, indices: tuple) -> list[np.ndarray]:
    return np.split(arr, indices, axis=axis)


def combine_arrays(arrays: list[np.ndarray], axis: int) -> np.ndarray:
    return np.concatenate(arrays, axis=axis)


def main():

    min, max = 1, 9999
    rng = np.random.default_rng(41)
    arr = rng.integers(min, max, (6, 6))
    print_array(arr, "[Task 1.1] Random Array:")

    transposed = transpose_array(arr)
    print_array(transposed, "[Task 2.1] Transposed Array:")

    reshaped = reshape_array(arr, (3, 12))
    print_array(reshaped, "[Task 2.2] Reshaped Array:")

    splitted = split_array(arr, axis=1, indices=(1, 3))
    print("[Task 2.3] Split Array:", splitted)

    print_array(combine_arrays(splitted, axis=1), "[Task 2.4] Combined Array:")


if __name__ == "__main__":
    main()
