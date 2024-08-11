import numpy as np


def print_array(arr: np.ndarray, msg: str = "", precision=2) -> None:
    """
    Display the beautified array with a given message.

    Parameters:
    array (numpy.ndarray): The array to be printed.
    message (str, optional): A message to display before the array. Default is None.
    precision (int, optional): Number of decimal places to display for floating-point numbers. Default is 2.
    """

    if msg:
        print(msg)

    # Set print options for NumPy arrays
    np.set_printoptions(precision=precision, suppress=True)

    # Convert array to string with custom formatting
    array_str = np.array2string(
        arr, formatter={"float_kind": lambda x: f"{x: .{precision}f}"}, separator=", "
    )

    # Print the formatted array
    print(array_str)
