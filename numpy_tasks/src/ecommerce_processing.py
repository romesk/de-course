# Practical Task 2: Analyzing and Visualizing E-Commerce Transactions with NumPy

import os
import sys
import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import print_array  # noqa: E402


TRANSACTIONS_COLUMNS = [
    "transaction_id",
    "user_id",
    "product_id",
    "quantity",
    "price",
    "timestamp",
]

transaction_id_index = TRANSACTIONS_COLUMNS.index("transaction_id")
user_id_index = TRANSACTIONS_COLUMNS.index("user_id")
product_id_index = TRANSACTIONS_COLUMNS.index("product_id")
quantity_index = TRANSACTIONS_COLUMNS.index("quantity")
price_index = TRANSACTIONS_COLUMNS.index("price")
timestamp_index = TRANSACTIONS_COLUMNS.index("timestamp")


def generate_transactions(num_transactions: int = 10, rand_seed: int = None) -> np.ndarray:
    """
    Generates a NumPy array simulating e-commerce transactions.

    Parameters:
    num_transactions (int): The number of transactions to generate. Default is 10.

    Returns:
    numpy.ndarray: A multi-dimensional array representing e-commerce transactions.
    """
    np.random.seed(rand_seed)  # For reproducibility, enable to get the same random values each time

    # Generate random data for each field
    transaction_ids = np.arange(1, num_transactions + 1, dtype=np.int64)

    # adding 997 to num_transactions to ensure some user_ids are repeated
    user_ids = np.random.randint(1000, 997 + num_transactions, size=num_transactions, dtype=np.int64)

    # adding 95 to num_transactions to ensure some product_ids are repeated
    product_ids = np.random.randint(100, 95 + num_transactions, size=num_transactions, dtype=np.int64)

    # generate price for each product, ensuring same product id has same price
    product_prices = {product_id: np.round(np.random.uniform(10.0, 500.0), 2) for product_id in np.unique(product_ids)}
    prices = np.array([product_prices[product_id] for product_id in product_ids], dtype=np.float64)

    quantities = np.random.randint(0, 5, size=num_transactions, dtype=np.int64)  # may be zero

    # Generate unix timestamps
    base_time = datetime.datetime(2024, 1, 1)
    timestamps = np.array(
        [
            int((base_time + datetime.timedelta(days=np.random.randint(0, 365))).timestamp())
            for _ in range(num_transactions)
        ],
        dtype=np.int64,
    )

    # Stack all fields into a single multi-dimensional array
    transactions = np.column_stack(
        (transaction_ids, user_ids, product_ids, quantities, prices, timestamps),
    )

    return transactions


def get_transactions_in_range(
    transactions: np.ndarray,
    start_dt: datetime.datetime = datetime.datetime.min,
    end_dt: datetime.datetime = datetime.datetime.max,
) -> np.ndarray:
    """
    Filter transactions based on the timestamp column within a given date range.

    Parameters:
    transactions (numpy.ndarray): The e-commerce transactions array.
    start_dt (datetime.datetime): The start date of the range. Optional, default is datetime.datetime.min.
    end_dt (datetime.datetime): The end date of the range. Optional, default is datetime.datetime.max.

    Returns:
    numpy.ndarray: A filtered array containing transactions within the specified date range.
    """

    if start_dt > end_dt:
        raise ValueError("The start date cannot be after the end date.")

    if start_dt == datetime.datetime.min and end_dt == datetime.datetime.max:
        return transactions

    start_timestamp = start_dt.timestamp()
    end_timestamp = end_dt.timestamp()

    mask = (transactions[:, timestamp_index] >= start_timestamp) & (transactions[:, timestamp_index] <= end_timestamp)
    return transactions[mask]


def calculate_revenue(
    transactions: np.ndarray,
    start_dt: datetime.datetime = datetime.datetime.min,
    end_dt: datetime.datetime = datetime.datetime.max,
) -> float:

    transactions = get_transactions_in_range(transactions, start_dt, end_dt)

    quantities = transactions[:, quantity_index]
    prices = transactions[:, price_index]

    revenue = np.sum(prices * quantities)
    return round(revenue, 2)


def count_unique_users(transactions: np.ndarray) -> int:
    unique = np.unique(transactions[:, user_id_index])
    return len(unique)


def get_most_purchased_product(transactions: np.ndarray) -> int:
    unique, counts = np.unique(transactions[:, product_id_index], return_counts=True)
    return unique[np.argmax(counts)]


def convert_price_to_int(transactions_array: np.ndarray) -> np.ndarray:

    prices = transactions_array[:, price_index]

    # round the prices to the nearest integer
    rounded = np.round(prices)
    return (rounded).astype(np.int64)


def get_column_types(transactions: np.ndarray) -> dict:
    return {
        column_name: transactions[:, column_index].dtype
        for column_index, column_name in enumerate(TRANSACTIONS_COLUMNS)
    }


def get_column_quantities(transactions: np.ndarray, column_index: int, quantities_index: int = -1) -> np.ndarray:
    """
    Calculate the total quantity of each unique value in a given column.

    Parameters:
    transactions (numpy.ndarray): The e-commerce transactions array.
    column_index (int): The index of the column to analyze.
    quantities_index (int): The index of the column containing the quantities that will be summed up.
                            If not passed or set to -1, the function will count the occurrences of each unique value.

    Returns:
    numpy.ndarray: A 2D array containing unique values from the specified column and their total quantities.
    """

    # Extract the relevant columns
    column_data = transactions[:, column_index]

    if quantities_index < 0:
        unique_values, counts = np.unique(column_data, return_counts=True)
        return np.column_stack((unique_values, counts))

    quantities = transactions[:, quantities_index]

    # Get unique values and their corresponding indices
    unique_values, inverse_indices = np.unique(column_data, return_inverse=True)

    # Sum quantities for each unique value using bincount
    total_quantities = np.bincount(inverse_indices, weights=quantities)

    return np.column_stack((unique_values, total_quantities))


def exclude_low_quantity_transactions(transactions: np.ndarray, min_quantity: int = 1) -> np.ndarray:
    """
    Filter out transactions with quantities less than a specified minimum value.

    Parameters:
    transactions (numpy.ndarray): The e-commerce transactions array.
    min_quantity (int): The minimum quantity threshold. Default is 1.

    Returns:
    numpy.ndarray: A filtered array containing transactions with quantities greater than or equal to min_quantity.
    """

    quantities = transactions[:, quantity_index]

    # create a mask for non-zero quantities
    zero_quantity_mask = quantities < min_quantity

    masked_transactions = transactions[~zero_quantity_mask]
    return masked_transactions


def increase_prices(transactions: np.ndarray, increase_percent: float) -> np.ndarray:

    prices = transactions[:, price_index]

    # increase the prices by the given percentage
    increased_prices = prices * (1 + increase_percent / 100)

    # replace the original prices with the increased prices
    transactions[:, price_index] = np.round(increased_prices, 2)
    return transactions


def compare_half_year_revenue(transactions: np.ndarray) -> None:
    """
    Compare the revenue from the first half of the year to the second half and prints the result.

    Parameters:
    transactions (numpy.ndarray): The e-commerce transactions array.

    """

    start_date = datetime.datetime(2024, 1, 1)
    mid_date = datetime.datetime(2024, 7, 1)
    end_date = datetime.datetime(2025, 1, 1)

    first_half_revenue = calculate_revenue(transactions, start_date, mid_date)
    second_half_revenue = calculate_revenue(transactions, mid_date, end_date)

    revenue_msg = (
        f"[Task 4.3] First half revenue: ${first_half_revenue:.2f}, Second half revenue: ${second_half_revenue:.2f}.\n"
    )
    # Compare the revenues
    if first_half_revenue < second_half_revenue:
        msg = f"Revenue increased in the second half of the year for ${second_half_revenue - first_half_revenue:.2f}."
    elif first_half_revenue > second_half_revenue:
        msg = f"Revenue decreased in the second half of the year for ${first_half_revenue - second_half_revenue:.2f}."
    else:
        msg = f"Revenue remained the same in the second half of the year it is ${first_half_revenue:.2f}."

    print(revenue_msg + msg)


def get_transactions_filtered_by(transactions: np.ndarray, column_index: int, value: float) -> np.ndarray:

    column_data = transactions[:, column_index]
    mask = column_data == value  # all values are float type
    return transactions[mask]


def get_top_products_transactions_by_revenue(transactions: np.ndarray, products_N: int = 5) -> np.ndarray:

    prices = transactions[:, price_index]
    quantities = transactions[:, quantity_index]
    revenue_per_transaction = prices * quantities

    # aggregate revenue per product
    unique_products, inverse_indices = np.unique(transactions[:, product_id_index], return_inverse=True)
    total_revenue_per_product = np.bincount(inverse_indices, weights=revenue_per_transaction)

    # get the top N products by revenue
    top_product_indices = np.argsort(total_revenue_per_product)[::-1][:products_N]
    top_products = unique_products[top_product_indices]

    # filter transactions for the top 5 products
    mask = np.isin(transactions[:, product_id_index], top_products)
    top_transactions = transactions[mask]

    return top_transactions


def main():
    # [1] Array creation
    transactions: np.ndarray = generate_transactions()
    print_array(transactions, "[Task 1.1] E-commerce transactions:")

    # [2] Data analysis functions
    print("[Task 2.1] Total revenue:", calculate_revenue(transactions))
    print("[Task 2.2] Number of unique users:", count_unique_users(transactions))
    print("[Task 2.3] Most purchased product ID:", get_most_purchased_product(transactions))
    print_array(convert_price_to_int(transactions), "[Task 2.4] Prices as integers:")
    print("[Task 2.5] Column data types:", get_column_types(transactions))

    # [3] Array manipulation functions
    print_array(get_column_quantities(transactions, product_id_index, quantity_index), "[Task 3.1] Product quantity:")
    print_array(get_column_quantities(transactions, user_id_index), "[Task 3.2] User transaction counts:")
    print_array(exclude_low_quantity_transactions(transactions), "[Task 3.3] Transactions with non-zero quantities:")

    # [4] Arithmetic and Comparison Functions
    print_array(increase_prices(transactions, 10), "[Task 4.1] Transactions with increased prices:")
    print_array(exclude_low_quantity_transactions(transactions, 2), "[Task 4.2] Transactions with quantity > 1:")
    compare_half_year_revenue(transactions)

    # [5] Indexing and Slicing Functions
    print_array(
        get_transactions_filtered_by(transactions, user_id_index, 1001), "[Task 5.1] Transactions for user ID 1001:"
    )
    print_array(
        get_transactions_in_range(transactions, datetime.datetime(2024, 6, 1), datetime.datetime(2024, 8, 1)),
        "[Task 5.2] Transactions in June-July 2024:",
    )
    print_array(
        get_top_products_transactions_by_revenue(transactions, 3),  # to get 5 just change or remove number
        "[Task 5.3] Top 3 products by revenue:",
    )


if __name__ == "__main__":
    main()
