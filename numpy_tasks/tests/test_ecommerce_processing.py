import os
import sys
import unittest
import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import print_array
from src.ecommerce_processing import (  # noqa: E402
    generate_transactions,
    get_transactions_in_range,
    calculate_revenue,
    count_unique_users,
    get_most_purchased_product,
    convert_price_to_int,
    get_column_types,
    get_column_quantities,
    exclude_low_quantity_transactions,
    increase_prices,
    compare_half_year_revenue,
    get_transactions_filtered_by,
    get_top_products_transactions_by_revenue,
    TRANSACTIONS_COLUMNS,
    transaction_id_index,
    user_id_index,
    product_id_index,
    quantity_index,
    price_index,
    timestamp_index,
)


class TestTransactionFunctions(unittest.TestCase):

    TRANSACTION_NUM = 10

    def setUp(self):
        """Set up a fixed transactions array for testing."""
        self.transactions = generate_transactions(self.TRANSACTION_NUM, rand_seed=42)
        # Manually set timestamps for easier testing of date range filters
        base_time = datetime.datetime(2024, 1, 1)
        self.transactions[:, timestamp_index] = np.array(
            [int((base_time + datetime.timedelta(days=i * 60)).timestamp()) for i in range(10)], dtype=np.int64
        )

    def tearDown(self):
        """Validate transactions array is correct after each test."""
        self.validate_transactions_array(self.transactions)

    def validate_transactions_array(self, transactions: np.ndarray, allow_less_rows=False):
        """Helper function to validate the integrity and dimensions of the transactions array."""

        self.assertEqual(
            transactions.shape[1], len(TRANSACTIONS_COLUMNS), "Array must have the correct number of columns."
        )
        if not allow_less_rows:
            self.assertEqual(
                transactions.shape[0], self.TRANSACTION_NUM, "Array must have the correct number of rows."
            )
        self.assertTrue(np.issubdtype(transactions.dtype, np.number), "Array must be of numeric type.")
        self.assertTrue(np.all(transactions[:, quantity_index] >= 0), "Quantities must be non-negative.")
        self.assertTrue(np.all(transactions[:, price_index] >= 0), "Prices must be non-negative.")

    def test_generate_transactions(self):
        """Test that transactions are generated with the correct shape and expected columns."""
        self.validate_transactions_array(self.transactions)

    def test_get_transactions_in_range(self):
        """Test filtering transactions within a date range."""
        start_dt = datetime.datetime(2024, 3, 1)
        end_dt = datetime.datetime(2024, 6, 1)
        filtered = get_transactions_in_range(self.transactions, start_dt, end_dt)
        self.assertEqual(filtered.shape[0], 2)  # Expect 2 transactions within this range
        self.validate_transactions_array(filtered, allow_less_rows=True)

    def test_calculate_revenue(self):
        """Test calculation of total revenue."""
        revenue = calculate_revenue(self.transactions)
        expected_revenue = 3471.52
        self.assertAlmostEqual(revenue, expected_revenue, places=2)

    def test_count_unique_users(self):
        """Test counting the number of unique users."""
        unique_users = count_unique_users(self.transactions)
        self.assertEqual(unique_users, len(np.unique(self.transactions[:, user_id_index])))

    def test_get_most_purchased_product(self):
        """Test identification of the most purchased product."""
        most_purchased_product = get_most_purchased_product(self.transactions)
        expected_product = 102
        self.assertEqual(most_purchased_product, expected_product)

    def test_convert_price_to_int(self):
        """Test conversion of prices to integers."""
        int_prices = convert_price_to_int(self.transactions)
        expected_prices = np.round(self.transactions[:, price_index]).astype(np.int64)
        np.testing.assert_array_equal(int_prices, expected_prices)

    def test_get_column_types(self):
        """Test retrieval of data types for each column."""
        column_types = get_column_types(self.transactions)
        expected_types = {col: self.transactions[:, idx].dtype for idx, col in enumerate(TRANSACTIONS_COLUMNS)}
        self.assertEqual(column_types, expected_types)

    def test_get_column_quantities(self):
        """Test calculation of total quantities for a specific column."""
        quantities = get_column_quantities(self.transactions, product_id_index, quantity_index)
        expected_quantities = [[101.0, 5.0], [102.0, 5.0], [103.0, 4.0], [104.0, 5.0]]
        np.testing.assert_array_almost_equal(quantities, expected_quantities)

    def test_exclude_low_quantity_transactions(self):
        """Test filtering out transactions with low quantities."""
        filtered_transactions = exclude_low_quantity_transactions(self.transactions, min_quantity=1)
        self.assertTrue(np.all(filtered_transactions[:, quantity_index] >= 1))
        self.validate_transactions_array(self.transactions, allow_less_rows=True)

    def test_increase_prices(self):
        """Test increasing prices by a percentage."""
        increased_transactions = increase_prices(self.transactions.copy(), 10)
        expected_prices = np.round(self.transactions[:, price_index] * 1.10, 2)
        np.testing.assert_array_almost_equal(increased_transactions[:, price_index], expected_prices)

    def test_compare_half_year_revenue(self):
        """Test comparing half-year revenue (This function prints, so we'll use unittest.mock to capture output)."""
        from unittest.mock import patch

        with patch("builtins.print") as mock_print:
            compare_half_year_revenue(self.transactions)
            mock_print.assert_called()  # Ensure the function prints something

    def test_get_transactions_filtered_by(self):
        """Test filtering transactions by a specific column value."""
        filtered_transactions = get_transactions_filtered_by(
            self.transactions, user_id_index, self.transactions[0, user_id_index]
        )
        self.assertTrue(np.all(filtered_transactions[:, user_id_index] == self.transactions[0, user_id_index]))
        self.validate_transactions_array(filtered_transactions, allow_less_rows=True)

    def test_get_top_products_transactions_by_revenue(self):
        """Test retrieval of transactions for the top N products by revenue."""
        top_transactions = get_top_products_transactions_by_revenue(self.transactions, 3)
        self.assertLessEqual(len(np.unique(top_transactions[:, product_id_index])), 3)
        self.validate_transactions_array(top_transactions, allow_less_rows=True)


if __name__ == "__main__":
    unittest.main()
