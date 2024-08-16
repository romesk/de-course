import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import print_dataframe_info, print_divider  # noqa: E402


def main():

    dir_path = os.path.dirname(__file__)
    data_file = "AB_NYC_2019.csv"
    data_path = os.path.join(dir_path, "..", "data", data_file)

    # Load the data
    print_divider("[Task 1.1]")
    data = pd.read_csv(data_path)
    print_dataframe_info(data, "Dataframe NYC:")

    # Inspect the first few rows of the dataset using the head() method to understand its structure.
    print_divider("[Task 1.2]")
    print_dataframe_info(data.head())

    # Basic info
    print_divider("[Task 1.3]")
    print_dataframe_info(data)

    # Identify columns with missing values and count the number of missing entries per column
    print_divider("[Task 2.1]")
    print_dataframe_info(data, "Dataframe NYC all data:")

    # Handle missing values in the name, host_name, and last_review columns
    print_divider("[Task 2.2]")
    data_cleaned = data.copy()
    data_cleaned[["name", "host_name"]] = data_cleaned[["name", "host_name"]].fillna("Unknown")
    data_cleaned["last_review"] = data_cleaned["last_review"].fillna("NaT")
    print_dataframe_info(data_cleaned, "Dataframe NYC cleaned:")

    # Categorize Listings by Price Range: Create a new column price_category that categorizes listings into different price ranges, such as Low, Medium, High
    print_divider("[Task 3.1]")
    data_cleaned["price_category"] = pd.cut(
        data_cleaned["price"],
        bins=[0, 100, 300, data_cleaned["price"].max()],
        labels=["Low", "Medium", "High"],
    )


    # Categorize listings by price range
    print_divider("[Task 3.2]")
    # Create a length_of_stay_category column: Categorize listings based on their minimum_nights into short-term, medium-term, and long-term stays.
    data_cleaned["length_of_stay_category"] = pd.cut(
        data_cleaned["minimum_nights"],
        bins=[0, 3, 14, data_cleaned["minimum_nights"].max()],
        labels=["short-term", "medium-term", "long-term"],
    )
    print_dataframe_info(data_cleaned, "Dataframe NYC categorized by length of stay:")

    # Verify that the data transformations and cleaning steps were successful by reinspecting the DataFrame.
    print_divider("[Task 4.1]")
    print_dataframe_info(data_cleaned, "Dataframe NYC categorized by length of stay:")

    # Ensure that the dataset has no missing values in critical columns (name, host_name, last_review).
    print_divider("[Task 4.2]")

    np.testing.assert_array_equal(data_cleaned["name"].notnull(), True)
    np.testing.assert_array_equal(data_cleaned["host_name"].notnull(), True)
    np.testing.assert_array_equal(data_cleaned["last_review"].notnull(), True)

    # Confirm that all price values are greater than 0. If you find rows with price equal to 0, remove them.
    print_divider("[Task 4.3]")
    data_cleaned = data_cleaned[data_cleaned["price"] > 0]
    np.testing.assert_array_equal(data_cleaned["price"] > 0, True)

    # Save the cleaned dataset to a new CSV file
    new_path = os.path.join(dir_path, "..", "data", "cleaned_airbnb_data.csv")
    data_cleaned.to_csv(new_path, index=False)


if __name__ == "__main__":
    main()
