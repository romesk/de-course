# Task 2: : Data Selection, Filtering, and Aggregation with Pandas

import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import print_dataframe_info, print_divider, print_grouped_data  # noqa: E402


def get_rows_and_columns(data: pd.DataFrame):

    # Select the first row using .iloc
    first_row = data.iloc[0]
    print("First row using .iloc:\n", first_row)
    print()

    # Select the last row using .iloc
    last_row = data.iloc[-1]
    print("Last row using .iloc:\n", last_row)
    print()

    # Select the first column using .iloc
    first_column = data.iloc[:, 0]
    print("First column using .iloc:\n", first_column)
    print()

    # Select the last column using .iloc
    last_column = data.iloc[:, -1]
    print("Last column using .iloc:\n", last_column)
    print()

    # Select the first row using .loc
    first_row_loc = data.loc[0]
    print("First row using .loc:\n", first_row_loc)
    print()

    # Select the last row using .loc
    last_row_loc = data.loc[data.index[-1]]
    print("Last row using .loc:\n", last_row_loc)
    print()

    # Select the first column using .loc
    first_column_loc = data.loc[:, "id"]
    print("First column using .loc:\n", first_column_loc)
    print()

    # Select the last column using .loc
    last_column_loc = data.loc[:, data.columns[-1]]
    print("Last column using .loc:\n", last_column_loc)
    print()


def filter_by_neighbourhood_group(data: pd.DataFrame, neighbourhood_group: str):
    # Filter the data by the specified neighbourhood_group
    filtered_data = data[data["neighbourhood_group"] == neighbourhood_group]
    return filtered_data


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_airbnb_data.csv")

    # Load the data
    data = pd.read_csv(data_path)
    print_dataframe_info(data, "Dataframe NYC:")

    # Use .iloc and .loc to select specific rows and columns based on both position and labels.
    data_copy = data.copy()
    print_divider("[Task 1.1]")
    get_rows_and_columns(data_copy)

    # Filter the data by the specified neighbourhood_group
    print_divider("[Task 1.2]")
    neighbourhood_group = "Brooklyn"
    filtered_data = filter_by_neighbourhood_group(data_copy, neighbourhood_group)
    print_dataframe_info(filtered_data, f"Dataframe NYC filtered by neighbourhood_group: {neighbourhood_group}")

    # Further filter the dataset to include only listings with a price greater than $100 and a number_of_reviews greater than 10.
    print_divider("[Task 1.3]")
    filtered_data = filtered_data[(filtered_data["price"] > 100) & (filtered_data["number_of_reviews"] > 10)]
    print_dataframe_info(filtered_data, "Dataframe NYC filtered by price and number_of_reviews")

    # Select columns of interest such as neighbourhood_group, price, minimum_nights, number_of_reviews, price_category and availability_365 for further analysis.
    print_divider("[Task 1.4]")
    selected_columns = [
        "neighbourhood_group",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "price_category",
        "availability_365",
    ]
    selected_data = filtered_data[selected_columns]
    print_dataframe_info(selected_data, "Selected columns for further analysis:")

    # Group the filtered dataset by neighbourhood_group and price_category to calculate aggregate statistics
    grouped_data = selected_data.groupby(["neighbourhood_group", "price_category"])
    print_grouped_data(grouped_data, "Grouped data by neighbourhood_group and price_category:")

    # avarage price and minimum nights for each group
    print_divider("[Task 2.1]")
    aggregate_stats = grouped_data.agg({"price": np.average, "minimum_nights": np.average})
    print_grouped_data(aggregate_stats, "Aggregate statistics by neighbourhood_group and price_category:")

    # compurt the average number of reviews and availability_365 for each group
    print_divider("[Task 2.2]")
    aggregate_stats = grouped_data.agg({"number_of_reviews": np.average, "availability_365": np.average})
    print_grouped_data(aggregate_stats, "Aggregate statistics by neighbourhood_group and price_category:")

    # data sorting and ranking
    print_divider("[Task 2.3]")
    # Sort the data by price in descending order and by number_of_reviews in ascending order
    sorted_data = selected_data.sort_values(by=["price", "number_of_reviews"], ascending=[False, True])
    print_grouped_data(sorted_data, "Sorted data by price and number_of_reviews:")

    # Create a ranking of neighborhoods based on the total number of listings and the average price
    print_divider("[Task 2.4]")
    neighbourhood_ranking = selected_data.groupby("neighbourhood_group").agg(
        total_listings=("price", "count"), average_price=("price", np.average)
    )
    print_grouped_data(neighbourhood_ranking, "Neighbourhood ranking based on total listings and average price:")

    # save the aggregated data to a new CSV file
    new_path = os.path.join(os.path.dirname(__file__), "..", "data", "aggregated_airbnb_data.csv")
    aggregate_stats.to_csv(new_path)


if __name__ == "__main__":
    main()
