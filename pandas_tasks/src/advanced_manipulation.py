# Task 3: Advanced Data Manipulation, Descriptive Statistics, and Time Series Analysis with Pandas

import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import print_analysis_results, print_dataframe_info, print_divider, print_grouped_data  # noqa: E402


def main():
    """
    Use the pivot_table function to create a detailed summary that reveals the
    average price for different combinations of neighbourhood_group and
    room_type. This analysis will help identify high-demand areas and optimize
    pricing strategies across various types of accommodations (e.g., Entire
    home/apt vs. Private room).
    """

    # Load the data

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_airbnb_data.csv")
    data = pd.read_csv(data_path)

    # Create a pivot table to summarize the average price for different combinations of neighbourhood_group and room_type
    print_divider("[Task 1.1]")
    pivot_table = data.pivot_table(
        index="neighbourhood_group",
        columns="room_type",
        values="price",
        aggfunc=np.mean,
    )
    print_analysis_results(pivot_table, "Average Price by Neighbourhood Group and Room Type:")

    # Melt the dataset to transform it from wide to long format
    print_divider("[Task 1.2]")
    melted_data = data.melt(
        id_vars=["id", "name", "host_id", "host_name", "neighbourhood_group", "room_type"],
        value_vars=["price", "minimum_nights"],
        var_name="metric",
        value_name="value",
    )

    print_dataframe_info(melted_data, "Melted Data:")

    print_divider("[Task 1.3]")
    data["availability_status"] = data["availability_365"].apply(
        lambda x: "Rarely Available" if x < 50 else "Occasionally Available" if 50 <= x < 200 else "Highly Available"
    )
    print_analysis_results(data["availability_status"].value_counts(), "Availability Status Categories:")

    # Group the data by availability_status and calculate the mean price, number_of_reviews, and minimum_nights
    print_divider("[Task 1.4]")
    grouped_data = data.groupby(["availability_status", "neighbourhood_group"]).agg(
        {
            "price": np.mean,
            "number_of_reviews": np.mean,
            "minimum_nights": np.mean,
        }
    )
    print_grouped_data(grouped_data, "Aggregate Statistics by Availability Status:")

    # Desciptive Statistics
    print_divider("[Task 2.1]")
    descriptive_stats = data[["price", "minimum_nights", "number_of_reviews"]].describe()
    print_analysis_results(descriptive_stats, "Descriptive Statistics:")

    # Time Series Analysis
    print_divider("[Task 3.1]")

    # convert last_review to datetime and set it as the index
    data["last_review"] = pd.to_datetime(data["last_review"])
    data.set_index("last_review", inplace=True)

    # Resample the data to obsetve monthly trends in the number_of_reviews and average prices
    print_divider("[Task 3.2]")
    monthly_data = data.resample("M").agg(
        {
            "number_of_reviews": np.sum,
            "price": np.mean,
        }
    )

    print_analysis_results(monthly_data, "Monthly Trends in Number of Reviews and Average Price:")

    # analyze seasonal patterns in the data
    print_divider("[Task 3.3]")

    # group the data by month and calculate monthly averages and analyze seasonal patterns
    monthly_avg_data = data.groupby(data.index.month).agg(
        {
            "number_of_reviews": np.mean,
            "price": np.mean,
        }
    )

    print_analysis_results(monthly_avg_data, "Monthly Averages for Number of Reviews and Price:")

    # save the results to a new CSV file
    new_path = os.path.join(os.path.dirname(data_path), "time_series_airbnb_data.csv")
    monthly_data.to_csv(new_path)


if __name__ == "__main__":
    main()
