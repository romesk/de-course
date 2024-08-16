import pandas as pd


def print_divider(task_num: str):
    print(f"{task_num} {'=' * 50}")


def print_dataframe_info(df: pd.DataFrame, msg: str = None):
    if msg:
        print(msg)

    print("\nDataFrame Info:")
    print(f"Number of entries: {len(df)}")

    # Display basic DataFrame info
    print("\nBasic Info:")
    df.info()

    # Display missing values count
    print("\nMissing Values:")
    print(df.isnull().sum())

    # descriptive stats
    print("\nDescriptive Statistics:")
    print(df.describe(include="all"))

    # display dataframe
    print("\nDataFrame:")
    print(df)


def print_grouped_data(grouped_data: pd.DataFrame, msg: str = None):
    if msg:
        print(msg)
    print(grouped_data)
