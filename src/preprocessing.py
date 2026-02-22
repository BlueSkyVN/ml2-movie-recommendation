import os
import pandas as pd
import numpy as np


# 1. DATA LOADING

def load_ratings():
    """
    Load ratings.csv from dataset directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "..", "data", "ml-latest-small", "ratings.csv")
    return pd.read_csv(path)


def load_movies():
    """
    Load movies.csv from dataset directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "..", "data", "ml-latest-small", "movies.csv")
    return pd.read_csv(path)


# 2. DATA CLEANING

def clean_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates and missing values.
    """
    ratings_df = ratings_df.drop_duplicates()
    ratings_df = ratings_df.dropna()
    return ratings_df


# 3. USER-ITEM MATRIX

def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-item matrix using pivot table.
    """
    return ratings_df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )


# 4. MISSING VALUE HANDLING

def fill_missing_values(user_item_matrix: pd.DataFrame, strategy="none"):
    """
    Handle missing values.

    strategy:
        - "none": keep NaN
        - "zero": fill with 0
        - "mean": fill with user mean
    """

    if strategy == "zero":
        return user_item_matrix.fillna(0)

    elif strategy == "mean":
        return user_item_matrix.apply(
            lambda row: row.fillna(row.mean()),
            axis=1
        )

    return user_item_matrix


# 5. OPTIONAL NORMALIZATION
def mean_centering(user_item_matrix: pd.DataFrame):
    """
    Mean-center ratings (useful for Pearson similarity).
    """
    return user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0)