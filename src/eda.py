# src/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def basic_statistics(ratings: pd.DataFrame):
    """
    Print basic dataset statistics.
    """
    print("Number of users:", ratings['userId'].nunique())
    print("Number of movies:", ratings['movieId'].nunique())
    print("Total ratings:", len(ratings))
    print("Average rating:", ratings['rating'].mean())
    print("Min rating:", ratings['rating'].min())
    print("Max rating:", ratings['rating'].max())


def plot_rating_distribution(ratings: pd.DataFrame):
    """
    Plot histogram of ratings.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(ratings['rating'], bins=10, kde=False)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()


def create_user_item_matrix(ratings: pd.DataFrame):
    """
    Create user-item matrix using pivot table.
    """
    user_item = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )
    return user_item


def compute_sparsity(user_item_matrix: pd.DataFrame):
    """
    Compute sparsity of user-item matrix.
    """
    total_possible = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    actual_ratings = user_item_matrix.count().sum()

    sparsity = 1 - (actual_ratings / total_possible)

    print("User-Item Matrix Shape:", user_item_matrix.shape)
    print("Sparsity:", sparsity)

    return sparsity


def plot_sparsity_heatmap(user_item_matrix: pd.DataFrame):
    """
    Visualize sparsity pattern (small sample for visualization).
    """
    sample = user_item_matrix.iloc[:100, :100]

    plt.figure(figsize=(8, 6))
    sns.heatmap(sample.isna(), cbar=False)
    plt.title("User-Item Sparsity Visualization (Sample)")
    plt.show()