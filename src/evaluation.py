import os
import pandas as pd

from sklearn.model_selection import train_test_split
from metrics import rmse, mae


# Train/Test split function
def train_test_split_ratings(ratings_df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state
    )
    return train_df, test_df


# Evaluation framework
def evaluate_model(test_df, predict_func):
    """
    test_df: DataFrame gồm [userId, movieId, rating]
    predict_func: hàm dự đoán rating
    """
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        user = row['userId']
        movie = row['movieId']
        true_rating = row['rating']

        pred_rating = predict_func(user, movie)

        y_true.append(true_rating)
        y_pred.append(pred_rating)

    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    }


# Dummy predictor (for testing only)
def dummy_predict(userId, movieId):
    return 3.5


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "ratings.csv")

    ratings = pd.read_csv(DATA_PATH)

    train_df, test_df = train_test_split_ratings(ratings)

    results = evaluate_model(test_df, dummy_predict)

    print("Evaluation with dummy predictor:")
    print(results)
