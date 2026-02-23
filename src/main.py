from preprocessing import (
    load_ratings,
    load_movies,
    clean_ratings,
    create_user_item_matrix
)

from similarity import compute_item_similarity
from item_based_cf import ItemBasedCF
from recommender import recommend_movies
from evaluation import evaluate_model, train_test_split_ratings


def main():

    # 1. LOAD DATA
    ratings = load_ratings()
    movies = load_movies()

    # 2. CLEAN DATA
    ratings = clean_ratings(ratings)

    # 3. TRAIN / TEST SPLIT
    train_df, test_df = train_test_split_ratings(ratings)

    # 4. CREATE USER-ITEM MATRIX
    user_item = create_user_item_matrix(train_df)

    # 5. COMPUTE SIMILARITY
    similarity = compute_item_similarity(user_item)

    # 6. BUILD MODEL
    model = ItemBasedCF(train_df, similarity)

    # 7. EVALUATION
    print("Evaluation Results:")
    print(evaluate_model(test_df, model.predict_rating))

    # 8. RECOMMENDATION DEMO
    try:
        user_id = int(input("\nType user ID to see movies recommendation: "))
        print(f"\nTop 5 recommendations for user {user_id}:")
        print(recommend_movies(user_id, model, train_df, movies))
    except ValueError:
        print("Please enter a valid user ID.")


if __name__ == "__main__":
    main()