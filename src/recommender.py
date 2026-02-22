def recommend_movies(user_id, model, ratings_df, movies_df, top_n=5):
    rated_movies = ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
    all_movies = movies_df["movieId"].tolist()

    candidates = [m for m in all_movies if m not in rated_movies]

    predictions = []
    for movie_id in candidates:
        rating = model.predict_rating(user_id, movie_id)
        predictions.append((movie_id, rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:top_n]

    return movies_df[movies_df["movieId"].isin([m for m, _ in top_movies])]
