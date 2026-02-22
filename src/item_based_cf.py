import numpy as np


class ItemBasedCF:
    def __init__(self, ratings_df, similarity_matrix):
        self.ratings_df = ratings_df
        self.similarity = similarity_matrix

        # Tạo user-item matrix để truy cập nhanh
        self.user_item_matrix = ratings_df.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        )

        # Tính global mean một lần
        self.global_mean = ratings_df["rating"].mean()

    # ======================================================
    # Predict rating using weighted similarity
    # ======================================================
    def predict_rating(self, user_id, movie_id, k=20):

        # Nếu movie không tồn tại trong similarity
        if movie_id not in self.similarity.columns:
            return self.global_mean

        # Nếu user chưa từng xuất hiện
        if user_id not in self.user_item_matrix.index:
            return self.global_mean

        # Lấy rating của user
        user_ratings = self.user_item_matrix.loc[user_id].dropna()

        if len(user_ratings) == 0:
            return self.global_mean

        # Lấy similarity của movie cần dự đoán
        sim_scores = self.similarity[movie_id]

        # Chỉ giữ movie user đã rated
        sim_scores = sim_scores[user_ratings.index]

        # Lấy top-k similar movies
        sim_scores = sim_scores.sort_values(ascending=False)[:k]

        numerator = np.dot(sim_scores, user_ratings[sim_scores.index])
        denominator = np.sum(np.abs(sim_scores))

        if denominator == 0:
            return self.global_mean

        return numerator / denominator