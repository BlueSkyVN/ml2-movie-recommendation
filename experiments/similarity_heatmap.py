import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings
ratings = pd.read_csv(r"C:\Users\PC\OneDrive\Máy tính\machineLearning2GroupProject\ml2-movie-recommendation\data\ratings.csv")

# Tạo user-item matrix
user_item = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# Lấy sample 20 movie đầu để vẽ cho gọn
sample_movies = user_item.columns[:20]
sample_matrix = user_item[sample_movies]

# Tính similarity
item_similarity = cosine_similarity(sample_matrix.T)

similarity_df = pd.DataFrame(
    item_similarity,
    index=sample_movies,
    columns=sample_movies
)

plt.figure(figsize=(8,6))
sns.heatmap(
    similarity_df,
    cmap="coolwarm",
    square=True
)

plt.title("Item–Item Similarity Matrix (Cosine)")
plt.tight_layout()
plt.show()