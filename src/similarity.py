import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compute_item_similarity(user_item_matrix):
    item_matrix = user_item_matrix.fillna(0).T
    similarity = cosine_similarity(item_matrix)
    return pd.DataFrame(
        similarity,
        index=item_matrix.index,
        columns=item_matrix.index
    )
