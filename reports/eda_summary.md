## 3. Exploratory Data Analysis (EDA)

### 3.1 Dataset Overview
- Users: 610
- Movies: 9724
- Ratings: 100836
- Rating scale: min 0.5 / max 5.0 / mean ~ 3.502

Observation:
- The dataset contains a moderate number of users and a large catalog of movies.
- Ratings are explicit on a 0.5–5.0 scale, suitable for collaborative filtering.

### 3.2 Rating Distribution
Figure: reports/figures/rating_hist.png

Observation:
- Ratings tend to concentrate around mid-to-high values (typically 3–4), indicating users often rate movies they already like.
- Extremely low ratings are less frequent compared to medium/high ratings.

### 3.3 User Activity (Ratings per User)
Figure: reports/figures/ratings_per_user.png

Observation:
- The number of ratings per user is highly imbalanced (long-tail): most users rate a small number of movies, while a few users rate many.
- This suggests user activity varies widely, which can affect similarity computations.

### 3.4 Movie Popularity (Ratings per Movie)
Figure: reports/figures/ratings_per_movie.png

Observation:
- Movie popularity is also long-tailed: many movies receive only a few ratings, while a small set of movies receives a large number.
- This indicates potential cold-start issues for less-rated movies and popularity bias.

### 3.5 Sparsity of User–Item Matrix
- Sparsity: ~ 98.30% (fill from notebook)
Figure: reports/figures/sparsity_heatmap_sample.png

Observation:
- The user–item rating matrix is extremely sparse, meaning most user-movie pairs are missing ratings.
- This supports the use of collaborative filtering, but also implies item similarity may be noisy when co-rated overlaps are small.

### 3.6 Key Observations for Modeling
- The rating matrix is highly sparse → neighborhood-based CF is appropriate.
- Both user activity and movie popularity show long-tail distributions.
- Many movies have very few ratings → cold-start / unstable similarity for rare items.
- Item-based CF can work well because popular items have sufficient overlap, but should use k-nearest neighbors / shrinkage to reduce noise.
