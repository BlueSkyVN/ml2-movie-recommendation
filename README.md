# 🎬 Movie Recommendation System

Machine Learning II – Group Project

---

## Overview

This project implements an **Item-Based Collaborative Filtering (IBCF)** model using the MovieLens dataset.
The system predicts user preferences and generates personalized movie recommendations based on historical ratings.

**Input:** user_id, movie_id, rating
**Output:** Top-N recommended movies

---

## Dataset

We use the **MovieLens Latest Small Dataset**.

* 610 users
* 9,724 movies
* 100,836 ratings
* Rating scale: 0.5 – 5.0
* Average rating: 3.502

Files used:

* ratings.csv
* movies.csv

---

## Method

The recommendation pipeline includes:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing
3. User–Item matrix construction
4. Item–Item similarity computation (Cosine Similarity)
5. Rating prediction
6. Top-N recommendation generation
7. Evaluation (RMSE, MAE)

---

## Installation

### 1. Create virtual environment (Python 3.10 recommended)

```
py -3.10 -m venv ml_env
ml_env\Scripts\activate
```

### 2. Install dependencies

```
pip install numpy pandas matplotlib seaborn scikit-learn scikit-surprise
```

---

## Run the Project

Place the dataset inside:

```
data/ml-latest-small/
```

Run:

```
python src/recommend.py
```

Example usage:

```
recommend_movies(user_id=1, top_n=10)
```

---

## Evaluation

We evaluate the model using:

* RMSE
* MAE

Baseline comparison is included for performance reference.

---

## Git Workflow

* `main` → Final stable version
* `dev` → Integration branch
* `feature/*` → Individual tasks

Commit format:

```
Verb + Object + Optional Detail
```

Example:

```
Implement item-based collaborative filtering
Fix similarity normalization bug
```

---

## Team

* SV1 – Introduction & Dataset
* SV2 – EDA
* SV3 – Preprocessing
* SV4 – Item-Based CF
* SV5 – Integration & Evaluation
* SV6 – Literature Review

---

## Future Improvements

* Matrix Factorization (SVD, ALS)
* Hyperparameter tuning
* Hybrid recommendation system
* Web deployment

---

This project demonstrates a complete end-to-end implementation of Item-Based Collaborative Filtering with structured team collaboration and evaluation.
