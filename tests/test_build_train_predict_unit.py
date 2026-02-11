import pickle

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import create_user_matrix, read_movies, read_ratings
from src.models.predict_model import make_predictions
from src.models.train_model import train_model


def test_read_ratings_and_read_movies(tmp_path):
    ratings_df = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 11, 10],
            "rating": [4.0, 5.0, 3.5],
            "timestamp": [100, 200, 300],
        }
    )
    movies_df = pd.DataFrame(
        {
            "movieId": [10, 11],
            "title": ["Movie A", "Movie B"],
            "genres": ["Action|Comedy", "Drama"],
        }
    )
    ratings_df.to_csv(tmp_path / "ratings.csv", index=False)
    movies_df.to_csv(tmp_path / "movies.csv", index=False)

    loaded_ratings = read_ratings("ratings.csv", data_dir=str(tmp_path))
    loaded_movies = read_movies("movies.csv", data_dir=str(tmp_path))

    assert list(loaded_ratings.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert {"movieId", "title", "Action", "Comedy", "Drama"}.issubset(set(loaded_movies.columns))
    assert loaded_movies.loc[loaded_movies["movieId"] == 10, "Action"].iloc[0] == 1


def test_create_user_matrix_aggregates_genre_means():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 11, 10],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": [1, 2, 3],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 11],
            "title": ["A", "B"],
            "Action": [1, 0],
            "Comedy": [0, 1],
        }
    )

    user_matrix = create_user_matrix(ratings, movies)

    assert list(user_matrix.index) == [1, 2]
    assert list(user_matrix.columns) == ["Action", "Comedy"]
    assert user_matrix.loc[1, "Action"] == 0.5
    assert user_matrix.loc[1, "Comedy"] == 0.5
    assert user_matrix.loc[2, "Action"] == 1.0


def test_train_model_and_predict_shape(tmp_path):
    rng = np.random.default_rng(0)

    movie_matrix = pd.DataFrame(
        {
            "movieId": np.arange(100, 120),
            "f1": rng.random(20),
            "f2": rng.random(20),
            "f3": rng.random(20),
        }
    )
    model = train_model(movie_matrix)
    assert hasattr(model, "kneighbors")

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    user_matrix = pd.DataFrame(
        {
            "userId": [1, 2],
            "f1": [0.1, 0.9],
            "f2": [0.2, 0.8],
            "f3": [0.3, 0.7],
        }
    )
    user_matrix_path = tmp_path / "user_matrix.csv"
    user_matrix.to_csv(user_matrix_path, index=False)

    preds = make_predictions([1, 2], str(model_path), str(user_matrix_path))
    assert preds.shape == (2, 10)
    assert preds.min() >= 0
    assert preds.max() < len(movie_matrix)


def test_predict_raises_for_unknown_users(tmp_path):
    movie_matrix = pd.DataFrame(
        {
            "movieId": np.arange(100, 120),
            "f1": np.linspace(0, 1, 20),
            "f2": np.linspace(1, 0, 20),
        }
    )
    model = train_model(movie_matrix)
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    user_matrix = pd.DataFrame({"userId": [1], "f1": [0.2], "f2": [0.8]})
    user_matrix_path = tmp_path / "user_matrix.csv"
    user_matrix.to_csv(user_matrix_path, index=False)

    with pytest.raises(ValueError):
        make_predictions([999], str(model_path), str(user_matrix_path))
