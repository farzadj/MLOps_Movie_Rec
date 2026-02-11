import importlib
import pickle
import sys

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.neighbors import NearestNeighbors


def _build_test_client(tmp_path, monkeypatch):
    movie_matrix = pd.DataFrame(
        {
            "movieId": [10, 11, 12, 13],
            "Action": [1, 0, 1, 0],
            "Comedy": [0, 1, 1, 0],
        }
    )
    user_matrix = pd.DataFrame(
        {
            "userId": [1, 2],
            "Action": [1.0, 0.0],
            "Comedy": [0.0, 1.0],
        }
    )
    movies = pd.DataFrame(
        {"movieId": [10, 11, 12, 13], "title": ["A", "B", "C", "D"]}
    )

    model = NearestNeighbors(n_neighbors=3, algorithm="auto", metric="cosine").fit(
        movie_matrix.drop(columns=["movieId"])
    )

    model_path = tmp_path / "model.pkl"
    user_matrix_path = tmp_path / "user_matrix.csv"
    movie_matrix_path = tmp_path / "movie_matrix.csv"
    movies_path = tmp_path / "movies.csv"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    user_matrix.to_csv(user_matrix_path, index=False)
    movie_matrix.to_csv(movie_matrix_path, index=False)
    movies.to_csv(movies_path, index=False)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("USER_MATRIX_PATH", str(user_matrix_path))
    monkeypatch.setenv("MOVIE_MATRIX_PATH", str(movie_matrix_path))
    monkeypatch.setenv("MOVIES_PATH", str(movies_path))

    sys.modules.pop("src.models.inference_api", None)
    module = importlib.import_module("src.models.inference_api")
    return TestClient(module.app)


def test_health_endpoint(tmp_path, monkeypatch):
    client = _build_test_client(tmp_path, monkeypatch)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_success_with_missing_users(tmp_path, monkeypatch):
    client = _build_test_client(tmp_path, monkeypatch)
    response = client.post("/recommend", json={"user_ids": [1, 999], "top_k": 10})
    body = response.json()

    assert response.status_code == 200
    assert len(body["recommendations"]) == 1
    assert body["recommendations"][0]["user_id"] == 1
    assert len(body["recommendations"][0]["items"]) == 3  # capped by model.n_neighbors
    assert body["missing_user_ids"] == [999]


def test_recommend_returns_404_for_unknown_users(tmp_path, monkeypatch):
    client = _build_test_client(tmp_path, monkeypatch)
    response = client.post("/recommend", json={"user_ids": [999], "top_k": 5})

    assert response.status_code == 404
    assert response.json()["detail"] == "None of the provided user_ids were found"
