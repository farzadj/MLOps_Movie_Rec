import os
import pickle
from pathlib import Path
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _to_path(path_str: str) -> Path:
    return Path(path_str).resolve()


MODEL_PATH = _to_path(os.getenv("MODEL_PATH", "models/model.pkl"))
USER_MATRIX_PATH = _to_path(os.getenv("USER_MATRIX_PATH", "data/processed/user_matrix.csv"))
MOVIE_MATRIX_PATH = _to_path(os.getenv("MOVIE_MATRIX_PATH", "data/processed/movie_matrix.csv"))
MOVIES_PATH = _to_path(os.getenv("MOVIES_PATH", "data/raw/movies.csv"))

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

USER_MATRIX = pd.read_csv(USER_MATRIX_PATH)
MOVIE_MATRIX = pd.read_csv(MOVIE_MATRIX_PATH)
MOVIES = pd.read_csv(MOVIES_PATH)[["movieId", "title"]]
MOVIES_LOOKUP = dict(zip(MOVIES["movieId"], MOVIES["title"]))

FEATURE_COLUMNS = [c for c in USER_MATRIX.columns if c != "userId"]

app = FastAPI(title="Movie Recommendation Inference API", version="0.1.0")


class RecommendRequest(BaseModel):
    user_ids: list[int] = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict:
    user_ids = request.user_ids
    top_k = request.top_k

    users = USER_MATRIX[USER_MATRIX["userId"].isin(user_ids)].copy()
    if users.empty:
        raise HTTPException(status_code=404, detail="None of the provided user_ids were found")

    if top_k > MODEL.n_neighbors:
        top_k = MODEL.n_neighbors

    user_features = users[FEATURE_COLUMNS]
    distances, indices = MODEL.kneighbors(user_features)

    response = {"recommendations": []}
    for row_i, user_id in enumerate(users["userId"].tolist()):
        recs = []
        for rank, (movie_idx, distance) in enumerate(zip(indices[row_i][:top_k], distances[row_i][:top_k]), start=1):
            movie_id = int(MOVIE_MATRIX.iloc[movie_idx]["movieId"])
            recs.append(
                {
                    "rank": rank,
                    "movie_index": int(movie_idx),
                    "movie_id": movie_id,
                    "title": MOVIES_LOOKUP.get(movie_id, "Unknown"),
                    "distance": float(distance),
                }
            )
        response["recommendations"].append({"user_id": int(user_id), "items": recs})

    found_ids = set(users["userId"].tolist())
    missing_ids = [uid for uid in user_ids if uid not in found_ids]
    if missing_ids:
        response["missing_user_ids"] = missing_ids

    return response


if __name__ == "__main__":
    uvicorn.run("src.models.inference_api:app", host="0.0.0.0", port=8000, reload=False)
