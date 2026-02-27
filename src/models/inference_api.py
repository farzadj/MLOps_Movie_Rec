import os
import pickle
import time
import threading
import subprocess
import sys
from pathlib import Path
import uvicorn
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field


def _to_path(path_str: str) -> Path:
    return Path(path_str).resolve()


MODEL_PATH = _to_path(os.getenv("MODEL_PATH", "models/model.pkl"))
USER_MATRIX_PATH = _to_path(os.getenv("USER_MATRIX_PATH", "data/processed/user_matrix.csv"))
MOVIE_MATRIX_PATH = _to_path(os.getenv("MOVIE_MATRIX_PATH", "data/processed/movie_matrix.csv"))
MOVIES_PATH = _to_path(os.getenv("MOVIES_PATH", "data/raw/movies.csv"))
RATINGS_PATH = _to_path(os.getenv("RATINGS_PATH", "data/raw/ratings.csv"))
TRAIN_SCRIPT_PATH = _to_path(os.getenv("TRAIN_SCRIPT_PATH", "src/models/train_model.py"))
BUILD_FEATURES_SCRIPT_PATH = _to_path(os.getenv("BUILD_FEATURES_SCRIPT_PATH", "src/features/build_features.py"))
TRAIN_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "movie_reco_knn")

AUTH_SECRET_KEY = "local-dev-only-secret"
AUTH_ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 60 * 60 * 8
USERS = {
    "user": {"password": "user123", "role": "user"},
    "admin": {"password": "admin123", "role": "admin"},
}

MODEL = None
USER_MATRIX = None
MOVIE_MATRIX = None
MOVIES_LOOKUP = {}

FEATURE_COLUMNS = []
STATE_LOCK = threading.Lock()

app = FastAPI(title="Movie Recommendation Inference API", version="0.1.0")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

REQUEST_COUNT = Counter(
    "movie_api_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "movie_api_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)


class RecommendRequest(BaseModel):
    user_ids: list[int] = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str


class RetrainResponse(BaseModel):
    status: str
    details: str


class RatingRow(BaseModel):
    userId: int
    movieId: int
    rating: float = Field(ge=0.0, le=5.0)
    timestamp: int | None = None


class AppendRatingsRequest(BaseModel):
    rows: list[RatingRow] = Field(min_length=1)


def load_runtime_objects() -> None:
    global MODEL, USER_MATRIX, MOVIE_MATRIX, MOVIES_LOOKUP, FEATURE_COLUMNS
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    USER_MATRIX = pd.read_csv(USER_MATRIX_PATH)
    MOVIE_MATRIX = pd.read_csv(MOVIE_MATRIX_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)[["movieId", "title"]]
    MOVIES_LOOKUP = dict(zip(movies_df["movieId"], movies_df["title"]))
    FEATURE_COLUMNS = [c for c in USER_MATRIX.columns if c != "userId"]


def create_token(username: str, role: str) -> str:
    now = int(time.time())
    payload = {"sub": username, "role": role, "iat": now, "exp": now + TOKEN_EXPIRE_SECONDS}
    return jwt.encode(payload, AUTH_SECRET_KEY, algorithm=AUTH_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    unauthorized = HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        payload = jwt.decode(token, AUTH_SECRET_KEY, algorithms=[AUTH_ALGORITHM])
        username = str(payload.get("sub", ""))
        role = str(payload.get("role", ""))
        if not username or role not in {"user", "admin"}:
            raise unauthorized
        return {"username": username, "role": role}
    except JWTError:
        raise unauthorized


def require_user_or_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] not in {"user", "admin"}:
        raise HTTPException(status_code=403, detail="Insufficient role")
    return current_user


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return current_user


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    REQUEST_LATENCY.labels(method=method, path=path).observe(latency)
    REQUEST_COUNT.labels(method=method, path=path, status_code=str(response.status_code)).inc()
    return response


@app.on_event("startup")
def startup_load() -> None:
    load_runtime_objects()


@app.post("/auth/token", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    record = USERS.get(form_data.username)
    if not record or form_data.password != record["password"]:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_token(form_data.username, record["role"])
    return TokenResponse(access_token=token, token_type="bearer", role=record["role"])


@app.get("/health")
def health(_: dict = Depends(require_admin)) -> dict:
    return {"status": "ok", "model_path": str(MODEL_PATH)}


@app.get("/metrics")
def metrics(_: dict = Depends(require_admin)):
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend")
def recommend(request: RecommendRequest, _: dict = Depends(require_user_or_admin)) -> dict:
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


@app.post("/admin/retrain", response_model=RetrainResponse)
def retrain_model(_: dict = Depends(require_admin)) -> RetrainResponse:
    cmd = [
        sys.executable,
        str(BUILD_FEATURES_SCRIPT_PATH),
    ]
    build = subprocess.run(cmd, text=True, capture_output=True)
    if build.returncode != 0:
        return RetrainResponse(status="failed", details=f"build_features failed: {build.stderr.strip()}")

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT_PATH),
        "--output-model-path",
        str(MODEL_PATH),
        "--registered-model-name",
        TRAIN_REGISTERED_MODEL_NAME,
    ]
    train = subprocess.run(cmd, text=True, capture_output=True)
    if train.returncode != 0:
        return RetrainResponse(status="failed", details=f"train_model failed: {train.stderr.strip()}")

    with STATE_LOCK:
        load_runtime_objects()
    return RetrainResponse(status="ok", details="Model retrained, reloaded, and registered in MLflow.")


@app.post("/admin/dataset/append-ratings")
def append_ratings(payload: AppendRatingsRequest, _: dict = Depends(require_admin)) -> dict:
    rows = []
    now_ts = int(time.time())
    for row in payload.rows:
        rows.append(
            {
                "userId": int(row.userId),
                "movieId": int(row.movieId),
                "rating": float(row.rating),
                "timestamp": int(row.timestamp if row.timestamp is not None else now_ts),
            }
        )

    RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    header = not RATINGS_PATH.exists()
    new_df.to_csv(RATINGS_PATH, mode="a", index=False, header=header)
    return {"status": "ok", "rows_appended": len(rows), "ratings_path": str(RATINGS_PATH)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
