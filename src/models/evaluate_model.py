import argparse
import os
import pickle

import mlflow
import numpy as np
import pandas as pd


def _split_train_test(
    ratings: pd.DataFrame, test_size: float, min_interactions: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["userId", "timestamp"]).copy()
    group_size = ratings.groupby("userId")["movieId"].transform("size")
    ratings = ratings[group_size >= min_interactions].copy()

    ratings["rank"] = ratings.groupby("userId").cumcount()
    ratings["size"] = ratings.groupby("userId")["movieId"].transform("size")
    split_idx = (ratings["size"] * (1.0 - test_size)).astype(int)
    split_idx = split_idx.clip(lower=1, upper=ratings["size"] - 1)
    ratings["split_idx"] = split_idx

    train = ratings[ratings["rank"] < ratings["split_idx"]].copy()
    test = ratings[ratings["rank"] >= ratings["split_idx"]].copy()
    return train, test


def _precision_recall_hit_at_k(
    recommended: list[int], relevant: set[int], k: int
) -> tuple[float, float, float]:
    if not relevant:
        return 0.0, 0.0, 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k).intersection(relevant))
    precision = hits / float(k)
    recall = hits / float(len(relevant))
    hit_rate = 1.0 if hits > 0 else 0.0
    return precision, recall, hit_rate


def _load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_mlflow_metric_name(name: str) -> str:
    # MLflow metric names do not allow '@'
    return name.replace("@", "_at_")


def _recommend_for_users(model, user_rows: pd.DataFrame, movie_matrix: pd.DataFrame) -> np.ndarray:
    feature_cols = [c for c in movie_matrix.columns if c != "movieId" and c in user_rows.columns]
    if hasattr(model, "n_features_in_") and len(feature_cols) != int(model.n_features_in_):
        raise ValueError(
            f"Feature mismatch: model expects {model.n_features_in_} features, got {len(feature_cols)}."
        )

    _, indices = model.kneighbors(user_rows[feature_cols])
    movie_ids = movie_matrix.iloc[indices.reshape(-1)]["movieId"].to_numpy().reshape(indices.shape)
    return movie_ids


def evaluate(args: argparse.Namespace) -> dict:
    ratings = pd.read_csv(args.ratings_path, usecols=["userId", "movieId", "rating", "timestamp"])
    user_matrix = pd.read_csv(args.user_matrix_path)
    movie_matrix = pd.read_csv(args.movie_matrix_path)

    model = _load_model(args.model_path)
    base_model = _load_model(args.base_model_path)

    train, test = _split_train_test(ratings, test_size=args.test_size, min_interactions=args.min_interactions)
    train_seen = train.groupby("userId")["movieId"].agg(set).to_dict()
    test_relevant = (
        test[test["rating"] >= args.relevance_threshold]
        .groupby("userId")["movieId"]
        .agg(set)
        .to_dict()
    )

    available_users = set(user_matrix["userId"].astype(int))
    eval_user_ids = sorted(set(test_relevant).intersection(available_users))
    if args.max_users > 0 and len(eval_user_ids) > args.max_users:
        rng = np.random.default_rng(args.random_seed)
        eval_user_ids = sorted(rng.choice(eval_user_ids, size=args.max_users, replace=False).tolist())

    if not eval_user_ids:
        raise ValueError("No evaluable users found. Try lowering min_interactions or relevance_threshold.")

    user_rows = user_matrix[user_matrix["userId"].isin(eval_user_ids)].copy()
    user_rows = user_rows.set_index("userId").loc[eval_user_ids]
    model_movie_ids = _recommend_for_users(model, user_rows, movie_matrix)
    base_movie_ids = _recommend_for_users(base_model, user_rows, movie_matrix)

    metrics = {
        "model_precision": [],
        "model_recall": [],
        "model_hit_rate": [],
        "base_model_precision": [],
        "base_model_recall": [],
        "base_model_hit_rate": [],
    }

    for row_idx, user_id in enumerate(eval_user_ids):
        relevant = test_relevant.get(user_id, set())
        seen = train_seen.get(user_id, set())
        if not relevant:
            continue

        model_candidates = [int(mid) for mid in model_movie_ids[row_idx] if int(mid) not in seen]
        base_candidates = [int(mid) for mid in base_movie_ids[row_idx] if int(mid) not in seen]

        mp, mr, mh = _precision_recall_hit_at_k(model_candidates, relevant, args.top_k)
        bp, br, bh = _precision_recall_hit_at_k(base_candidates, relevant, args.top_k)

        metrics["model_precision"].append(mp)
        metrics["model_recall"].append(mr)
        metrics["model_hit_rate"].append(mh)
        metrics["base_model_precision"].append(bp)
        metrics["base_model_recall"].append(br)
        metrics["base_model_hit_rate"].append(bh)

    return {
        "users_evaluated": len(metrics["model_precision"]),
        f"model_precision@{args.top_k}": float(np.mean(metrics["model_precision"])),
        f"model_recall@{args.top_k}": float(np.mean(metrics["model_recall"])),
        f"model_hit_rate@{args.top_k}": float(np.mean(metrics["model_hit_rate"])),
        f"base_model_precision@{args.top_k}": float(np.mean(metrics["base_model_precision"])),
        f"base_model_recall@{args.top_k}": float(np.mean(metrics["base_model_recall"])),
        f"base_model_hit_rate@{args.top_k}": float(np.mean(metrics["base_model_hit_rate"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model.pkl against base_model.pkl.")
    parser.add_argument("--ratings-path", default="data/raw/ratings.csv")
    parser.add_argument("--user-matrix-path", default="data/processed/user_matrix.csv")
    parser.add_argument("--movie-matrix-path", default="data/processed/movie_matrix.csv")
    parser.add_argument("--model-path", default="models/model.pkl")
    parser.add_argument("--base-model-path", default="models/base_model.pkl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-interactions", type=int, default=20)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--max-users", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = evaluate(args)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("movie_reco_knn_compare")
    with mlflow.start_run(run_name="eval_model_vs_base_model"):
        mlflow.log_param("ratings_path", args.ratings_path)
        mlflow.log_param("user_matrix_path", args.user_matrix_path)
        mlflow.log_param("movie_matrix_path", args.movie_matrix_path)
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("base_model_path", args.base_model_path)
        mlflow.log_param("top_k", args.top_k)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("min_interactions", args.min_interactions)
        mlflow.log_param("relevance_threshold", args.relevance_threshold)
        mlflow.log_param("max_users", args.max_users)
        mlflow.log_param("random_seed", args.random_seed)

        for key, value in output.items():
            if isinstance(value, float):
                mlflow.log_metric(_to_mlflow_metric_name(key), value)
            else:
                mlflow.log_param(key, value)

    print("Comparison results")
    for key, value in output.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
