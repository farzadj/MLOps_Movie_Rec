import os
import pickle
import time
import argparse
import subprocess
import re
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def train_model(movie_matrix, n_neighbors, algorithm, metric, p, leaf_size, n_jobs):
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric,
        p=p,
        leaf_size=leaf_size,
        n_jobs=n_jobs,
    ).fit(movie_matrix.drop("movieId", axis=1))
    return nbrs


def parse_args():
    parser = argparse.ArgumentParser(description="Train KNN model and log runs to MLflow.")
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--algorithm", type=str, default="auto")
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--leaf-size", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--movie-matrix-path", type=str, default="data/processed/movie_matrix.csv")
    parser.add_argument("--output-model-path", type=str, default="models/model.pkl")
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", ""),
        help="Optional MLflow registered model name.",
    )
    return parser.parse_args()


def _run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5, text=True)
        return out.strip()
    except Exception:
        return ""


def get_git_commit() -> str:
    return _run_cmd(["git", "rev-parse", "HEAD"])


def get_dvc_default_remote() -> tuple[str, str]:
    remote_name = _run_cmd(["dvc", "remote", "default"])
    if not remote_name:
        return "", ""

    remote_list = _run_cmd(["dvc", "remote", "list"])
    remote_url = ""
    for line in remote_list.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0] == remote_name:
            remote_url = parts[1]
            break
    return remote_name, remote_url


def get_dvc_data_rev(movie_matrix_path: str) -> str:
    matrix_path = Path(movie_matrix_path).resolve()
    processed_dvc = matrix_path.parent.parent / "processed.dvc"
    if not processed_dvc.exists():
        return ""
    text = processed_dvc.read_text(encoding="utf-8")
    match = re.search(r"md5:\s*([a-f0-9]{32}\.dir)", text)
    return match.group(1) if match else ""


if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "movie_reco_knn"))

    movie_matrix = pd.read_csv(args.movie_matrix_path)
    with mlflow.start_run(run_name="train_knn_genre"):
        git_commit = get_git_commit()
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)

        dvc_data_rev = get_dvc_data_rev(args.movie_matrix_path)
        if dvc_data_rev:
            mlflow.set_tag("dvc_data_rev", dvc_data_rev)

        dvc_remote_name, dvc_remote_url = get_dvc_default_remote()
        if dvc_remote_name:
            mlflow.set_tag("dvc_remote", dvc_remote_name)
        if dvc_remote_url:
            mlflow.set_tag("dvc_remote_url", dvc_remote_url)

        mlflow.log_param("n_neighbors", args.n_neighbors)
        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("metric", args.metric)
        mlflow.log_param("p", args.p)
        mlflow.log_param("leaf_size", args.leaf_size)
        mlflow.log_param("n_jobs", args.n_jobs)
        mlflow.log_param("movie_matrix_path", args.movie_matrix_path)
        mlflow.log_param("output_model_path", args.output_model_path)

        t0 = time.time()
        model = train_model(
            movie_matrix=movie_matrix,
            n_neighbors=args.n_neighbors,
            algorithm=args.algorithm,
            metric=args.metric,
            p=args.p,
            leaf_size=args.leaf_size,
            n_jobs=args.n_jobs,
        )
        mlflow.log_metric("train_time_sec", time.time() - t0)
        mlflow.log_metric("n_movies", float(len(movie_matrix)))
        log_model_kwargs = {"artifact_path": "model"}
        if args.registered_model_name:
            log_model_kwargs["registered_model_name"] = args.registered_model_name
        mlflow.sklearn.log_model(model, **log_model_kwargs)

    with open(args.output_model_path, "wb") as f:
        pickle.dump(model, f)
