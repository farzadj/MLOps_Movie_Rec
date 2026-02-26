import os
import pickle
import time
import argparse

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "movie_reco_knn"))

    movie_matrix = pd.read_csv(args.movie_matrix_path)
    with mlflow.start_run(run_name="train_knn_genre"):
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
        mlflow.sklearn.log_model(model, artifact_path="model")

    with open(args.output_model_path, "wb") as f:
        pickle.dump(model, f)
