import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path("/app")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_INCREMENTAL_DIR = PROJECT_ROOT / "data" / "raw_incremental"
STATE_DIR = PROJECT_ROOT / "data" / "incremental_state"

FULL_RATINGS_PATH = STATE_DIR / "ratings_full.csv"
BASE_RATINGS_PATH = STATE_DIR / "base_ratings.csv"
CURRENT_RATINGS_PATH = STATE_DIR / "current_ratings.csv"
PROGRESS_PATH = STATE_DIR / "progress.json"

STATIC_RAW_FILES = [
    "movies.csv",
    "links.csv",
    "genome-tags.csv",
    "genome-scores.csv",
    "tags.csv",
    "README.txt",
]


def _copy_static_raw_files() -> None:
    RAW_INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in STATIC_RAW_FILES:
        source = RAW_DIR / filename
        if source.exists():
            shutil.copy2(source, RAW_INCREMENTAL_DIR / filename)


def _write_training_ratings_to_raw_incremental() -> None:
    if not CURRENT_RATINGS_PATH.exists():
        raise FileNotFoundError(f"Missing {CURRENT_RATINGS_PATH}")
    _copy_static_raw_files()
    shutil.copy2(CURRENT_RATINGS_PATH, RAW_INCREMENTAL_DIR / "ratings.csv")


def prepare_incremental_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    source_ratings = RAW_DIR / "ratings.csv"
    if not source_ratings.exists():
        raise FileNotFoundError(f"Missing {source_ratings}")

    if not FULL_RATINGS_PATH.exists():
        shutil.copy2(source_ratings, FULL_RATINGS_PATH)

    if BASE_RATINGS_PATH.exists() and CURRENT_RATINGS_PATH.exists() and PROGRESS_PATH.exists():
        _write_training_ratings_to_raw_incremental()
        return

    ratings = pd.read_csv(FULL_RATINGS_PATH)
    if "timestamp" in ratings.columns:
        ratings = ratings.sort_values("timestamp").reset_index(drop=True)

    split_idx = len(ratings) // 2
    base = ratings.iloc[:split_idx].copy()
    remaining = ratings.iloc[split_idx:].copy()
    bulks = np.array_split(remaining, 5)

    base.to_csv(BASE_RATINGS_PATH, index=False)
    base.to_csv(CURRENT_RATINGS_PATH, index=False)

    for idx, chunk in enumerate(bulks, start=1):
        chunk.to_csv(STATE_DIR / f"bulk_{idx}.csv", index=False)

    progress = {"next_bulk": 1}
    PROGRESS_PATH.write_text(json.dumps(progress), encoding="utf-8")
    _write_training_ratings_to_raw_incremental()


def append_next_bulk_daily() -> None:
    if not PROGRESS_PATH.exists():
        raise FileNotFoundError(f"Missing {PROGRESS_PATH}. Run prepare task first.")

    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    next_bulk = int(progress.get("next_bulk", 1))
    if next_bulk <= 5:
        bulk_path = STATE_DIR / f"bulk_{next_bulk}.csv"
        if bulk_path.exists() and bulk_path.stat().st_size > 0:
            current = pd.read_csv(CURRENT_RATINGS_PATH)
            bulk = pd.read_csv(bulk_path)
            updated = pd.concat([current, bulk], ignore_index=True)
            updated.to_csv(CURRENT_RATINGS_PATH, index=False)
        progress["next_bulk"] = next_bulk + 1

    PROGRESS_PATH.write_text(json.dumps(progress), encoding="utf-8")
    _write_training_ratings_to_raw_incremental()


def train_and_evaluate() -> None:
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    env.setdefault("MLFLOW_EXPERIMENT_NAME", "movie_reco_knn")

    subprocess.run(
        ["python", "src/data/make_dataset.py", "data/raw_incremental", "data/processed"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        ["python", "src/features/build_features.py"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        ["python", "src/models/train_model.py", "--output-model-path", "models/model.pkl"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    base_model = PROJECT_ROOT / "models" / "base_model.pkl"
    model = PROJECT_ROOT / "models" / "model.pkl"
    if not base_model.exists() and model.exists():
        shutil.copy2(model, base_model)

    subprocess.run(
        [
            "python",
            "src/models/evaluate_model.py",
            "--ratings-path",
            "data/raw_incremental/ratings.csv",
            "--user-matrix-path",
            "data/processed/user_matrix.csv",
            "--movie-matrix-path",
            "data/processed/movie_matrix.csv",
            "--model-path",
            "models/model.pkl",
            "--base-model-path",
            "models/base_model.pkl",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="daily_incremental_training",
    default_args=default_args,
    description="Append one ratings bulk each run and retrain/evaluate the model.",
    schedule="*/2 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["movie-rec", "training", "incremental"],
) as dag:
    prepare_state = PythonOperator(
        task_id="prepare_incremental_state",
        python_callable=prepare_incremental_state,
    )

    append_bulk = PythonOperator(
        task_id="append_next_bulk_daily",
        python_callable=append_next_bulk_daily,
    )

    retrain = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=train_and_evaluate,
    )

    prepare_state >> append_bulk >> retrain
