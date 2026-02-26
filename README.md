# MLOps Movie Recommendation

End-to-end movie recommendation workflow with:

- Feature engineering from MovieLens data
- KNN training and evaluation with MLflow tracking
- FastAPI inference service
- Streamlit UI (poster-based movie selection)
- Airflow incremental retraining DAG
- Prometheus + Grafana monitoring
- Docker Compose orchestration

## Workflow Diagram

![MLOps Movie Recommendation Workflow](assets/workflow.png)

## Project Structure

- `src/data/`: dataset preparation scripts
- `src/features/`: feature generation (`movie_matrix.csv`, `user_matrix.csv`)
- `src/models/`: training, evaluation, prediction, inference API
- `src/visualization/`: Streamlit app
- `dags/`: Airflow DAG for incremental retraining
- `monitoring/`: Prometheus and Grafana provisioning
- `models/`: saved models (`model.pkl`, `base_model.pkl`)
- `data/`: raw and processed datasets
- `tests/`: pytest unit tests

## Local Python Setup

```powershell
python -m venv .env
.\.env\Scripts\activate
pip install -r requirements.txt
```

## Run Core Pipeline (without Docker)

```powershell
python .\src\data\make_dataset.py .\data\raw .\data\processed
python .\src\features\build_features.py
python .\src\models\train_model.py --output-model-path .\models\model.pkl
python .\src\models\evaluate_model.py --model-path .\models\model.pkl --base-model-path .\models\base_model.pkl
```

## MLflow Usage

Start a tracking server (local):

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

In another terminal, point scripts to MLflow:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "movie_reco_knn"
```

Run training and evaluation (both log runs, params, metrics, and artifacts):

```powershell
python .\src\models\train_model.py --n-neighbors 15 --algorithm auto --metric cosine --leaf-size 30 --output-model-path .\models\model.pkl
python .\src\models\evaluate_model.py --model-path .\models\model.pkl --base-model-path .\models\base_model.pkl --top-k 10
```

Open MLflow UI:

- `http://127.0.0.1:5000`
- Compare runs in the experiment table by changing train/eval arguments and rerunning.

## Run with Docker Compose

Start MLflow first (pipeline logs to MLflow):

```powershell
docker compose up -d mlflow
```

Run the pipeline once to generate/update processed features and model files:

```powershell
docker compose --profile pipeline up pipeline
```

Then start the rest of the services:

```powershell
docker compose up -d airflow inference-api streamlit prometheus grafana
```

Services:

- MLflow: `http://127.0.0.1:5000`
- Airflow: `http://127.0.0.1:8080` (default: `admin/admin`)
- FastAPI: `http://127.0.0.1:8000`
- Streamlit: `http://127.0.0.1:8501`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000` (default: `admin/admin`)

## API Quick Test

PowerShell:

```powershell
$body = @{ user_ids = @(1); top_k = 5 } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://127.0.0.1:8000/recommend" -Method Post -ContentType "application/json" -Body $body
```

## Streamlit App Usage

Open:

- `http://127.0.0.1:8501`

Main features:

- **Recommend by Selected Movies** tab:
  - Select posters to build a temporary user profile
  - Click **Recommend from selected movies** to get top-K similar movies
  - Use **Show more posters** to load more candidates
- **Recommend by User IDs** tab:
  - Enter existing dataset user IDs and request API recommendations
  - Generate demo traffic for monitoring dashboards

Sidebar settings:

- Inference API URL
- Model/data paths (`model.pkl`, `movie_matrix.csv`, `movies.csv`, `links.csv`)
- Search filter and number of posters to display
- Optional TMDB posters key: $env:STREAMLIT_TMDB_API_KEY = "your_tmdb_api_key_here" before launching Streamlit

## Monitoring

- FastAPI exposes metrics at `GET /metrics`
- Prometheus scrape config: `monitoring/prometheus/prometheus.yml`
- Grafana datasource provisioning: `monitoring/grafana/provisioning/datasources/datasource.yml`

### Prometheus

Open:

- `http://127.0.0.1:9090`

How it works:

- Prometheus scrapes FastAPI metrics endpoint (`/metrics`) from `inference-api`
- You can query metrics directly in Prometheus expression browser
- Example useful metrics: request count, latency buckets, error count

### Grafana

Open:

- `http://127.0.0.1:3000` (default `admin/admin`)

How to use:

- Data source is provisioned automatically from `monitoring/grafana/provisioning/datasources/datasource.yml`
- Create dashboards/panels with PromQL queries from Prometheus
- Suggested panels:
  - API request rate
  - P95/P99 latency
  - Error ratio
  - Total recommendations served

## Airflow Incremental Retraining

The DAG in `dags/daily_incremental_training_dag.py` supports incremental updates:

- Uses 50% of ratings as initial training data
- Splits remaining 50% into 5 bulks
- Appends one bulk per run and retrains/evaluates

For testing, schedule can run every 2 minutes; for production, switch to daily.

## Run Tests

```powershell
pytest -q
```
