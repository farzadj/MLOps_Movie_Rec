import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


def read_raw_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(input_dir / "ratings.csv")
    movies = pd.read_csv(input_dir / "movies.csv")
    return ratings, movies


def build_processed_tables(ratings: pd.DataFrame, movies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed_ratings_cols = ["userId", "movieId", "rating", "timestamp"]
    needed_movies_cols = ["movieId", "title", "genres"]
    ratings = ratings[needed_ratings_cols].dropna()
    movies = movies[needed_movies_cols].dropna()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    movies["movieId"] = movies["movieId"].astype(int)

    #valid_movie_ids = sorted(set(ratings["movieId"]).intersection(set(movies["movieId"])))
    #movie_id_map = {movie_id: idx for idx, movie_id in enumerate(valid_movie_ids)}

    #ratings = ratings[ratings["movieId"].isin(movie_id_map)].copy()
    #movies = movies[movies["movieId"].isin(movie_id_map)].copy()
    #ratings["movieId"] = ratings["movieId"].map(movie_id_map)
    #movies["movieId"] = movies["movieId"].map(movie_id_map)

    return movies, ratings


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True, path_type=Path))
@click.argument("output_filepath", type=click.Path(path_type=Path))
def main(input_filepath: Path, output_filepath: Path) -> None:
    """Turn raw csv files into processed matrices."""
    logger = logging.getLogger(__name__)
    output_filepath.mkdir(parents=True, exist_ok=True)

    ratings, movies = read_raw_tables(input_filepath)
    movies, ratings = build_processed_tables(ratings, movies)

    movies.to_csv(output_filepath / "movies.csv", index=False)
    ratings.to_csv(output_filepath / "ratings.csv", index=False)

    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()