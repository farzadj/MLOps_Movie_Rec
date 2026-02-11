import pandas as pd
import pickle
import numpy as np


def make_predictions(users_id, model_filename, user_matrix_filename):
    # Read user_matrix
    users = pd.read_csv(user_matrix_filename)

    # Filter with the list of users_id
    users = users[users["userId"].isin(users_id)]

    # Delete userId
    users = users.drop("userId", axis=1)

    # Open model
    filehandler = open(model_filename, "rb")
    model = pickle.load(filehandler)
    filehandler.close()

    # Calculate nearest neighbors
    _, indices = model.kneighbors(users)

    # Select 10 random numbers from each row
    selection = np.array(
        [np.random.choice(row, size=10, replace=False) for row in indices]
    )

    return selection


if __name__ == "__main__":
    # Take the 5 first users Id of the DB
    users_id = [1, 2, 3, 4, 5]

    # Make predictions using `model.pkl`
    predictions = make_predictions(
        users_id, "models/model.pkl", "data/processed/user_matrix.csv"
    )
    print(predictions)
    movie_matrix = pd.read_csv("data/processed/movie_matrix.csv")
    movies_raw = pd.read_csv("data/raw/movies.csv")[["movieId", "title"]]

    for user_i, recs in enumerate(predictions, start=1):
        movie_ids = movie_matrix.iloc[recs]["movieId"].tolist()
        titles = movies_raw[movies_raw["movieId"].isin(movie_ids)][["movieId", "title"]]
        print(f"\nUser {user_i} recommendations:")
        print(titles.to_string(index=False))

