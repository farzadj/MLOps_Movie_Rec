import os
import pickle
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from streamlit_image_select import image_select


DEFAULT_API_URL = os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL_PATH = os.getenv("STREAMLIT_MODEL_PATH", "models/model.pkl")
DEFAULT_MOVIE_MATRIX_PATH = os.getenv("STREAMLIT_MOVIE_MATRIX_PATH", "data/processed/movie_matrix.csv")
DEFAULT_MOVIES_PATH = os.getenv("STREAMLIT_MOVIES_PATH", "data/processed/movies.csv")
DEFAULT_LINKS_PATH = os.getenv("STREAMLIT_LINKS_PATH", "data/raw/links.csv")
TMDB_API_KEY = os.getenv("STREAMLIT_TMDB_API_KEY") or os.getenv("TMDB_API_KEY", "")


def parse_user_ids(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Please provide at least one user ID.")
    return values


def api_headers(token: str) -> dict:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def login_api(api_url: str, username: str, password: str) -> dict:
    resp = requests.post(
        f"{api_url}/auth/token",
        data={"username": username, "password": password},
        timeout=15,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code}: {resp.text}")
    return resp.json()


def call_recommend(api_url: str, token: str, user_ids: list[int], top_k: int) -> dict:
    payload = {"user_ids": user_ids, "top_k": top_k}
    response = requests.post(f"{api_url}/recommend", json=payload, headers=api_headers(token), timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    return response.json()


@st.cache_data(show_spinner=False)
def load_movie_catalog(movie_matrix_path: str, movies_path: str) -> pd.DataFrame:
    movie_matrix = pd.read_csv(movie_matrix_path)
    movies = pd.read_csv(movies_path)
    if "title" not in movies.columns:
        raise ValueError(f"'title' column not found in {movies_path}")
    movie_cols = ["movieId", "title"]
    if "genres" in movies.columns:
        movie_cols.append("genres")
    catalog = movie_matrix[["movieId"]].merge(movies[movie_cols], on="movieId", how="left")
    catalog["title"] = catalog["title"].fillna("Unknown")
    if "genres" not in catalog.columns:
        catalog["genres"] = "(no genres listed)"
    catalog["genres"] = catalog["genres"].fillna("(no genres listed)")
    catalog["label"] = catalog["title"] + " (id=" + catalog["movieId"].astype(str) + ")"
    return catalog


@st.cache_data(show_spinner=False)
def load_links(links_path: str) -> pd.DataFrame:
    links = pd.read_csv(links_path)
    if "movieId" not in links.columns or "tmdbId" not in links.columns:
        raise ValueError(f"movieId/tmdbId columns not found in {links_path}")
    links = links[["movieId", "tmdbId"]].copy()
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    return links


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def recommend_from_selected_movies(
    model,
    movie_matrix_path: str,
    movies_path: str,
    selected_movie_ids: list[int],
    top_k: int,
) -> pd.DataFrame:
    movie_matrix = pd.read_csv(movie_matrix_path)
    catalog = load_movie_catalog(movie_matrix_path, movies_path)

    selected = movie_matrix[movie_matrix["movieId"].isin(selected_movie_ids)]
    if selected.empty:
        raise ValueError("None of the selected movies exist in movie_matrix.csv.")

    # Build a profile vector by averaging selected movies features.
    query_vec = selected.drop(columns=["movieId"]).mean(axis=0).to_numpy(dtype=np.float64).reshape(1, -1)
    n_neighbors = min(len(movie_matrix), max(top_k + len(selected_movie_ids), top_k))
    distances, indices = model.kneighbors(query_vec, n_neighbors=n_neighbors)

    picked = set(selected_movie_ids)
    seen = set()
    rows = []
    for rank_src, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        movie_id = int(movie_matrix.iloc[idx]["movieId"])
        if movie_id in picked or movie_id in seen:
            continue
        seen.add(movie_id)
        title_row = catalog[catalog["movieId"] == movie_id]
        title = title_row.iloc[0]["title"] if not title_row.empty else "Unknown"
        rows.append(
            {
                "rank": len(rows) + 1,
                "movie_id": movie_id,
                "title": title,
                "distance": float(dist),
            }
        )
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def _clean_title(title: str) -> str:
    # MovieLens titles often include year in parentheses, e.g. "Toy Story (1995)".
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def _fetch_tmdb_poster_url(tmdb_id: int) -> str:
    if not TMDB_API_KEY:
        return ""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=8,
        )
        if response.status_code != 200:
            return ""
        data = response.json()
        poster_path = data.get("poster_path", "")
        if not poster_path:
            return ""
        return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def fetch_poster_url(title: str, tmdb_id: int | None = None) -> str:
    fallback = "https://placehold.co/300x450?text=No+Poster"
    if tmdb_id is not None:
        tmdb_url = _fetch_tmdb_poster_url(tmdb_id)
        if tmdb_url:
            return tmdb_url
    # Fallback to iTunes Search API (no key required).
    try:
        query = _clean_title(title)
        response = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "entity": "movie", "limit": 1},
            timeout=8,
        )
        if response.status_code != 200:
            return fallback
        data = response.json().get("results", [])
        if not data:
            return fallback
        url = data[0].get("artworkUrl100", "")
        if not url:
            return fallback
        return url.replace("100x100bb", "300x450bb")
    except Exception:
        return fallback


@st.cache_data(show_spinner=False)
def fetch_poster_image(title: str, tmdb_id: int | None = None) -> Image.Image:
    poster_url = fetch_poster_url(title, tmdb_id=tmdb_id)
    try:
        resp = requests.get(poster_url, timeout=8)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return Image.new("RGB", (300, 450), color=(230, 230, 230))


def add_checkmark_badge(image: Image.Image) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cx, cy = w // 2, h // 2
    r = max(34, min(w, h) // 6)

    # Dim the center area and draw a central checkmark badge.
    draw.ellipse((cx - (r + 18), cy - (r + 18), cx + (r + 18), cy + (r + 18)), fill=(0, 0, 0, 95))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(22, 163, 74, 245))
    p1 = (cx - int(0.45 * r), cy + int(0.02 * r))
    p2 = (cx - int(0.10 * r), cy + int(0.38 * r))
    p3 = (cx + int(0.52 * r), cy - int(0.30 * r))
    draw.line([p1, p2, p3], fill=(255, 255, 255, 255), width=max(5, r // 4), joint="curve")

    return Image.alpha_composite(img, overlay).convert("RGB")


def build_genre_balanced_pool(df: pd.DataFrame, seed: int) -> list[int]:
    if df.empty:
        return []

    work = df[["movieId", "genres"]].drop_duplicates("movieId").copy()
    work["genres"] = work["genres"].fillna("(no genres listed)").astype(str)
    work["genre"] = work["genres"].str.split("|")
    exploded = work.explode("genre")
    exploded["genre"] = exploded["genre"].fillna("(no genres listed)")

    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = {}
    for genre, group in exploded.groupby("genre"):
        ids = group["movieId"].drop_duplicates().astype(int).tolist()
        rng.shuffle(ids)
        buckets[str(genre)] = ids

    genres = list(buckets.keys())
    rng.shuffle(genres)

    ordered_ids: list[int] = []
    seen: set[int] = set()
    added = True
    while added:
        added = False
        for genre in genres:
            bucket = buckets[genre]
            while bucket and bucket[0] in seen:
                bucket.pop(0)
            if bucket:
                movie_id = int(bucket.pop(0))
                if movie_id not in seen:
                    seen.add(movie_id)
                    ordered_ids.append(movie_id)
                    added = True

    return ordered_ids


def show_recommendations(result: dict) -> None:
    recommendations = result.get("recommendations", [])
    if not recommendations:
        st.warning("No recommendations returned.")
        return

    for user_block in recommendations:
        user_id = user_block.get("user_id")
        st.subheader(f"User {user_id}")
        items = user_block.get("items", [])
        if not items:
            st.info("No items for this user.")
            continue
        df = pd.DataFrame(items)
        preferred_cols = ["rank", "movie_id", "title", "distance"]
        cols = [c for c in preferred_cols if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

    missing = result.get("missing_user_ids", [])
    if missing:
        st.info(f"Missing user IDs: {missing}")


def generate_demo_traffic(api_url: str, token: str, user_ids: list[int], top_k: int, num_requests: int, delay_ms: int) -> None:
    payload = {"user_ids": user_ids, "top_k": top_k}
    ok = 0
    failed = 0
    progress = st.progress(0)
    status_box = st.empty()

    for idx in range(num_requests):
        try:
            response = requests.post(f"{api_url}/recommend", json=payload, headers=api_headers(token), timeout=15)
            if response.status_code < 400:
                ok += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        ratio = int(((idx + 1) / num_requests) * 100)
        progress.progress(ratio)
        status_box.info(f"Traffic generation: {idx + 1}/{num_requests} requests")
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    status_box.success(f"Traffic complete. Successful: {ok}, Failed: {failed}")


def enable_scroll_restore() -> None:
    components.html(
        """
        <script>
        (function() {
          const key = "movie_picker_scroll_y";
          const p = window.parent;
          try {
            if (!p.__movieScrollListenerAttached) {
              p.addEventListener("scroll", function() {
                p.sessionStorage.setItem(key, String(p.scrollY || p.pageYOffset || 0));
              }, { passive: true });
              p.__movieScrollListenerAttached = true;
            }
            const saved = p.sessionStorage.getItem(key);
            if (saved !== null) {
              p.scrollTo(0, parseInt(saved, 10) || 0);
            }
          } catch (e) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_userid_page(api_url: str, token: str) -> None:
    st.subheader("Recommend by Existing User IDs")
    if not token:
        st.info("Login first to call /recommend.")
        return
    users_raw = st.text_input("User IDs (comma-separated)", value="1")
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1, key="topk_user")

    if st.button("Get Recommendations", type="primary"):
        try:
            user_ids = parse_user_ids(users_raw)
            result = call_recommend(api_url, token, user_ids, int(top_k))
            show_recommendations(result)
        except Exception as exc:
            st.error(f"Request failed: {exc}")

def render_admin_page(api_url: str, token: str, role: str) -> None:
    st.subheader("Admin Controls")
    if role != "admin":
        st.info("Login as admin to access health check, retraining, and dataset update.")
        return

    if st.button("Check API health (admin)"):
        try:
            r = requests.get(f"{api_url}/health", headers=api_headers(token), timeout=10)
            st.success(f"Health: {r.status_code} {r.text}")
        except Exception as exc:
            st.error(f"Health check failed: {exc}")

    if st.button("Run retraining (admin)"):
        with st.spinner("Running build_features + train_model..."):
            try:
                r = requests.post(f"{api_url}/admin/retrain", headers=api_headers(token), timeout=600)
                if r.status_code >= 400:
                    st.error(f"Retrain failed: {r.status_code} {r.text}")
                else:
                    st.success(r.json().get("details", "Retraining completed."))
            except Exception as exc:
                st.error(f"Retrain failed: {exc}")

    st.divider()
    st.markdown("### Generate demo traffic")
    users_raw = st.text_input("User IDs (comma-separated)", value="1", key="admin_users_raw")
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1, key="admin_topk_user")
    col1, col2 = st.columns(2)
    with col1:
        num_requests = st.number_input("Number of requests", min_value=1, max_value=500, value=50, step=1, key="admin_num_requests")
    with col2:
        delay_ms = st.number_input("Delay (ms)", min_value=0, max_value=5000, value=50, step=10, key="admin_delay_ms")
    if st.button("Generate Traffic for Monitoring (admin)"):
        try:
            user_ids = parse_user_ids(users_raw)
            generate_demo_traffic(
                api_url=api_url,
                token=token,
                user_ids=user_ids,
                top_k=int(top_k),
                num_requests=int(num_requests),
                delay_ms=int(delay_ms),
            )
        except Exception as exc:
            st.error(f"Traffic generation failed: {exc}")

    st.divider()
    st.markdown("### Append sample ratings")
    st.caption("Format: one line per rating -> userId,movieId,rating[,timestamp]")
    sample_rows = st.text_area("Ratings rows", value="999001,1,4.5\n999001,32,4.0")
    if st.button("Append sample dataset rows (admin)"):
        try:
            rows = []
            for raw in sample_rows.splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    raise ValueError(f"Invalid row: {line}")
                item = {"userId": int(parts[0]), "movieId": int(parts[1]), "rating": float(parts[2])}
                if len(parts) >= 4 and parts[3]:
                    item["timestamp"] = int(parts[3])
                rows.append(item)
            if not rows:
                raise ValueError("No valid rows provided.")
            r = requests.post(
                f"{api_url}/admin/dataset/append-ratings",
                headers=api_headers(token),
                json={"rows": rows},
                timeout=30,
            )
            if r.status_code >= 400:
                st.error(f"Append failed: {r.status_code} {r.text}")
            else:
                data = r.json()
                st.success(f"Appended {data.get('rows_appended')} rows to {data.get('ratings_path')}")
        except Exception as exc:
            st.error(f"Append failed: {exc}")


def render_movie_picker_page(token: str) -> None:
    if not token:
        st.subheader("Recommend by Selected Movies")
        st.info("Login first to use movie recommendations.")
        return
    enable_scroll_restore()
    st.subheader("Recommend by Selected Movies")
    st.caption("Pick movies by poster; selected movies are marked with a check.")

    model_path = st.session_state.model_path
    movie_matrix_path = st.session_state.movie_matrix_path
    movies_path = st.session_state.movies_path
    links_path = st.session_state.links_path
    top_k = int(st.session_state.top_k_movie)
    search_term = st.session_state.search_movie
    max_cards = int(st.session_state.max_cards)

    if not TMDB_API_KEY:
        st.info(
            "TMDB key not set. Using fallback poster lookup. "
            "Set STREAMLIT_TMDB_API_KEY (or TMDB_API_KEY) for better posters."
        )

    try:
        _ = Path(model_path)
        catalog = load_movie_catalog(movie_matrix_path, movies_path)
        links = load_links(links_path)
        catalog = catalog.merge(links, on="movieId", how="left")
        model = load_model(model_path)
    except Exception as exc:
        st.error(f"Loading data/model failed: {exc}")
        return

    filtered_df = catalog.copy()
    if search_term.strip():
        filtered_df = filtered_df[filtered_df["title"].str.contains(search_term, case=False, na=False)]
    if "poster_seed" not in st.session_state:
        st.session_state.poster_seed = int(time.time())
    pool_ids = build_genre_balanced_pool(filtered_df, seed=int(st.session_state.poster_seed))
    total_filtered = len(pool_ids)
    if "visible_cards" not in st.session_state:
        st.session_state.visible_cards = int(max_cards)
    if "last_search_term" not in st.session_state:
        st.session_state.last_search_term = search_term
    if st.session_state.last_search_term != search_term:
        st.session_state.visible_cards = int(max_cards)
        st.session_state.last_search_term = search_term
    st.session_state.visible_cards = max(int(max_cards), int(st.session_state.visible_cards))
    if total_filtered > 0:
        id_map = filtered_df.set_index("movieId")
        current_ids = pool_ids[: int(st.session_state.visible_cards)]
        view_df = id_map.loc[current_ids].reset_index()
    else:
        view_df = filtered_df.head(0).copy()

    if "selected_movie_ids" not in st.session_state:
        st.session_state.selected_movie_ids = set()
    if "movie_picker_version" not in st.session_state:
        st.session_state.movie_picker_version = 0
    st.caption("Click a poster image to select/deselect.")

    if len(view_df) > 0:
        poster_images = []
        captions = []
        movie_ids = []
        for _, row in view_df.iterrows():
            movie_id = int(row["movieId"])
            title = str(row["title"])
            tmdb_id = int(row["tmdbId"]) if "tmdbId" in row and pd.notna(row["tmdbId"]) else None
            is_selected = movie_id in st.session_state.selected_movie_ids
            poster = fetch_poster_image(title, tmdb_id=tmdb_id)
            if is_selected:
                poster = add_checkmark_badge(poster)
            poster_images.append(poster)
            captions.append(title)
            movie_ids.append(movie_id)

        clicked_idx = image_select(
            label="Movie posters",
            images=poster_images,
            captions=captions,
            index=-1,
            return_value="index",
            key=f"movie_picker_{st.session_state.movie_picker_version}",
            use_container_width=False,
        )

        if clicked_idx is not None and int(clicked_idx) >= 0:
            clicked_movie_id = movie_ids[int(clicked_idx)]
            if clicked_movie_id in st.session_state.selected_movie_ids:
                st.session_state.selected_movie_ids.discard(clicked_movie_id)
            else:
                st.session_state.selected_movie_ids.add(clicked_movie_id)
            st.session_state.movie_picker_version += 1
            st.rerun()

    selected_ids = sorted(list(st.session_state.selected_movie_ids))
    st.write(f"Selected movies: {len(selected_ids)}")
    if len(view_df) < total_filtered:
        if st.button(f"Show {int(max_cards)} more posters"):
            st.session_state.visible_cards = min(
                total_filtered,
                int(st.session_state.visible_cards) + int(max_cards),
            )
            st.rerun()
    clear = st.button("Clear selected movies")
    if clear:
        st.session_state.selected_movie_ids = set()
        st.session_state.movie_picker_version += 1
        st.rerun()

    if st.button("Recommend from selected movies", type="primary"):
        try:
            if not selected_ids:
                raise ValueError("Select at least one movie.")
            result_df = recommend_from_selected_movies(
                model=model,
                movie_matrix_path=movie_matrix_path,
                movies_path=movies_path,
                selected_movie_ids=selected_ids,
                top_k=int(top_k),
            )
            if result_df.empty:
                st.warning("No recommendations found.")
            else:
                result_df = result_df.copy()
                tmdb_map = catalog.set_index("movieId")["tmdbId"].to_dict()
                result_df["poster_url"] = result_df.apply(
                    lambda row: fetch_poster_url(row["title"], int(tmdb_map.get(row["movie_id"])) if pd.notna(tmdb_map.get(row["movie_id"])) else None),
                    axis=1,
                )

                st.subheader("Recommended Movies")
                cols_per_row = 5
                for i in range(0, len(result_df), cols_per_row):
                    cols = st.columns(cols_per_row)
                    chunk = result_df.iloc[i : i + cols_per_row]
                    for col, (_, row) in zip(cols, chunk.iterrows()):
                        with col:
                            st.image(row["poster_url"], width=120)
                            st.caption(f"#{int(row['rank'])} {row['title']}")
                            st.caption(f"id={int(row['movie_id'])} | d={row['distance']:.4f}")
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Movie Recommendation", layout="wide")
    st.title("Movie Recommendation App")

    # Defaults for movie-picker settings (used by sidebar + tab render)
    if "model_path" not in st.session_state:
        st.session_state.model_path = DEFAULT_MODEL_PATH
    if "movie_matrix_path" not in st.session_state:
        st.session_state.movie_matrix_path = DEFAULT_MOVIE_MATRIX_PATH
    if "movies_path" not in st.session_state:
        st.session_state.movies_path = DEFAULT_MOVIES_PATH
    if "links_path" not in st.session_state:
        st.session_state.links_path = DEFAULT_LINKS_PATH
    if "top_k_movie" not in st.session_state:
        st.session_state.top_k_movie = 5
    if "search_movie" not in st.session_state:
        st.session_state.search_movie = ""
    if "max_cards" not in st.session_state:
        st.session_state.max_cards = 16
    if "api_token" not in st.session_state:
        st.session_state.api_token = ""
    if "api_role" not in st.session_state:
        st.session_state.api_role = ""

    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("Inference API URL", value=DEFAULT_API_URL)
        st.subheader("API Login")
        username = st.text_input("Username", value="user")
        password = st.text_input("Password", value="", type="password")
        col_login, col_logout = st.columns(2)
        with col_login:
            if st.button("Login"):
                try:
                    auth = login_api(api_url, username, password)
                    st.session_state.api_token = auth.get("access_token", "")
                    st.session_state.api_role = auth.get("role", "")
                    st.success(f"Logged in as {st.session_state.api_role}")
                except Exception as exc:
                    st.error(f"Login failed: {exc}")
        with col_logout:
            if st.button("Logout"):
                st.session_state.api_token = ""
                st.session_state.api_role = ""
                st.info("Logged out.")
        if st.session_state.api_role:
            st.caption(f"Role: {st.session_state.api_role}")

        st.subheader("Movie Picker Settings")
        st.session_state.model_path = st.text_input("Model path", value=st.session_state.model_path)
        st.session_state.movie_matrix_path = st.text_input("Movie matrix path", value=st.session_state.movie_matrix_path)
        st.session_state.movies_path = st.text_input("Movies metadata path", value=st.session_state.movies_path)
        st.session_state.links_path = st.text_input("Links path", value=st.session_state.links_path)
        st.session_state.top_k_movie = st.number_input(
            "Top K", min_value=1, max_value=50, value=int(st.session_state.top_k_movie), step=1
        )
        st.session_state.search_movie = st.text_input("Search movie title", value=st.session_state.search_movie)
        st.session_state.max_cards = st.number_input(
            "Movies to display", min_value=6, max_value=60, value=int(st.session_state.max_cards), step=6
        )

    if not st.session_state.api_token:
        st.warning("Login in sidebar to use API-backed recommendation pages.")

    tab_movie, tab_user, tab_admin = st.tabs(["Recommend by Selected Movies", "Recommend by User IDs", "Admin"])
    with tab_movie:
        render_movie_picker_page(st.session_state.api_token)
    with tab_user:
        render_userid_page(api_url, st.session_state.api_token)
    with tab_admin:
        render_admin_page(api_url, st.session_state.api_token, st.session_state.api_role)


if __name__ == "__main__":
    main()
