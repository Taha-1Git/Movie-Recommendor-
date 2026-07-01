

import numpy as np
import pandas as pd

RAW_PATH = "data/spotify_raw.csv"
SONGS_OUT = "data/songs.csv"
HISTORY_OUT = "data/listening_history.csv"

N_USERS = 80
np.random.seed(42)


def load_and_clean_songs(path=RAW_PATH, sample_size=4000):
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.dropna(subset=["track_id", "artists", "track_name", "track_genre"])
    df = df.drop_duplicates(subset=["track_id"])
   
    df = df[(df["duration_ms"] > 30000) & (df["popularity"] >= 0)]

    df = df.rename(columns={
        "track_id": "song_id",
        "track_name": "title",
        "artists": "artist",
        "track_genre": "genre",
    })
    df["duration_sec"] = (df["duration_ms"] / 1000).round(0).astype(int)

   
    if sample_size and len(df) > sample_size:
        per_genre = max(1, sample_size // df["genre"].nunique())
        parts = [g.sample(min(len(g), per_genre), random_state=42)
                 for _, g in df.groupby("genre")]
        df = pd.concat(parts, ignore_index=False)

    df = df.reset_index(drop=True)

    keep_cols = ["song_id", "title", "artist", "genre", "popularity",
                 "danceability", "energy", "valence", "tempo", "loudness",
                 "acousticness", "instrumentalness", "speechiness", "liveness",
                 "duration_sec", "explicit"]
    return df[keep_cols]


def generate_listening_history(songs_df, n_users=N_USERS):
   
    rows = []
    genres = songs_df["genre"].unique()

    for user_id in range(n_users):
        fav_genres = np.random.choice(genres, size=np.random.randint(1, 3), replace=False)
        candidates = songs_df[songs_df["genre"].isin(fav_genres)]
        if len(candidates) < 5:
            candidates = songs_df

        # popularity-weighted sampling: users are somewhat more likely to have heard popular tracks
        weights = candidates["popularity"].clip(lower=1).values
        weights = weights / weights.sum()

        n_plays = np.random.randint(10, 25)
        n_plays = min(n_plays, len(candidates))
        played_idx = np.random.choice(candidates.index, size=n_plays, replace=False, p=weights)
        played = candidates.loc[played_idx]

        for _, song in played.iterrows():
            rating = np.clip(np.random.normal(4.0, 0.9), 1, 5)
            rows.append({
                "user_id": user_id,
                "song_id": song["song_id"],
                "rating": round(rating, 1),
                "play_count": np.random.randint(1, 40),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    songs = load_and_clean_songs()
    history = generate_listening_history(songs)

    songs.to_csv(SONGS_OUT, index=False)
    history.to_csv(HISTORY_OUT, index=False)

    print(f"Cleaned catalog: {len(songs)} songs across {songs['genre'].nunique()} genres")
    print(f"Simulated listening history: {len(history)} records across {history['user_id'].nunique()} users")
    print(songs.head())
