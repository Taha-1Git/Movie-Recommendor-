# HexSoftwares_Music_Recommendation_System

A hybrid music recommendation engine built for the **Hex Softwares Machine Learning Internship**, powered by the real **Kaggle Spotify Tracks Dataset**.

## Overview
This project recommends songs to users using three complementary techniques:

1. **Content-Based Filtering** — recommends songs with similar audio features (danceability, energy, valence, tempo, acousticness, instrumentalness, loudness, speechiness, liveness) and genre, using cosine similarity over standardized + one-hot encoded features.
2. **Collaborative Filtering** — builds a user–item rating matrix from listening history and recommends songs liked by the most similar users (user-based CF with cosine similarity).
3. **Hybrid Recommender** — blends content-based and collaborative scores into a single ranked list, anchored on the user's highest-rated song.

## Dataset
Source: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle) — `dataset.csv`, ~114,000 tracks with audio features across 113 genres.

Since this dataset only contains track-level audio features (no real user listening data), user listening histories are **simulated** using genre-affinity, popularity-weighted sampling — each synthetic user is assigned 1-3 favorite genres and "plays"/rates real tracks from those genres. This is the standard approach for building a collaborative-filtering layer on top of a catalog-only dataset.

## Project Structure
```
HexSoftwares_Music_Recommendation_System/
├── data_loader.py         # Cleans the raw Kaggle dataset + simulates user listening history
├── recommender.py         # Core recommendation engine (content / collaborative / hybrid)
├── main.py                 # Demo script — run this to see recommendations
├── data/
│   ├── songs.csv               # Cleaned, deduplicated, stratified-sampled song catalog
│   └── listening_history.csv   # Simulated user listening/rating history
└── README.md
```
> Note: `data/spotify_raw.csv` (the original Kaggle file, ~20MB) is not included in this repo to keep it lightweight. Download `dataset.csv` from the [Kaggle page](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset), place it at `data/spotify_raw.csv`, then run `data_loader.py` to regenerate `songs.csv` and `listening_history.csv` (already included here for convenience).

## How It Works

### 1. Data Cleaning (`data_loader.py`)
- Drops rows with missing artist/title/genre, removes duplicate `track_id`s (the raw file has tracks repeated across genre playlists), filters out near-zero-duration tracks.
- Stratified-samples down to ~4,000 tracks (balanced across all 113 genres) for fast similarity computation while keeping genre diversity.
- Simulates 80 users with genre preferences and popularity-weighted listening histories.

### 2. Content-Based Filtering
Audio features are standardized (`StandardScaler`) and genre is one-hot encoded, then combined into a single feature matrix. Cosine similarity between songs recommends tracks that "sound like" a given song.

### 3. Collaborative Filtering
A user-item matrix is built from simulated ratings. Cosine similarity between users identifies "taste neighbors," and recommendations are the songs those neighbors rated highly that the target user hasn't heard yet (similarity-weighted average rating).

### 4. Hybrid
For a given user, the system anchors on their top-rated song (content signal) and their nearest-neighbor users (collaborative signal), normalizes both score sets, and blends them (default 50/50) into one final ranked list.

## Setup & Usage
```bash
pip install pandas numpy scikit-learn

# 1. Clean the raw dataset + simulate user histories
python data_loader.py

# 2. Run the recommender demo
python main.py
```

## Example Output
```
[Content-Based] Songs similar to: 'Maine Pyar Tumhi Se Kiya Hai' by Anuradha Paudwal;Kumar Sanu (k-pop)
...

[Hybrid] Final blended recommendations for User 3
               song_id                                   title                               artist genre  hybrid_score
6KlbKM85NIUo5ZaToCAFFc If You Really Love Me (How Will I Know)    David Guetta;MistaJam;John Newman dance         0.735
0nev2Tq1NaEnS7Ry3OYZQh                                   Ghost                           Katy Perry dance         0.719
...
```

## Tech Stack
- Python 3
- pandas, numpy
- scikit-learn (StandardScaler, OneHotEncoder, cosine_similarity)

## Author
Submitted as part of the Hex Softwares Machine Learning Internship Program.

