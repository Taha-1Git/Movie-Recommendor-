"""
HexSoftwares_Music_Recommendation_System
recommender.py

Implements:
1. Content-Based Filtering  -> recommends songs with similar audio features/genre
2. Collaborative Filtering  -> recommends songs liked by similar users (user-item matrix + cosine similarity)
3. Hybrid Recommender       -> blends both for a final ranked list
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


class MusicRecommender:
    def __init__(self, songs_path="data/songs.csv", history_path="data/listening_history.csv"):
        self.songs = pd.read_csv(songs_path)
        self.history = pd.read_csv(history_path)
        self._build_content_features()
        self._build_user_item_matrix()

    # ---------------------------------------------------------------
    # CONTENT-BASED FILTERING
    # ---------------------------------------------------------------
    def _build_content_features(self):
        numeric_cols = ["danceability", "energy", "valence", "tempo",
                         "acousticness", "instrumentalness", "loudness",
                         "speechiness", "liveness"]
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(self.songs[numeric_cols])

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        genre_encoded = encoder.fit_transform(self.songs[["genre"]])

        self.content_matrix = np.hstack([numeric_scaled, genre_encoded])
        self.content_sim = cosine_similarity(self.content_matrix)

    def content_based_recommend(self, song_id, top_n=10):
        idx = self.songs.index[self.songs["song_id"] == song_id][0]
        scores = list(enumerate(self.content_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        result = self.songs.iloc[[s[0] for s in scores]].copy()
        result["similarity_score"] = [round(s[1], 3) for s in scores]
        return result[["song_id", "title", "artist", "genre", "similarity_score"]]

    # ---------------------------------------------------------------
    # COLLABORATIVE FILTERING (user-based)
    # ---------------------------------------------------------------
    def _build_user_item_matrix(self):
        self.user_item = self.history.pivot_table(
            index="user_id", columns="song_id", values="rating", fill_value=0
        )
        self.user_sim = cosine_similarity(self.user_item)
        self.user_sim_df = pd.DataFrame(
            self.user_sim, index=self.user_item.index, columns=self.user_item.index
        )

    def collaborative_recommend(self, user_id, top_n=10):
        if user_id not in self.user_item.index:
            return pd.DataFrame(columns=["song_id", "title", "artist", "predicted_score"])

        sim_users = self.user_sim_df[user_id].drop(user_id).sort_values(ascending=False)
        top_sim_users = sim_users.head(10)

        already_heard = set(self.history[self.history["user_id"] == user_id]["song_id"])

        # weighted average rating from similar users for unseen songs
        weighted_scores = {}
        for other_user, sim in top_sim_users.items():
            if sim <= 0:
                continue
            other_ratings = self.user_item.loc[other_user]
            for song_id, rating in other_ratings[other_ratings > 0].items():
                if song_id in already_heard:
                    continue
                weighted_scores.setdefault(song_id, [0, 0])
                weighted_scores[song_id][0] += sim * rating
                weighted_scores[song_id][1] += sim

        scored = [(sid, num / den) for sid, (num, den) in weighted_scores.items() if den > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_n]

        result = self.songs[self.songs["song_id"].isin([s[0] for s in scored])].copy()
        score_map = dict(scored)
        result["predicted_score"] = result["song_id"].map(score_map).round(2)
        result = result.sort_values("predicted_score", ascending=False)
        return result[["song_id", "title", "artist", "genre", "predicted_score"]]

    # ---------------------------------------------------------------
    # HYBRID RECOMMENDER
    # ---------------------------------------------------------------
    def hybrid_recommend(self, user_id, top_n=10, content_weight=0.5):
        """
        Combines collaborative filtering (what similar users liked) with
        content-based filtering (songs similar to what THIS user already likes).
        """
        already_heard = self.history[self.history["user_id"] == user_id]
        if already_heard.empty:
            return pd.DataFrame(columns=["song_id", "title", "artist", "genre", "hybrid_score"])

        # User's favorite song (highest rated) anchors the content-based search
        top_song_id = already_heard.sort_values("rating", ascending=False).iloc[0]["song_id"]

        content_recs = self.content_based_recommend(top_song_id, top_n=30)
        collab_recs = self.collaborative_recommend(user_id, top_n=30)

        # normalize scores to 0-1 then blend
        content_recs = content_recs.rename(columns={"similarity_score": "score"})
        collab_recs = collab_recs.rename(columns={"predicted_score": "score"})
        if not collab_recs.empty:
            collab_recs["score"] = collab_recs["score"] / collab_recs["score"].max()

        combined = pd.concat([
            content_recs[["song_id", "score"]].assign(source="content"),
            collab_recs[["song_id", "score"]].assign(source="collaborative"),
        ])

        agg = combined.groupby("song_id")["score"].apply(
            lambda x: content_weight * x.mean() if len(x) == 1 else
            content_weight * combined[(combined.song_id == x.name) & (combined.source == "content")]["score"].mean()
            + (1 - content_weight) * combined[(combined.song_id == x.name) & (combined.source == "collaborative")]["score"].mean()
        ).fillna(0)

        # simpler, robust blend (avoids NaN edge cases above)
        pivot = combined.pivot_table(index="song_id", columns="source", values="score", fill_value=0)
        if "content" not in pivot:
            pivot["content"] = 0
        if "collaborative" not in pivot:
            pivot["collaborative"] = 0
        pivot["hybrid_score"] = (content_weight * pivot["content"] +
                                  (1 - content_weight) * pivot["collaborative"])
        pivot = pivot.sort_values("hybrid_score", ascending=False).head(top_n)

        result = self.songs[self.songs["song_id"].isin(pivot.index)].copy()
        result = result.set_index("song_id").loc[pivot.index].reset_index()
        result["hybrid_score"] = pivot["hybrid_score"].round(3).values
        return result[["song_id", "title", "artist", "genre", "hybrid_score"]]
