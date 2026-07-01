"""
HexSoftwares_Music_Recommendation_System
main.py - Demo / CLI entry point
"""

from recommender import MusicRecommender

def main():
    rec = MusicRecommender()

    print("=" * 70)
    print("HEX SOFTWARES — MUSIC RECOMMENDATION SYSTEM")
    print("=" * 70)

    sample_song = rec.songs.sample(1, random_state=1).iloc[0]
    print(f"\n[Content-Based] Songs similar to: '{sample_song['title']}' "
          f"by {sample_song['artist']} ({sample_song['genre']})\n")
    print(rec.content_based_recommend(sample_song["song_id"], top_n=5).to_string(index=False))

    sample_user = 3
    print(f"\n[Collaborative Filtering] Recommendations for User {sample_user} "
          f"(based on similar users' taste)\n")
    print(rec.collaborative_recommend(sample_user, top_n=5).to_string(index=False))

    print(f"\n[Hybrid] Final blended recommendations for User {sample_user}\n")
    print(rec.hybrid_recommend(sample_user, top_n=5).to_string(index=False))


if __name__ == "__main__":
    main()
