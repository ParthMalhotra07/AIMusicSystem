import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)

def build_user_vector(song_embeddings, user_history):
    """
    song_embeddings: np.array (N_songs, embedding_dim)
    user_history: list of song indices listened by user
    """
    user_vec = np.mean(song_embeddings[user_history], axis=0)
    user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)
    return user_vec.reshape(1, -1)

def build_user_vector_weighted(embeddings, history):
    weights = np.exp(np.linspace(0, 1, len(history)))  # recent songs matter more
    weighted_vecs = embeddings[history] * weights[:, None]
    user_vec = weighted_vecs.mean(axis=0)
    return user_vec.reshape(1, -1) / (np.linalg.norm(user_vec)+1e-8)

def recommend_songs(song_embeddings, song_ids, user_history, top_k=5):
    user_vec = build_user_vector_weighted(song_embeddings, user_history)

    similarities = cosine_similarity(user_vec, song_embeddings)[0]

    # Exclude already listened songs
    similarities[user_history] = -1  

    top_indices = similarities.argsort()[::-1][:top_k]

    recommendations = [(song_ids[i],i, similarities[i]) for i in top_indices if (similarities[i]>0)]
    return recommendations

def cold_start_recommendation(song_embeddings, song_ids, top_k=5):
    popularity_proxy = np.linalg.norm(song_embeddings, axis=1)
    top_indices = popularity_proxy.argsort()[::-1][:top_k]
    return [song_ids[i] for i in top_indices]

def explain_similarity(song_a_vec, song_b_vec):
    score = cosine_similarity(
        song_a_vec.reshape(1,-1),
        song_b_vec.reshape(1,-1)
    )[0][0]
    return score


# Only run demo code when executed directly, not when imported
if __name__ == "__main__":
    # Dummy example - update path for your system
    import os
    from pathlib import Path
    
    # Try to find embeddings in relative path
    script_dir = Path(__file__).parent.parent
    file = script_dir / "Member2" / "song_embeddings.npy"
    
    if not file.exists():
        file = script_dir / "data" / "song_embeddings.npy"
    
    if file.exists():
        song_embeddings = np.load(str(file))
        song_embeddings = normalize_embeddings(song_embeddings)
        song_ids = [f"Song_{i}" for i in range(song_embeddings.shape[0])]

        user_history = [2, 5, 7]  # User listened to these songs
        if (len(user_history)<2):
            recs = cold_start_recommendation(song_embeddings, song_ids)
        else:
            recs = recommend_songs(song_embeddings, song_ids, user_history)

        for song, i, score in recs:
            print(song, "Similarity:", round(score, 3))
        similarity = explain_similarity(song_embeddings[user_history[-1]], song_embeddings[recs[0][1]])
        print(similarity)
    else:
        print(f"Embeddings file not found at {file}")
