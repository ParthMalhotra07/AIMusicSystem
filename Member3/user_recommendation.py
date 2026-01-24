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

    recommendations = [(song_ids[i],i, similarities[i]) for i in top_indices]
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

# Dummy example
file = r"C:\Users\vksin\OneDrive\Desktop\AGMT\codes\AIMusicSystem\Member2\song_embeddings.npy"
song_embeddings = np.load(file)  # 20 songs, 64-dim embeddings
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
