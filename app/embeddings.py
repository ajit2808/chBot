from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts, batch_size=64):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.extend(emb)

    return np.array(embeddings)