import faiss
import numpy as np
import os
from app.config import settings

DB_PATH = "data/vectordb"
os.makedirs(DB_PATH, exist_ok=True)

INDEX_PATH = os.path.join(DB_PATH, "faiss.index")
META_PATH = os.path.join(DB_PATH, "faiss_meta.npy")

VECTOR_DIM = settings.VECTOR_DIM
NLIST = 100  # number of clusters for IVF


def create_new_index():
    quantizer = faiss.IndexFlatL2(VECTOR_DIM)
    index = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, NLIST)
    metadata = []
    return index, metadata


def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        metadata = np.load(META_PATH, allow_pickle=True).tolist()
        return index, metadata
    return create_new_index()


index, metadata = load_index()


def train_index_if_needed(vectors):
    if not index.is_trained:
        index.train(vectors)


def save_index():
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(metadata, dtype=object))


def add_documents(texts, embeddings, batch_size=512):
    global metadata

    vectors = np.array(embeddings).astype("float32")

    train_index_if_needed(vectors)

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.add(batch)

    metadata.extend(texts)
    save_index()


def search(query_embedding, top_k=10):
    if index.ntotal == 0:
        return []

    vector = np.array([query_embedding]).astype("float32")
    index.nprobe = 10  # accuracy/speed tradeoff

    D, I = index.search(vector, top_k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            results.append(metadata[idx])

    return results