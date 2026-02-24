import faiss
import numpy as np
import os
from app.config import settings

INDEX_PATH = "data/faiss.index"
META_PATH = "data/faiss_meta.npy"

os.makedirs("data", exist_ok=True)

# ===============================
# Initialize or Load Index Safely
# ===============================

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        metadata = np.load(META_PATH, allow_pickle=True).tolist()

        # Safety check
        if index.ntotal != len(metadata):
            print("⚠ Index/metadata mismatch. Rebuilding index.")
            return create_new_index()

        return index, metadata

    return create_new_index()


def create_new_index():
    index = faiss.IndexFlatL2(settings.VECTOR_DIM)
    metadata = []
    return index, metadata


index, metadata = load_index()


# ===============================
# Add Documents Safely
# ===============================

def add_documents(texts, embeddings):
    global index, metadata

    vectors = np.array(embeddings).astype("float32")

    if len(vectors) == 0:
        return

    index.add(vectors)
    metadata.extend(texts)

    save_index()


def save_index():
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(metadata, dtype=object))


# ===============================
# Safe Search
# ===============================

def search(query_embedding, top_k=5):
    if index.ntotal == 0:
        return []

    vector = np.array([query_embedding]).astype("float32")
    D, I = index.search(vector, top_k)

    results = []

    for idx in I[0]:
        if 0 <= idx < len(metadata):
            results.append(metadata[idx])

    return results