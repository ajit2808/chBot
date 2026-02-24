import torch
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def generate_embeddings(texts, batch_size=128):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings