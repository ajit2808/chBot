from app.embeddings import generate_embeddings
from app.vector_store import search

def retrieve(query):
    embedding = generate_embeddings([query])[0]
    return search(embedding, top_k=5)