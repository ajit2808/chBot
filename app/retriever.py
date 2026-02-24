from app.embeddings import generate_embeddings
from app.vector_store import search


def retrieve(question, top_k=15):
    query_embedding = generate_embeddings([question])[0]
    docs = search(query_embedding, top_k=top_k)
    return docs