def chunk_texts(texts, chunk_size=1000, overlap=200):
    chunks = []

    for text in texts:
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap

    return chunks