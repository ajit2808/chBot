def chunk_texts(texts, chunk_size=200):
    chunks = []
    buffer = ""

    for row in texts:
        if len(buffer) + len(row) < chunk_size:
            buffer += row + "\n"
        else:
            chunks.append(buffer.strip())
            buffer = row + "\n"

    if buffer:
        chunks.append(buffer.strip())

    return chunks