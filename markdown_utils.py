def count_tokens(text: str) -> int:
    return len(text.split())


def chunk_markdown(markdown: str, max_tokens=300):
    words = markdown.split()
    chunks, cur = [], []

    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []

    if cur:
        chunks.append(" ".join(cur))

    return chunks
