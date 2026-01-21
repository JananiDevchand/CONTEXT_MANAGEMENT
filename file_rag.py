from file_retriever import find_relevant_files, retrieve_top_chunks


def build_file_rag_context(user_id: str, query: str):
    file_ids = find_relevant_files(user_id, query)

    if not file_ids:
        return []

    chunks = retrieve_top_chunks(file_ids, query)

    if not chunks:
        return []

    return [{
        "role": "system",
        "content": "\n\n".join(
            f"### Relevant Section\n{c['chunk_markdown']}"
            for c in chunks
        ),
        "metadata": {
            "source": "user_file_rag"
        }
    }]
