from file_rag import build_file_rag_context

USER_ID = "11111111-1111-1111-1111-111111111111"

query = "Explain transformer architecture"

ctx = build_file_rag_context(USER_ID, query)

print(ctx)
