-- Document-level table
CREATE TABLE IF NOT EXISTS user_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,

    filename TEXT,
    markdown_content TEXT,

    summary_markdown TEXT,
    summary_embedding vector(384),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_files_summary_vec
ON user_files USING ivfflat (summary_embedding vector_cosine_ops)
WITH (lists = 100);

-- Chunk-level table
CREATE TABLE IF NOT EXISTS user_file_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID REFERENCES user_files(id) ON DELETE CASCADE,

    chunk_index INTEGER,
    chunk_markdown TEXT,
    chunk_embedding vector(384),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_file_chunks_vec
ON user_file_chunks USING ivfflat (chunk_embedding vector_cosine_ops)
WITH (lists = 100);
