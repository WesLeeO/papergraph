-- Paper vectors for the M4b embedder sweep, one table per candidate model
-- (a pgvector column has a fixed dimension). Filled on the LitSearch schema
-- first; the public corpus only gets the winner. Empty tables cost nothing.
-- HNSW indexes are built by papergraph.ingest.embed after loading (see 0003).
CREATE TABLE paper_embeddings_bge_base (
    paper_id  TEXT PRIMARY KEY REFERENCES papers (id) ON DELETE CASCADE,
    embedding VECTOR(768) NOT NULL
);
CREATE TABLE paper_embeddings_bge_large (
    paper_id  TEXT PRIMARY KEY REFERENCES papers (id) ON DELETE CASCADE,
    embedding VECTOR(1024) NOT NULL
);
CREATE TABLE paper_embeddings_gte_base (
    paper_id  TEXT PRIMARY KEY REFERENCES papers (id) ON DELETE CASCADE,
    embedding VECTOR(768) NOT NULL
);
CREATE TABLE paper_embeddings_arctic_m (
    paper_id  TEXT PRIMARY KEY REFERENCES papers (id) ON DELETE CASCADE,
    embedding VECTOR(768) NOT NULL
);
CREATE TABLE paper_embeddings_nomic (
    paper_id  TEXT PRIMARY KEY REFERENCES papers (id) ON DELETE CASCADE,
    embedding VECTOR(768) NOT NULL
);
CREATE TABLE paper_embeddings_qwen3_0_6b (
    paper_id  TEXT PRIMARY KEY REFERENCES papers (id) ON DELETE CASCADE,
    embedding VECTOR(1024) NOT NULL
);
