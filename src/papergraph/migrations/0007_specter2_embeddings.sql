-- SPECTER2 paper vectors (768 dims), for comparison with bge-small (384).
-- One table per model because a pgvector column has a fixed dimension.
-- The HNSW index is built by papergraph.ingest.embed after loading (see 0003).
CREATE TABLE paper_embeddings_specter2 (
    paper_id  TEXT PRIMARY KEY REFERENCES papers (id) ON DELETE CASCADE,
    embedding VECTOR(768) NOT NULL
);
