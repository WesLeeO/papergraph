-- The HNSW index on paper_embeddings_bge_small is no longer part of the schema.
--
-- Created up front (0001), every embedding insert has to be linked into the
-- graph one row at a time, which gets slower as the table grows. Building the
-- index once over all vectors is much faster, so `papergraph.ingest.embed` now
-- drops it before a load and creates it again afterwards. Until that build
-- finishes, `<=>` queries still work, through an exact sequential scan.

DROP INDEX IF EXISTS paper_embeddings_bge_small_hnsw_idx;
