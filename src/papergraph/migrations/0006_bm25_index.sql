-- BM25 inverted index, built from papers.search_tsv by papergraph.retrieval.bm25.
-- Postgres full-text search has no BM25 (ts_rank has no IDF and no saturation),
-- so we keep the statistics BM25 needs ourselves. These tables are derived
-- data: `build_index` rebuilds them wholesale after the corpus changes.

-- One row per (term, paper): how often the stemmed term occurs, split by field
-- so a title boost can be applied at query time without a rebuild.
CREATE TABLE bm25_postings (
    term     TEXT NOT NULL,
    paper_id TEXT NOT NULL,       -- no FK: rebuilt wholesale, and an FK check
                                  -- per row would slow the bulk build down
    tf_title  SMALLINT NOT NULL,
    tf_body   SMALLINT NOT NULL,
    -- The paper's length (in indexed tokens) per field, copied into every
    -- posting so scoring needs no join. Per field because a title boost
    -- scales the length the same way it scales tf.
    len_title INTEGER NOT NULL,
    len_body  INTEGER NOT NULL,
    PRIMARY KEY (term, paper_id)  -- also the lookup index: all postings of a term
);

-- Document frequency: in how many papers each term occurs (for IDF).
CREATE TABLE bm25_terms (
    term TEXT PRIMARY KEY,
    df   INTEGER NOT NULL
);

-- Corpus-level statistics: a single row.
CREATE TABLE bm25_stats (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    n_docs        INTEGER NOT NULL,  -- every paper, including ones with no terms
    avg_len_title REAL NOT NULL,
    avg_len_body  REAL NOT NULL,
    built_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
