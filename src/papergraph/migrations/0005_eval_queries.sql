-- Benchmark queries and relevance labels, stored next to the corpus they label.
-- In a benchmark schema (bench_litsearch) these hold that benchmark's queries;
-- in public they stay empty until there are labelled queries for the arXiv corpus.

CREATE TABLE eval_queries (
    qid         TEXT PRIMARY KEY,     -- stable id derived from the query text
    text        TEXT NOT NULL,
    split       TEXT NOT NULL CHECK (split IN ('dev', 'test')),
    -- Breakdown dimensions: benchmark-specific labels such as LitSearch's
    -- {"query_set": "manual_iclr", "specificity": "broad"}.
    facets      JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE eval_qrels (
    qid      TEXT NOT NULL REFERENCES eval_queries (qid) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    gain     REAL NOT NULL DEFAULT 1,  -- relevance grade; binary benchmarks use 1
    PRIMARY KEY (qid, paper_id)
);
