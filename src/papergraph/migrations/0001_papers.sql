-- Paper-level corpus: one row per arXiv paper, plus its categories and authors.

CREATE TABLE papers (
    -- arXiv id, e.g. '2310.06825' (new style) or 'cs/0112017' (pre-2007 style).
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    abstract         TEXT NOT NULL,
    doi              TEXT,            -- used later to join OpenAlex citations
    primary_category TEXT NOT NULL,   -- first listed category, e.g. 'cs.CL'
    published        DATE NOT NULL,   -- date of v1: needed for leakage-safe evals
    updated          DATE,            -- last metadata update
    -- Keyword-search document, kept in sync by Postgres itself. Title matches
    -- (weight A) count more than abstract matches (weight B).
    search_tsv       TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', title), 'A') ||
        setweight(to_tsvector('english', abstract), 'B')
    ) STORED
);

CREATE INDEX papers_search_tsv_idx ON papers USING GIN (search_tsv);
CREATE INDEX papers_published_idx ON papers (published);

-- A paper can be cross-listed in several categories (cs.CL + cs.LG ...).
-- A separate table lets us filter by any of them with an index.
CREATE TABLE paper_categories (
    paper_id TEXT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    PRIMARY KEY (paper_id, category)
);

CREATE INDEX paper_categories_category_idx ON paper_categories (category);

-- Authors are stored as names in order. arXiv has no author ids, so two people
-- with the same name are indistinguishable here; OpenAlex author ids can
-- replace this later (milestone 6).
CREATE TABLE paper_authors (
    paper_id TEXT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    position SMALLINT NOT NULL,
    name     TEXT NOT NULL,
    PRIMARY KEY (paper_id, position)
);

CREATE INDEX paper_authors_name_idx ON paper_authors (name);

-- One table per embedding model, because a pgvector index needs a fixed
-- dimension (bge-small = 384, SPECTER2 = 768). SPECTER2 gets its own migration.
CREATE TABLE paper_embeddings_bge_small (
    paper_id  TEXT PRIMARY KEY REFERENCES papers (id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL
);

-- HNSW: approximate nearest-neighbor index. vector_cosine_ops makes the `<=>`
-- operator (cosine distance) use it.
CREATE INDEX paper_embeddings_bge_small_hnsw_idx
    ON paper_embeddings_bge_small USING hnsw (embedding vector_cosine_ops);
