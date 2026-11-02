-- Where a paper was published. NULL venue = preprint, or venue not found yet.

ALTER TABLE papers
    ADD COLUMN venue          TEXT,      -- normalized, e.g. 'ICLR', 'CVPR', 'Nature'
    ADD COLUMN venue_year     SMALLINT,  -- year of the venue, can differ from arXiv v1 year
    ADD COLUMN venue_raw      TEXT,      -- the string we normalized from, kept for debugging
    ADD COLUMN venue_source   TEXT
        CHECK (venue_source IN ('semantic_scholar', 'arxiv_comments')),
    ADD COLUMN s2_paper_id    TEXT,      -- Semantic Scholar id: needed again for citations
    ADD COLUMN citation_count INTEGER;   -- from Semantic Scholar; a possible ranking signal

-- Partial index: only published papers are indexed, which is what venue filters query.
CREATE INDEX papers_venue_idx ON papers (venue) WHERE venue IS NOT NULL;
