-- Benchmark corpora (LitSearch, SciDocs) reuse this schema but lack some arXiv
-- metadata: they have no arXiv category, and some have only a year or no date.
-- arXiv ingest still always fills both columns; only the constraint is relaxed.

ALTER TABLE papers
    ALTER COLUMN primary_category DROP NOT NULL,
    ALTER COLUMN published DROP NOT NULL;
