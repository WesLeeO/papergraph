"""Bulk-load arXiv ML papers into Postgres. Run with `make ingest [LIMIT=5000]`.

Each batch is COPY'd into temporary staging tables, then upserted. COPY streams
rows in Postgres's bulk format and is far faster than one INSERT per row; the
upsert (INSERT ... ON CONFLICT) makes re-running the ingest safe.
"""

import argparse
import time
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

import psycopg

from papergraph.config import get_settings
from papergraph.ingest.arxiv import Paper, iter_papers

DEFAULT_PATH = Path("data/raw/arxiv-metadata-oai-snapshot.json")

_STAGE = """
CREATE TEMP TABLE stage_papers (
    id TEXT, title TEXT, abstract TEXT, doi TEXT, primary_category TEXT,
    published DATE, updated DATE, venue TEXT, venue_year SMALLINT, venue_raw TEXT
) ON COMMIT DROP;
CREATE TEMP TABLE stage_categories (paper_id TEXT, category TEXT) ON COMMIT DROP;
CREATE TEMP TABLE stage_authors (paper_id TEXT, position SMALLINT, name TEXT) ON COMMIT DROP;
"""

_UPSERT = """
INSERT INTO papers (id, title, abstract, doi, primary_category, published, updated,
                    venue, venue_year, venue_raw, venue_source)
SELECT DISTINCT ON (id) id, title, abstract, doi, primary_category, published, updated,
       venue, venue_year, venue_raw, CASE WHEN venue IS NOT NULL THEN 'arxiv_comments' END
FROM stage_papers
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title,
    abstract = EXCLUDED.abstract,
    doi = EXCLUDED.doi,
    primary_category = EXCLUDED.primary_category,
    published = EXCLUDED.published,
    updated = EXCLUDED.updated,
    -- Only overwrite venues that came from this same regex source; a better
    -- source (DBLP, Semantic Scholar) must not be clobbered by a re-ingest.
    venue = CASE WHEN papers.venue_source IS DISTINCT FROM 'arxiv_comments'
                      AND papers.venue_source IS NOT NULL
                 THEN papers.venue ELSE EXCLUDED.venue END,
    venue_year = CASE WHEN papers.venue_source IS DISTINCT FROM 'arxiv_comments'
                           AND papers.venue_source IS NOT NULL
                      THEN papers.venue_year ELSE EXCLUDED.venue_year END,
    venue_raw = CASE WHEN papers.venue_source IS DISTINCT FROM 'arxiv_comments'
                          AND papers.venue_source IS NOT NULL
                     THEN papers.venue_raw ELSE EXCLUDED.venue_raw END,
    venue_source = CASE WHEN papers.venue_source IS DISTINCT FROM 'arxiv_comments'
                             AND papers.venue_source IS NOT NULL
                        THEN papers.venue_source ELSE EXCLUDED.venue_source END;

-- Categories and authors are replaced wholesale for every paper in the batch.
DELETE FROM paper_categories WHERE paper_id IN (SELECT id FROM stage_papers);
INSERT INTO paper_categories SELECT DISTINCT paper_id, category FROM stage_categories;
DELETE FROM paper_authors WHERE paper_id IN (SELECT id FROM stage_papers);
INSERT INTO paper_authors SELECT DISTINCT ON (paper_id, position) * FROM stage_authors;
"""


def _batches(papers: Iterable[Paper], size: int) -> Iterator[list[Paper]]:
    it = iter(papers)
    while batch := list(islice(it, size)):
        yield batch


def load_batch(conn: psycopg.Connection, batch: list[Paper]) -> None:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(_STAGE)
        with cur.copy("COPY stage_papers FROM STDIN") as copy:
            for p in batch:
                v = p.venue
                copy.write_row(
                    (
                        p.id,
                        p.title,
                        p.abstract,
                        p.doi,
                        p.categories[0],
                        p.published,
                        p.updated,
                        v and v.venue,
                        v and v.year,
                        v and v.raw,
                    )
                )
        with cur.copy("COPY stage_categories FROM STDIN") as copy:
            for p in batch:
                for c in p.categories:
                    copy.write_row((p.id, c))
        with cur.copy("COPY stage_authors FROM STDIN") as copy:
            for p in batch:
                for i, name in enumerate(p.authors):
                    copy.write_row((p.id, i, name))
        cur.execute(_UPSERT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--limit", type=int, default=None, help="stop after N ML papers")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    start = time.perf_counter()
    total = with_venue = 0
    with psycopg.connect(get_settings().database_url) as conn:
        for batch in _batches(iter_papers(args.path, args.limit), args.batch_size):
            load_batch(conn, batch)
            total += len(batch)
            with_venue += sum(p.venue is not None for p in batch)
            print(f"{total:>9,} papers  ({time.perf_counter() - start:,.0f}s)", flush=True)

    rate = with_venue / total if total else 0.0
    print(f"done: {total:,} papers, {with_venue:,} with a venue ({rate:.1%})")


if __name__ == "__main__":
    main()
