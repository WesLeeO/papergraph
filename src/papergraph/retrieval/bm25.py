"""BM25 keyword retrieval over our own inverted index in Postgres.

    score(q, d) = sum over query terms t of
                  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))

- idf: rare terms count more. Lucene's variant, ln(1 + (N - df + 0.5) / (df + 0.5)),
  which is never negative (the classic one is, for terms in > half the corpus).
- k1 (default 1.2): tf saturation. The first occurrences of a term matter most;
  k1 controls how fast extra occurrences stop adding score.
- b (default 0.75): length normalization. 0 ignores length, 1 fully divides by it.
- title_boost (default 1.0 = plain BM25): title occurrences count `title_boost`
  times, and so do title tokens in the length. A simplified BM25F.

Terms are Postgres's English stems (the same tokenizer as papers.search_tsv), so
"attention" and "attentive" both match `attent`, and stop words are dropped.

Build the index after loading or changing a corpus:
    python -m papergraph.retrieval.bm25 --schema bench_litsearch
"""

import argparse
import math
import time
from collections.abc import Iterable, Mapping
from typing import Any

import psycopg
from psycopg import sql

from papergraph.db import connect
from papergraph.retrieval.filters import Filters, filter_sql

K1 = 1.2
B = 0.75


def idf(df: int, n_docs: int) -> float:
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def term_score(
    tf: float, df: int, n_docs: int, doc_len: float, avg_len: float, k1: float, b: float
) -> float:
    """BM25 contribution of one query term to one document."""
    if tf <= 0:
        return 0.0
    norm = k1 * (1 - b + b * doc_len / avg_len)
    return idf(df, n_docs) * tf * (k1 + 1) / (tf + norm)


def score_docs(
    query_terms: Iterable[str],
    docs: Mapping[str, Mapping[str, tuple[int, int]]],
    *,
    k1: float = K1,
    b: float = B,
    title_boost: float = 1.0,
) -> dict[str, float]:
    """Reference BM25 in pure Python: doc id -> score, for docs matching any term.

    `docs` maps doc id -> {term: (tf_title, tf_body)} over the WHOLE corpus
    (corpus statistics come from it). Slow, but obviously correct: the tests
    and the SQL consistency check compare `search` against it.
    """
    n_docs = len(docs)

    def length(terms: Mapping[str, tuple[int, int]]) -> float:
        return sum(title_boost * t + body for t, body in terms.values())

    avg_len = sum(length(d) for d in docs.values()) / n_docs
    terms = set(query_terms)
    df = {t: sum(t in d for d in docs.values()) for t in terms}
    scores: dict[str, float] = {}
    for doc_id, doc in docs.items():
        matched = terms & doc.keys()
        if not matched:
            continue
        doc_len = length(doc)
        scores[doc_id] = sum(
            term_score(title_boost * doc[t][0] + doc[t][1], df[t], n_docs, doc_len, avg_len, k1, b)
            for t in matched
        )
    return scores


# --- index build --------------------------------------------------------------

_BUILD = """
TRUNCATE bm25_postings, bm25_terms, bm25_stats;

-- unnest(tsvector) gives one row per (paper, stem) with the stem's positions and
-- their weights: 'A' = title, 'B' = abstract (see papers.search_tsv).
-- Inserted in term order so each term's postings are contiguous on disk: a
-- query then reads a few pages per term instead of one page per posting.
INSERT INTO bm25_postings (term, paper_id, tf_title, tf_body, len_title, len_body)
WITH tf AS (
    SELECT p.id, u.lexeme,
           cardinality(array_positions(u.weights, 'A')) AS tf_title,
           cardinality(u.weights) - cardinality(array_positions(u.weights, 'A')) AS tf_body
    FROM papers p, unnest(p.search_tsv) u
)
SELECT lexeme, id, tf_title, tf_body,
       sum(tf_title) OVER (PARTITION BY id), sum(tf_body) OVER (PARTITION BY id)
FROM tf
ORDER BY lexeme, id;

INSERT INTO bm25_terms (term, df)
SELECT term, count(*) FROM bm25_postings GROUP BY term;

-- Averages over ALL papers: a paper with no indexed terms has length 0.
INSERT INTO bm25_stats (n_docs, avg_len_title, avg_len_body)
SELECT n.n_docs,
       coalesce(sum(l.len_title), 0)::real / n.n_docs,
       coalesce(sum(l.len_body), 0)::real / n.n_docs
FROM (SELECT count(*) AS n_docs FROM papers) n
LEFT JOIN (SELECT DISTINCT paper_id, len_title, len_body FROM bm25_postings) l ON TRUE
GROUP BY n.n_docs;
"""


def build_index(conn: psycopg.Connection) -> None:
    """Rebuild the BM25 tables from papers.search_tsv, atomically."""
    with conn.transaction():
        conn.execute(_BUILD)
    # Fresh planner statistics: the query plan depends on how many postings
    # a term has, and the tables were just rewritten.
    conn.execute("ANALYZE bm25_postings, bm25_terms, bm25_stats")


# --- search -------------------------------------------------------------------

_SEARCH = """
WITH s AS (
    SELECT n_docs, %(boost)s * avg_len_title + avg_len_body AS avg_len FROM bm25_stats
),
q_all AS (
    -- Distinct stems of the query, from the same tokenizer as the index.
    SELECT t.term, t.df, row_number() OVER (ORDER BY t.df) AS rarity
    FROM unnest(tsvector_to_array(to_tsvector('english', %(q)s))) AS u(term)
    JOIN bm25_terms t ON t.term = u.term
),
q AS (
    -- Query-term pruning: skip very common terms (low IDF, huge posting
    -- lists), but always keep the two rarest so no query ends up empty.
    SELECT q_all.term FROM q_all, s
    WHERE q_all.rarity <= 2 OR q_all.df <= %(max_df)s * s.n_docs
),
m AS (
    SELECT p.paper_id,
           ln(1 + (s.n_docs - t.df + 0.5) / (t.df + 0.5)) AS idf,
           %(boost)s * p.tf_title + p.tf_body AS tf,
           %(boost)s * p.len_title + p.len_body AS len,
           s.avg_len
    FROM q
    JOIN bm25_terms t USING (term)
    JOIN bm25_postings p USING (term)
    CROSS JOIN s
    {filter_join}
)
SELECT paper_id
FROM m
GROUP BY paper_id
ORDER BY sum(idf * tf * (%(k1)s + 1)
             / (tf + %(k1)s * (1 - %(b)s + %(b)s * len / avg_len))) DESC,
         paper_id
LIMIT %(k)s
"""


def search(
    conn: psycopg.Connection,
    query: str,
    k: int,
    k1: float = K1,
    b: float = B,
    title_boost: float = 1.0,
    filters: Filters | None = None,
    max_df: float = 1.0,
) -> list[str]:
    """Top-k paper ids by BM25. Papers sharing no stem with the query are never returned.

    max_df: skip query terms found in more than this fraction of papers (the
    two rarest terms are always kept). 1.0 = exact BM25; ~0.1 trades a little
    precision for speed on large corpora, where common terms dominate the work.
    """
    params: dict[str, Any] = {
        "q": query,
        "k": k,
        "k1": k1,
        "b": b,
        "boost": title_boost,
        "max_df": max_df,
    }
    filter_join: sql.Composable = sql.SQL("")
    if filters is not None and not filters.empty:
        condition, filter_params = filter_sql(filters, "pp")
        filter_join = sql.SQL("JOIN papers pp ON pp.id = p.paper_id WHERE {}").format(condition)
        params.update(filter_params)
    statement = sql.SQL(_SEARCH).format(filter_join=filter_join)
    return [row[0] for row in conn.execute(statement, params)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--schema", default="public", help="corpus namespace to index")
    args = parser.parse_args()
    start = time.perf_counter()
    with connect(args.schema, autocommit=True) as conn:
        build_index(conn)
        row = conn.execute(
            "SELECT (SELECT count(*) FROM bm25_postings), (SELECT count(*) FROM bm25_terms), "
            "n_docs, avg_len_title + avg_len_body FROM bm25_stats"
        ).fetchone()
    assert row is not None
    postings, terms, n_docs, avg_len = row
    print(
        f"bm25 index for {args.schema}: {n_docs:,} papers, {terms:,} terms, "
        f"{postings:,} postings, avg length {avg_len:.1f} tokens "
        f"({time.perf_counter() - start:.0f}s)"
    )


if __name__ == "__main__":
    main()
