"""Reference retrievers for checking the harness. Real retrieval lives in retrieval/.

Every retriever has the signature `(conn, query, k, **params) -> list[str]`:
paper ids, best first. The connection's search_path selects the corpus.
"""

import psycopg

# plainto_tsquery ANDs every word, so a long natural-language question matches
# almost nothing. Rewriting & to | ORs them instead; ts_rank then favours papers
# matching more (and title-weighted) terms. No IDF and no length normalization
# tuned for this corpus: that is what real BM25 adds. The cast back is a plain
# ::tsquery, not to_tsquery, so the already-stemmed lexemes aren't stemmed twice.
_PG_FTS = """
WITH q AS (
    SELECT replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')::tsquery AS tsq
)
SELECT p.id
FROM papers p, q
WHERE p.search_tsv @@ q.tsq
ORDER BY ts_rank(p.search_tsv, q.tsq, %(norm)s) DESC, p.id
LIMIT %(k)s
"""


def pg_fts(conn: psycopg.Connection, query: str, k: int, normalization: int = 0) -> list[str]:
    """Postgres full-text search over title + abstract, OR semantics.

    `normalization` is ts_rank's length-normalization bitmask: 0 ignores
    document length, 1 divides the rank by 1 + log(length), 2 by the length.
    Without it, long abstracts win just by containing more of the query words.
    """
    if not query.strip():
        return []
    params = {"q": query, "k": k, "norm": normalization}
    return [row[0] for row in conn.execute(_PG_FTS, params)]
