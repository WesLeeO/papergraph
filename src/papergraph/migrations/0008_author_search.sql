-- Fuzzy author search. Paper text search (search_tsv) covers titles and
-- abstracts only, so "papers by Yann LeCun" needs the authors table. arXiv
-- spells names inconsistently ("Yann LeCun", "Y. LeCun", "Yann Lecun"), so
-- lookups use case-insensitive substring matching, made fast by a trigram
-- index (pg_trgm): it indexes every 3-letter chunk of each name, so
-- LIKE '%lecun%' doesn't have to scan millions of rows.
CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public;

CREATE INDEX paper_authors_name_trgm_idx
    ON paper_authors USING GIN (lower(name) public.gin_trgm_ops);
