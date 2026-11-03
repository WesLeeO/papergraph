# M4: search baselines on LitSearch (dev)

**Question:** how much do keyword, dense and hybrid retrieval improve over the
Postgres full-text baseline from M3, and which embedding model should the
hybrid use?

**Setup.** LitSearch dev split (198 queries, hashed stratified split; test is
untouched), corpus of 64,183 papers in schema `bench_litsearch`. Every run is
from commit `9d4df6c`. The baseline run is clean; the other seven are flagged
`dirty` only because this report's draft (an untracked file under `reports/`)
existed while they ran. No code differed from `9d4df6c`, and the runner now
ignores `reports/` in its dirty check. Means with bootstrap 95% CIs; method
comparisons are paired (same queries) with bootstrap CIs on the difference and
a sign-flip permutation p-value (`make compare`). A difference whose CI contains
0 is reported as no difference.

## Results

| Method | Recall@10 | Recall@100 | nDCG@10 | MRR | p50 / p95 latency | Run |
|---|---|---|---|---|---|---|
| FTS baseline (`ts_rank`, M3) | 0.459 [0.389, 0.529] | 0.673 [0.607, 0.737] | 0.352 [0.293, 0.412] | 0.326 [0.269, 0.384] | 794 / 1384 ms | [20261103T153358Z_baseline-fts](../runs/20261103T153358Z_baseline-fts) |
| BM25 | 0.565 [0.496, 0.634] | 0.763 [0.703, 0.822] | 0.440 [0.379, 0.500] | 0.408 [0.348, 0.469] | 360 / 651 ms | [20261103T153639Z_bm25](../runs/20261103T153639Z_bm25) |
| Dense, bge-small | 0.579 [0.508, 0.647] | 0.730 [0.667, 0.790] | 0.450 [0.390, 0.510] | 0.416 [0.358, 0.476] | 19 / 25 ms | [20261103T153753Z_dense-bge](../runs/20261103T153753Z_dense-bge) |
| Dense, SPECTER2 | 0.392 [0.326, 0.460] | 0.685 [0.620, 0.748] | 0.251 [0.203, 0.302] | 0.221 [0.174, 0.270] | 18 / 34 ms | [20261103T153846Z_dense-specter2](../runs/20261103T153846Z_dense-specter2) |
| **Hybrid RRF, BM25 + bge-small** | **0.632** [0.564, 0.698] | **0.813** [0.758, 0.866] | **0.496** [0.436, 0.556] | **0.462** [0.402, 0.523] | 394 / 683 ms | [20261103T153902Z_hybrid-bm25-bge](../runs/20261103T153902Z_hybrid-bm25-bge) |
| Hybrid RRF, BM25 + SPECTER2 | 0.617 [0.549, 0.685] | 0.789 [0.731, 0.843] | 0.455 [0.397, 0.515] | 0.415 [0.357, 0.475] | 402 / 683 ms | [20261103T154037Z_hybrid-bm25-specter2](../runs/20261103T154037Z_hybrid-bm25-specter2) |

Ablations (dense bge-small):

| Variant | nDCG@10 | Paired diff vs dense bge-small | Run |
|---|---|---|---|
| Without the BGE query instruction | 0.454 | +0.004 [-0.018, +0.026], no difference | [20261103T153811Z_dense-bge-noinstr](../runs/20261103T153811Z_dense-bge-noinstr) |
| Exact search instead of HNSW | 0.464 | +0.015 [-0.001, +0.035], no difference | [20261103T153827Z_dense-bge-exact](../runs/20261103T153827Z_dense-bge-exact) |

## Paired comparisons

| A → B | Recall@10 | nDCG@10 | MRR | Verdict |
|---|---|---|---|---|
| FTS baseline → BM25 | +0.106 [+0.051, +0.162] | +0.088 [+0.047, +0.129] | +0.083 [+0.041, +0.125] | BM25 better (p ≤ 0.001) |
| BM25 → dense bge | +0.014 [-0.056, +0.083] | +0.010 [-0.045, +0.065] | +0.007 [-0.048, +0.064] | no difference |
| BM25 → hybrid | +0.067 [+0.015, +0.121] | +0.056 [+0.018, +0.094] | +0.053 [+0.015, +0.093] | hybrid better |
| dense bge → hybrid | +0.053 [+0.000, +0.109] | +0.046 [+0.008, +0.083] | +0.046 [+0.007, +0.084] | hybrid better on nDCG/MRR; Recall@10 CI touches 0 |
| dense bge → dense SPECTER2 | -0.187 [-0.255, -0.116] | -0.198 [-0.253, -0.143] | -0.195 [-0.252, -0.138] | bge better (p = 0.0001) |
| hybrid bge → hybrid SPECTER2 | -0.015 [-0.066, +0.035] | -0.041 [-0.086, +0.003] | -0.047 [-0.094, -0.002] | bge better on MRR only |
| FTS baseline → hybrid | +0.173 [+0.118, +0.231] | +0.143 [+0.103, +0.186] | +0.136 [+0.093, +0.181] | hybrid better (p = 0.0001) |

## By query type (nDCG@10, paired)

| Comparison | Broad (n = 51) | Specific (n = 147) |
|---|---|---|
| BM25 → dense bge | +0.086 [-0.006, +0.183] | -0.017 [-0.082, +0.050] |
| BM25 → hybrid | **+0.097** [+0.048, +0.153] | +0.041 [-0.006, +0.088] |
| dense bge → hybrid | +0.011 [-0.077, +0.092] | **+0.058** [+0.016, +0.099] |

Read with the facet sizes in mind: CIs on 51 broad queries are wide.

## Findings

1. **BM25 beats `ts_rank` clearly** (+0.088 nDCG@10). IDF (rare words weigh
   more), term-frequency saturation and partial length normalization are what
   `ts_rank` lacks. BM25's Lucene defaults (k1 = 1.2, b = 0.75) were already the
   best on dev: a small exploratory grid over k1, b and a title boost moved
   nDCG@10 by at most ±0.02, and no variant beat the defaults.
2. **Dense (bge-small) ties BM25 overall but fails differently.** Dense helps on
   broad, paraphrased questions; BM25 wins on rare, exact terms. On dev, each
   finds the right paper in the top 10 for about 25 queries where the other
   misses (top-10 oracle union: 0.707 vs 0.58 for either alone).
   Typical dense failures: recent jargon the model barely knows ("RLAIF",
   "DExperts"), words with two senses ("alignment" retrieved machine-translation
   word alignment), and topic matches that miss the specific claim.
   Dense wins on citation-derived (inline) questions; BM25 wins on
   expert-written (manual) ones.
3. **Hybrid RRF is the best method**: significantly better than BM25 on every
   metric and better than dense on nDCG@10, MRR and Recall@100. The gain over
   BM25 comes from broad queries, the gain over dense from specific ones:
   each half covers the other's blind spot. Recall@100 = 0.813 is the
   candidate pool for M5's reranker.
4. **SPECTER2 is worse than bge-small** here, alone (−0.198 nDCG@10) and in the
   hybrid. It was trained for paper-to-paper similarity from citations, and its
   ad-hoc query adapter for short keyword queries; LitSearch questions are long
   and specific. An exploratory check (not a reported run) found the paper
   adapter beat the query adapter for these queries (Recall@10 0.49 vs 0.40),
   still below bge-small. SPECTER2 vectors were verified to match AllenAI's
   `adapters` implementation exactly.
5. **Ablations:** the BGE query instruction makes no difference on these long
   questions; HNSW loses about 0.015 nDCG@10 to exact search (not significant),
   returning 97% of the exact top 10.

## Data notes

- 6,197 LitSearch papers (9.7%) have no title or abstract in the source data.
  5 of the 639 labelled papers are among them, so no text method can find them
  (a 0.8% ceiling). They are kept as distractors but excluded from embedding.
- Latency is for the dev runs on an M-series Mac with Postgres in Docker.
  Hybrid latency is almost all BM25, run sequentially before dense.

## Decision

The hybrid (BM25 + bge-small, RRF k = 60, equal weights, 100 candidates each)
is the first-stage retriever going into M5. RRF weights and k were not tuned.
