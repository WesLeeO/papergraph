# M5: reranking on LitSearch (dev)

**Question:** how much does a cross-encoder reranker add on top of M4's best
first stage (hybrid RRF, BM25 + bge-small), and how fast can it be made
without losing that gain?

**Setup.** LitSearch dev split (198 queries), schema `bench_litsearch`. The
reranker reorders the first-stage top `top_n` candidates; papers below keep
their first-stage ranks, so Recall@100 is unchanged. Bootstrap 95% CIs; method
comparisons are paired against the plain hybrid (`make compare`). Latency is
per stage on an M-series Mac (MPS), p50. All runs are flagged `dirty`: they
were made before the reranker code was committed in `e535ff9`. Only its
defaults changed afterwards (model, top_n), and every run sets those
explicitly in its config. Treat them as dev evidence; final numbers come from
the test split in M10.

## Results

| Pipeline | nDCG@10 | MRR | Recall@10 | Rerank p50 | Total p50 | Run |
|---|---|---|---|---|---|---|
| Hybrid, no reranking (M4) | 0.496 [0.436, 0.556] | 0.462 [0.402, 0.523] | 0.632 | — | 402 ms | [20261103T153902Z_hybrid-bm25-bge](../runs/20261103T153902Z_hybrid-bm25-bge) |
| + bge-reranker-v2-m3 (568M), top 100 | 0.557 [0.496, 0.618] | 0.527 [0.465, 0.589] | 0.674 | 5,051 ms | 5,504 ms | [20261103T155337Z_hybrid-rerank-bge-m3](../runs/20261103T155337Z_hybrid-rerank-bge-m3) |
| + bge-reranker-v2-m3 (568M), top 20 | 0.556 [0.493, 0.618] | 0.532 [0.469, 0.595] | 0.656 | 1,068 ms | 1,466 ms | [20261103T161448Z_rerank20-bge-m3](../runs/20261103T161448Z_rerank20-bge-m3) |
| **+ mxbai-rerank-xsmall (71M), top 20** | **0.537** [0.477, 0.600] | 0.505 [0.443, 0.568] | 0.668 | **433 ms** | **906 ms** | [20261103T162237Z_rerank20-mxbai-xsmall](../runs/20261103T162237Z_rerank20-mxbai-xsmall) |
| + ms-marco-MiniLM-L6 (23M), top 20 | 0.488 [0.428, 0.548] | 0.450 [0.390, 0.511] | 0.641 | 90 ms | 513 ms | [20261103T162036Z_rerank20-minilm](../runs/20261103T162036Z_rerank20-minilm) |

## Paired differences vs the plain hybrid (nDCG@10)

| Reranker | Overall | Broad (n = 51) | Specific (n = 147) |
|---|---|---|---|
| bge-v2-m3, top 100 | **+0.062** [+0.024, +0.101] | +0.012 [-0.051, +0.077] | **+0.079** [+0.033, +0.127] |
| bge-v2-m3, top 20 | **+0.061** [+0.026, +0.097] | +0.024 [-0.040, +0.090] | **+0.073** [+0.031, +0.117] |
| mxbai-xsmall, top 20 | **+0.042** [+0.004, +0.082] | -0.026 [-0.087, +0.034] | **+0.065** [+0.019, +0.114] |
| MiniLM-L6, top 20 | -0.008 [-0.043, +0.027] | -0.039 [-0.112, +0.028] | +0.003 [-0.036, +0.044] |

mxbai-xsmall vs bge-v2-m3 (both top 20): -0.019 [-0.045, +0.009], no difference.

## Rerank depth (bge-reranker-v2-m3)

Derived exactly from the top-100 run: a cross-encoder scores each pair on its
own, so reranking only the top N equals the first-stage top N sorted by the
same scores. Rerank time is estimated as linear in N (the real top-20 run
measured 1,068 ms against an estimate of 1,010 ms).

| top_n | nDCG@10 | diff vs hybrid | Est. rerank time | Right paper in hybrid top N |
|---|---|---|---|---|
| 5 | 0.535 | +0.039 [+0.016, +0.064] | ~250 ms | — |
| 10 | 0.549 | +0.054 [+0.024, +0.083] | ~500 ms | 63.6% |
| 20 | 0.556 | +0.061 [+0.026, +0.097] | ~1,000 ms | 69.7% |
| 30 | 0.557 | +0.061 [+0.027, +0.097] | ~1,500 ms | 74.2% |
| 50 | 0.560 | +0.065 [+0.028, +0.104] | ~2,500 ms | 77.8% |
| 100 | 0.557 | +0.062 [+0.024, +0.101] | ~5,000 ms | 81.8% |

## Findings

1. **A cross-encoder adds about +0.06 nDCG@10 and MRR** over the hybrid,
   almost entirely on **specific** questions. Reading the question and the
   abstract together lets it check the exact claim ("anti-expert", "first
   showed"), which the bi-encoder blurs.
2. **Depth 20 keeps the whole gain at 1/5 of the cost.** The hybrid already has
   the right paper in its top 20 for 70% of questions; candidates further down
   rarely reach the top 10. A strong first stage is what makes reranking
   affordable.
3. **A 71M model is fast enough and nearly as good.** mxbai-rerank-xsmall at
   top 20 is significantly better than no reranking (+0.042 nDCG@10), 2.5x
   faster than bge-v2-m3, and its gap to bge-v2-m3 is not significant. Its MRR
   gain (+0.043 [-0.002, +0.089]) just misses significance.
4. **Small is fine; wrong training data is not.** MiniLM-L6 (web-search
   passages) gives no gain on research questions despite being the fastest.
5. **fp16 and batch size:** fp16 gave an identical ranking to fp32 (Spearman
   1.0 on a dev query) and ~30% more speed; batch 8 beat batch 32 on MPS.

## Decision

Default pipeline: hybrid (BM25 + bge-small, RRF) → mxbai-rerank-xsmall on the
top 20, about 0.9 s per query end to end, half of it BM25. The next latency
lever is BM25, not the reranker. bge-reranker-v2-m3 stays as the quality
reference. LLM reranking was skipped (API usage is billed separately from a
Claude plan).
