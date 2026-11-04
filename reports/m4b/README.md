# M4b: a stronger embedder (Qwen3-Embedding-0.6B) on LitSearch (dev)

**Question:** M4 picked bge-small (33M) as the dense half of the hybrid
because it beat SPECTER2, not because it was the best embedder available. Does
a current retrieval model do better, alone, in the hybrid, and in the full
product pipeline?

**Setup.** LitSearch dev split (198 queries), schema `bench_litsearch`, same
paper text as bge-small (title + abstract, 512 tokens max). Qwen3 queries get
a task instruction ("Given a research question, retrieve the titles and
abstracts of papers that answer it"), written once and not tuned. Bootstrap
95% CIs; comparisons are paired on the same queries (`make compare`). All
Qwen3 runs are flagged `dirty`: the encoder change in `papergraph.encoders`
was not committed yet when they ran (no other code differed from `6b4ad74`).
Treat them as dev evidence; final numbers come from the test split in M10.

Embedding the 57,986 LitSearch papers with text took 1 h 42 min on an M4 Pro
(~9 papers/s, vs ~120 for bge-small).

## Results

| Pipeline | Recall@10 | Recall@100 | nDCG@10 | MRR | Run |
|---|---|---|---|---|---|
| Dense, bge-small (M4) | 0.579 | 0.730 | 0.450 [0.390, 0.510] | 0.416 | [20261103T153753Z_dense-bge](../runs/20261103T153753Z_dense-bge) |
| **Dense, Qwen3-0.6B** | 0.702 | 0.846 | 0.562 [0.503, 0.621] | 0.526 | [20261104T110103Z_dense-qwen3-0_6b](../runs/20261104T110103Z_dense-qwen3-0_6b) |
| Hybrid BM25 + bge-small (M4) | 0.632 | 0.813 | 0.496 [0.436, 0.556] | 0.462 | [20261103T153902Z_hybrid-bm25-bge](../runs/20261103T153902Z_hybrid-bm25-bge) |
| Hybrid BM25 + Qwen3 | 0.693 | **0.881** | 0.560 [0.498, 0.619] | 0.527 | [20261104T110122Z_hybrid-bm25-qwen3-0_6b](../runs/20261104T110122Z_hybrid-bm25-qwen3-0_6b) |
| Current product: hybrid bge-small → mxbai @20 | 0.651 | 0.811 | 0.537 [0.476, 0.600] | 0.510 | [20261103T170554Z_product-maxdf01](../runs/20261103T170554Z_product-maxdf01) |
| Hybrid Qwen3 → mxbai @20 | 0.681 | 0.871 | 0.553 [0.493, 0.614] | 0.522 | [20261104T110306Z_product-qwen3-0_6b](../runs/20261104T110306Z_product-qwen3-0_6b) |
| Hybrid Qwen3 → bge-reranker-v2-m3 @20 | 0.689 | 0.871 | **0.571** [0.510, 0.631] | **0.542** | [20261104T110612Z_qwen3-hybrid-rerank-bge-m3](../runs/20261104T110612Z_qwen3-hybrid-rerank-bge-m3) |

## Paired comparisons (nDCG@10)

| A → B | Diff [95% CI] | Verdict |
|---|---|---|
| dense bge-small → dense Qwen3 | **+0.113** [+0.066, +0.161] | Qwen3 better (p = 0.0001); also +0.123 Recall@10 |
| hybrid bge-small → hybrid Qwen3 | **+0.064** [+0.030, +0.100] | Qwen3 better (p = 0.0003) |
| dense Qwen3 → hybrid Qwen3 | −0.003 [−0.044, +0.038] | no difference; Recall@100 +0.035 [+0.005, +0.068] |
| hybrid Qwen3 → + mxbai @20 | −0.006 [−0.042, +0.029] | no difference |
| hybrid Qwen3 → + bge-v2-m3 @20 | +0.011 [−0.024, +0.046] | no difference |
| current product → dense Qwen3 alone | +0.025 [−0.027, +0.076] | no difference |
| current product → hybrid Qwen3 + mxbai | +0.016 [−0.003, +0.039] | no difference; Recall@50/100 better |
| current product → hybrid Qwen3 + bge-v2-m3 | **+0.034** [+0.002, +0.064] | better (p = 0.029); Recall@10–100 all better |

## Findings

1. **The embedder was the biggest lever so far.** Swapping bge-small for
   Qwen3-0.6B adds +0.113 nDCG@10 to dense retrieval, more than BM25 → hybrid
   (+0.056) or adding a reranker (+0.042) did in M4/M5. On its own, Qwen3
   dense (0.562) already matches the whole current product (0.537) at 46 ms
   instead of ~1 s. The gain is on specific questions (+0.133); on broad ones
   it is not significant (+0.054 [−0.027, +0.131]).
2. **BM25 now mostly adds recall depth.** With bge-small, fusion was a clear
   win (+0.046 over dense). With Qwen3, fusion leaves nDCG@10 unchanged but
   raises Recall@100 to 0.881, the best of any method, which matters as the
   candidate pool for a reranker. Fusion helps expert-written queries
   (manual_acl +0.129) and hurts citation-derived ones (inline −0.06 to
   −0.12): equal RRF weights now let the weaker retriever pull good dense
   hits down. Weighted RRF is the obvious next knob (dev only).
3. **Rerankers stop paying once the first stage is strong.** mxbai-xsmall gave
   +0.042 on the bge-small hybrid but nothing on the Qwen3 hybrid; even
   bge-reranker-v2-m3 adds only +0.011 (not significant). A cross-encoder
   helps when it is better than the first stage at judging the top 20; a
   71M reranker no longer is.
4. **Only one Qwen3 pipeline is a significant win over the current product:**
   hybrid Qwen3 → bge-reranker-v2-m3 (+0.034 nDCG@10, better recall at every
   depth), at ~1.4 s p50 per query.

## Caveats

- 198 dev queries: CIs are about ±0.06 on means, ±0.03–0.05 on paired diffs.
- **Possible training-data overlap.** Qwen3-Embedding (2025) was trained on
  large mixed and synthetic retrieval data; LitSearch (2024) papers and
  questions may overlap with it. A second benchmark (SciDocs) or haystack mode
  would show whether the gain generalizes.
- Latency numbers depend on machine load (the current-product rerun measured
  1.45 s p50 against 0.9 s in M5); compare stages within one session.
- Product cost: embedding the 581k corpus with Qwen3 at ~9 papers/s is
  ~18 h on this Mac, vs ~1.3 h for bge-small. Vectors are 1024-d
  (4 KB each, ~2.4 GB for the corpus vs ~0.9 GB).

## Product configuration: BM25 pruning and fusion weights

Dropping the reranker exposed a cost hidden by it: the product prunes BM25
query terms (`max_df`) because unpruned BM25 takes ~2 s per query on the 581k
corpus (median of 8 queries; 4.5 s worst). With the reranker, pruning at 0.1
cost nothing; without it, the served hybrid scored 0.516, −0.043
[−0.065, −0.022] below the unpruned one
([20261104T111607Z_product-qwen3-hybrid](../runs/20261104T111607Z_product-qwen3-hybrid)).
A small dev grid over `max_df` and the dense weight in RRF:

| max_df | dense weight | nDCG@10 | Recall@100 | BM25 on 581k (median) | Run |
|---|---|---|---|---|---|
| 0.1 | 1 | 0.516 | 0.871 | ~180 ms | [20261104T111607Z](../runs/20261104T111607Z_product-qwen3-hybrid) |
| 0.1 | 2 | 0.551 | 0.859 | ~180 ms | [20261104T111808Z](../runs/20261104T111808Z_grid-maxdf01-w2) |
| 0.1 | 3 | 0.559 | 0.846 | ~180 ms | [20261104T111901Z](../runs/20261104T111901Z_grid-maxdf01-w3) |
| 0.2 | 1 | 0.542 | 0.866 | ~310 ms | [20261104T111950Z](../runs/20261104T111950Z_grid-maxdf02-w1) |
| **0.2** | **2** | **0.567** | **0.864** | ~310 ms | [20261104T112053Z](../runs/20261104T112053Z_grid-maxdf02-w2) |
| 0.2 | 3 | 0.573 | 0.846 | ~310 ms | [20261104T112155Z](../runs/20261104T112155Z_grid-maxdf02-w3) |
| 1.0 (none) | 1 | 0.560 | 0.881 | ~2,000 ms | [20261104T110122Z](../runs/20261104T110122Z_hybrid-bm25-qwen3-0_6b) |

Weighting dense higher recovers what pruning costs BM25. Weight 3 is not
better than 2 (differences within ±0.01) and changes what BM25 can do: with
k_rrf = 60, a paper ranked 1st by BM25 alone scores 1/61, less than dense's
100th at 3/160, so BM25 can only reorder dense's candidates (Recall@100 equals
dense alone, 0.846). At weight 2 it can still add papers dense missed.

## Decision

Product pipeline (`search.HYBRID`, `configs/product.yaml`): **BM25 (max_df
0.2) + Qwen3-Embedding-0.6B with weight 2, RRF, no reranker.** Re-run as
configured: [20261104T112328Z_product-qwen3-hybrid](../runs/20261104T112328Z_product-qwen3-hybrid),
nDCG@10 0.567 [0.507, 0.626], retrieval p50 266 ms on the benchmark.

Against the previous product (bge-small hybrid → mxbai @20), paired:
nDCG@10 +0.029 [−0.010, +0.069] (not significant), Recall@10 +0.052
[+0.007, +0.099], Recall@20 +0.080, Recall@50 +0.078, Recall@100 +0.053
[+0.009, +0.099] (all significant), at about a third of the latency.

The setting was picked as the best of six on dev, so its dev score is
slightly optimistic; the test split (M10) gives the unbiased number. The
full 581k corpus is being embedded with Qwen3 (~18 h, resumable); until it
finishes, semantic search covers only the newest papers embedded so far.
