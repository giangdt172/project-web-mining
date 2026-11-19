# Data Directory Layout

All large raw dumps, intermediate Parquet files, monthly snapshots, and embeddings are gitignored.

- `raw/` — Stack Exchange XML dumps (`Posts.xml`, `Votes.xml`, `PostHistory.xml`, `Tags.xml`, `TagSynonyms.xml`).
- `interim/` — Stream-parsed Parquet tables filtered to the candidate tag cluster.
- `processed/` — Cleaned datasets, canonical tag cluster, initial `PostHistory` text views (BM25 + dense), fixed-horizon labels (`useful`, `responded`), cumulative vote arrays, and per-answer tag weights `w_{a,t}`.
- `snapshots/` — Dependency-cached monthly point-in-time snapshots (`S_M`), storing `q_tau`, sparse `E_sum`/`n_tau`/`E_prod` (`.npz`), CSR transition matrices, BM25 indexes, and cached PPR outputs.
- `embeddings/` — Date-sorted `question_embeddings.npy` (`float32`, L2-normalized) and `question_ids.npy`.
