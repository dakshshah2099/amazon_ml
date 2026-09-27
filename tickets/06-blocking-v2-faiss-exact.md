# 06 — Create `blocking_v2.py` with FAISS exact inner-product search

**What to build:** A new blocking script that replaces the old TF-IDF+SVD+HNSW `blocking.py` with embedding-based FAISS exact search (`IndexFlatIP`). Still partitions by country. Outputs the same two TSV formats as the old blocking (detailed + contest) so downstream scripts can consume them.

**Blocked by:** 05 (Embeddings exist as .npy files)

**Status:** ready-for-agent

## Detailed instructions

### Create file: `code/business_entity_resolution/src/blocking_v2.py`

The full script is provided in the spec. Key design decisions:

#### Why `IndexFlatIP` (exact) not HNSW (approximate)
- 384-dim is manageable for exact search at hundreds-of-thousands scale.
- The old pipeline's approximate-search-masquerading-as-exact was a known bug risk. Do not repeat it.
- An `--approx` flag is included as a fallback if exact proves too slow, but defaults to `False`.

#### Parameters (defaults to be updated after ticket 07 tuning)
- `k=20` — top-K nearest neighbors per query entity
- `min_score=0.55` — minimum cosine similarity to include a candidate
- These are starting guesses. Ticket 07 will empirically tune them.

#### Country partitioning
- S1 entities are matched against S2 and S3 entities within the same country.
- This is a correctness constraint from the contest data model, not a model feature.

#### Output format (must match for downstream compatibility)

**Detailed TSV** (`{split}_candidate_pairs_detailed.tsv`):
```
source1_entity_id\tcandidate_entity_id\tsource\tembed_score
```
Note: 4 columns, NOT the old 6 columns. The old `signal_type`, `name_score`, `addr_score` columns are gone because embedding blocking uses a single joint embedding, not separate name/addr signals.

**Contest TSV** (`train_candidate_pairs.tsv` or `candidate_pairs.tsv` for test):
```
source1_entity_id\tcandidate_entity_ids
```
Same format as before (comma-separated candidate IDs).

#### Implementation structure
```python
import os, time, argparse
import numpy as np
import pandas as pd
import faiss

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def run_blocking_v2(split='train', out_dir='output_v2', k=20, min_score=0.55, approx=False):
    # Load parquets for entity_id + country columns only
    # Load .npy embeddings and .npy id files for all 3 sources
    # Assert row counts match between parquet and embeddings
    # Assert ID order alignment: (ids_npy == df['entity_id'].to_numpy()).all()
    #
    # For each country:
    #   For each target source (S2, S3):
    #     Build FAISS IndexFlatIP (or IndexHNSWFlat if --approx)
    #     Search S1 embeddings against target source embeddings
    #     Filter by min_score, write to detailed TSV
    #
    # Write contest-format TSV

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--out-dir', default='output_v2')
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--min-score', type=float, default=0.55)
    parser.add_argument('--approx', action='store_true', default=False)
    args = parser.parse_args()
    run_blocking_v2(**vars(args))
```

See the spec for the complete implementation. Key details to not miss:
- Batch FAISS queries in chunks of 20,000 rows to control memory.
- Buffer TSV writes in 100,000-line batches.
- `del index` after each country/source pair to free memory.
- Contest TSV uses `set()` for candidate IDs (deduplication).

### Run and validate (train split only at this point)

```powershell
cd code/business_entity_resolution/src
python blocking_v2.py --split train --out-dir output_v2
```

Validate:
1. No assertion errors (embedding/parquet row alignment).
2. `output_v2/train_candidate_pairs_detailed.tsv` exists and has 4 columns.
3. `output_v2/train_candidate_pairs.tsv` exists and has 2 columns.
4. Spot check: `wc -l output_v2/train_candidate_pairs_detailed.tsv` shows a reasonable number of pairs (should be in the hundreds of thousands to low millions).

- [ ] `blocking_v2.py` created with `IndexFlatIP` as default
- [ ] `--approx` flag falls back to `IndexHNSWFlat` only when explicitly set
- [ ] Country partitioning implemented
- [ ] ID order alignment assertions present and passing
- [ ] Train split blocking completes without errors
- [ ] Detailed TSV has 4 columns: `source1_entity_id`, `candidate_entity_id`, `source`, `embed_score`
- [ ] Contest TSV has 2 columns: `source1_entity_id`, `candidate_entity_ids`
