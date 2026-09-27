# 14 — Run test split end-to-end and generate submission

**What to build:** Execute the full pipeline on the test split: embed → block → predict → generate `output_v2/matching_results.tsv` for Unstop submission.

**Blocked by:** 12 (Train sanity check passes), 07 (Blocking thresholds tuned)

**Status:** ready-for-agent

## Detailed instructions

### Step 1: Embed test split

```powershell
cd code/business_entity_resolution/src
python embed_entities.py --split test --batch-size 256
```

Verify: 3 embedding .npy + 3 id .npy files created under `dataset/test/`.

### Step 2: Block test split

```powershell
python blocking_v2.py --split test --out-dir output_v2
```

Use the tuned `min_score` and `k` defaults (hardcoded into blocking_v2.py in ticket 07).

Verify:
- `output_v2/test_candidate_pairs_detailed.tsv` created (4-col)
- `output_v2/candidate_pairs.tsv` created (contest format, 2-col)

### Step 3: Predict test matches

```powershell
python predict_matches_v2.py --split test --out output_v2/matching_results.tsv
```

Verify:
- `output_v2/matching_results.tsv` created
- Has 2 columns: `source1_entity_id` and `matched_entity_ids`
- Every test S1 entity ID appears

### Step 4: Validate submission format

```python
import pandas as pd

# Load test S1 to get expected entity count
s1 = pd.read_parquet('dataset/test/test_source1_clean.parquet', columns=['entity_id'])
results = pd.read_csv('output_v2/matching_results.tsv', sep='\t')

print(f"Test S1 entities: {len(s1)}")
print(f"Result rows: {len(results)}")
assert len(results) == len(s1), "Row count mismatch — every S1 entity must appear"
print("Format OK")
```

### Step 5: Submit to Unstop

Submit `output_v2/matching_results.tsv` to Unstop and record the real test F0.5.

- [ ] Test embeddings generated (6 .npy files under dataset/test/)
- [ ] Test blocking completes (detailed + contest TSVs in output_v2/)
- [ ] Test prediction completes (output_v2/matching_results.tsv)
- [ ] Every test S1 entity appears in the results file
- [ ] File submitted to Unstop
