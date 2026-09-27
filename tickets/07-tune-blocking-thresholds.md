# 07 — Tune blocking thresholds (`min_score`, `k`) against training recall

**What to build:** A tuning script that sweeps `min_score` over a grid using a sample of train S1 entities, reports recall per country/source at each threshold, and picks the optimal `min_score`. Then hardcode the chosen values back into `blocking_v2.py` defaults.

**Blocked by:** 06 (blocking_v2.py exists and train embeddings exist)

**Status:** ready-for-agent

## Detailed instructions

### Create file: `code/business_entity_resolution/src/tune_blocking_v2.py`

The script must:

1. Load ground truth from `dataset/train/train_ground_truth.tsv` (format: `source1_entity_id\tmatched_entity_ids` where matched IDs are comma-separated).

2. Sample 5,000 S1 entities that have ground truth matches (use `random.seed(42)` for reproducibility).

3. For each country and each target source (S2, S3):
   - Build a FAISS `IndexFlatIP` index from that source's embeddings (within that country).
   - Search sampled S1 embeddings against it with `k=30` (higher than production to measure recall ceiling).
   - For each `min_score` in `(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)`:
     - Count how many ground-truth matches are found in the top-K results at that threshold.
     - Print: `country={country} source=source{N} min_score={ms:.2f} k=30: recall={recall*100:.2f}% (n_true={n})`

4. The target: pick the **highest `min_score`** that holds **macro recall >= 99.5%** for **every** country/source combination. Recall is a hard floor — one weak segment silently caps the whole pipeline.

5. If no threshold in the grid clears 99.5% for some segment, extend the grid lower (0.25, 0.20) and re-run. Do not accept < 99.5% blocking recall.

### Key implementation details

```python
import csv, random
import numpy as np
import pandas as pd
import faiss

def load_gt(gt_path='dataset/train/train_ground_truth.tsv'):
    gt_map = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                gt_map[row[0].strip()] = {m.strip() for m in row[1].split(',') if m.strip()}
    return gt_map
```

- Use `IndexFlatIP` (exact search) for tuning — must match what blocking_v2.py uses by default.
- Load embeddings from the `.npy` files created by ticket 05.
- Load parquets only for `entity_id` and `country` columns.

### After tuning: update `blocking_v2.py`

Once you have the optimal `min_score` and `k`:

1. Update the default args in `blocking_v2.py`'s `argparse`:
   ```python
   parser.add_argument('--min-score', type=float, default=<TUNED_VALUE>)
   parser.add_argument('--k', type=int, default=<TUNED_VALUE>)
   ```

2. Add a comment in `blocking_v2.py` citing the measured recall:
   ```python
   # Tuned via tune_blocking_v2.py on 5000 train entities:
   # min_score=X.XX achieves >=99.5% recall for all country/source combinations
   # k=XX
   ```

### Then re-run blocking with tuned values

```powershell
python blocking_v2.py --split train --out-dir output_v2
```

- [ ] `tune_blocking_v2.py` created
- [ ] Recall grid printed for all country/source combinations
- [ ] Chosen `min_score` achieves >= 99.5% recall for every country/source pair
- [ ] `blocking_v2.py` defaults updated with tuned values + comment citing recall
- [ ] Train blocking re-run with tuned values completes successfully
