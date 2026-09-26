# Running the Pipeline on Google Colab (Beginner Walkthrough)

## Overview

This guide explains why the `approach/v2-precision-engineered-gbdt` pipeline ran for 9+ hours
without finishing on a previous attempt, what was fixed on the `perf/fix-slow-pipeline` branch,
and gives copy-paste steps to run it in Google Colab from scratch.

---

## 1. What was actually causing the 9-hour run

The code was profiled locally against the real dataset (`train_source1/2/3.tsv`, ~2.2M / 5M / 5.3M
rows). At 3,000 and 15,000-record validation samples, every phase scaled roughly **linearly**
(candidate generation: 14.65s → 25.38s for a 4.9× bigger US bucket; feature engineering: 13.78s →
67.18s for 5.1× more pairs). Extrapolated to 50,000 records, the whole `run_50k_val.py` script
should complete in well under an hour on a normal machine — there is no infinite loop and no
quadratic blow-up at that sample size. That means the 9-hour hang was almost certainly caused by
the **environment**, not a logic bug, plus a couple of real (now-fixed) inefficiencies:

1. **No checkpointing anywhere.** If the notebook/session disconnected at hour 8, *all* progress
   (candidate generation, features, trained models) was lost — there was nothing to show for the
   9 hours. This is the single biggest practical problem: even a "just slow" run becomes
   "never completes" if a Colab disconnect wipes it. **Fixed**: both scripts now checkpoint to
   `experiments/checkpoints/` (candidate pairs after blocking, per-country progress for full test
   inference) and resume automatically on re-run.
2. **Google Drive I/O.** If the dataset is read directly from a Google Drive mount
   (`/content/drive/...`), every chunked `pd.read_csv` call turns into slow network file access.
   Locally this candidate-scan step takes ~90s; over Drive it can take 10-50x longer. **Fix**:
   the guide below always copies the dataset to local Colab disk (`/content/dataset`) before
   running anything.
2. **Looser blocking parameters in `run_50k_val.py`.** It used `top_n=45/25`, `threshold=0.20/0.25`,
   `cap=65` — noticeably wider than the already-validated, faster, and *more accurate* config in
   `experiments/v2-precision-engineered-gbdt/config.yaml` (`top_n=35/20`, `threshold=0.25/0.30`,
   `cap=50`, which scored 0.9846 vs ~0.98). **Fixed**: `run_50k_val.py` now uses the tuned config,
   cutting candidate-pair volume (and every downstream phase) by roughly a third.
3. **No memory cleanup per country** in `run_50k_val.py` (unlike `run_test_inference.py`, which
   already had `gc.collect()`/`del`). On a RAM-limited Colab instance this can cause swapping,
   which looks exactly like "stuck for hours". **Fixed**: added `del`/`gc.collect()` after each
   country's blocking pass.
4. **Only 2 countries in the whole dataset** (`US` ~60%, `India` ~40%). The "country-partitioned
   blocking" barely reduces the search space — you effectively get two large buckets, not many
   small ones. This is expected and fine at 50k scale, but it's the reason the **full test-set**
   run (`utils/run_test_inference.py`, ~1.7M+ candidate rows) is a fundamentally bigger job than
   the 50k validation script — budget hours, not minutes, for that one, and rely on the new
   per-country checkpoint/resume to survive disconnects.
5. Minor: ground-truth parsing used a `.iterrows()` Python loop — vectorized now, no behavior change.

---

## 2. What changed (branch: `perf/fix-slow-pipeline`)

| File | Change |
|---|---|
| `experiments/run_50k_val.py` | Vectorized ground-truth parsing; tightened blocking params to the tuned config; `gc.collect()`/`del` per country; **checkpoints candidate pairs to disk and resumes**; `DATA_DIR`/`OUTPUT_DIR`/`CKPT_DIR`/`VAL_SAMPLE_SIZE` are now env-var overridable; prints total wall-clock time. |
| `utils/run_test_inference.py` | `DATA_DIR`/`OUTPUT_DIR`/`MODEL_PATH`/`CKPT_DIR` env-var overridable; **saves a checkpoint after every completed country and skips already-done countries on re-run**. |

No modeling/feature logic changed — only speed, memory, and durability. Locally, a 15,000-record
run completed in 5m52s end-to-end (see commit for the exact numbers); 50,000 should be well under
an hour on a normal machine, and Colab should now be in the tens-of-minutes range once the dataset
is on local disk.

---

## 3. Step-by-step: Google Colab setup (first time)

You have a Google AI Pro subscription — when you open Colab signed into that account, pick
**Runtime → Change runtime type → CPU (or "High-RAM" if offered)**. This workload is CPU + RAM
bound (TF-IDF + LightGBM), **not GPU-bound**, so you do not need a GPU/TPU runtime here.

### Step 1 — Put the dataset and code somewhere Colab can reach

Easiest path: zip the `amazon_ml` folder (with its `dataset/` populated, or the dataset zip
separately) and upload it to your Google Drive, e.g. `MyDrive/amazon_ml_challenge/`.

### Step 2 — Open a new Colab notebook

Go to https://colab.research.google.com → **New notebook**.

### Step 3 — Mount Drive (run in a cell)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Follow the popup to authorize.

### Step 4 — Copy the dataset to LOCAL Colab disk (important — do not skip this)

```python
import shutil, os
os.makedirs('/content/amazon_ml', exist_ok=True)
# Adjust the source path to wherever you uploaded things in Drive
shutil.copytree('/content/drive/MyDrive/amazon_ml_challenge/amazon_ml', '/content/amazon_ml', dirs_exist_ok=True)
```

This step alone is likely the biggest speed fix versus the previous 9-hour run: reading a 480MB
TSV directly from a mounted Drive is far slower than reading it from Colab's local SSD.

### Step 5 — Install dependencies

```python
%cd /content/amazon_ml
!pip install -q -r code/business_entity_resolution/requirements.txt
```

If that requirements file ever fails to resolve, install the essentials directly:

```python
!pip install -q pandas numpy scipy scikit-learn rapidfuzz lightgbm joblib sparse-dot-topn pyarrow
```

### Step 6 — Point the scripts at your local copy and run the 50k validation

```python
import os
os.environ['DATA_DIR'] = '/content/amazon_ml/dataset'
os.environ['OUTPUT_DIR'] = '/content/amazon_ml/output'
# Keep checkpoints on Drive so they survive a disconnect even if /content is wiped
os.environ['CKPT_DIR'] = '/content/drive/MyDrive/amazon_ml_challenge/checkpoints'
os.environ['VAL_SAMPLE_SIZE'] = '50000'

!python experiments/run_50k_val.py
```

Watch the output — it now prints a line per phase (`PHASE 1/2/3/4`) with elapsed seconds, plus a
`Checkpoint saved: ...` line after blocking. If your session disconnects and you reconnect and
re-run the exact same cell, it will print `Found checkpoint, skipping blocking` and skip straight
to feature engineering + training instead of starting over.

Expected result: `Gate Target (>= 98%)` line and a `Total wall-clock time` print at the end.

### Step 7 — (Optional) Full test-set inference

This is a much bigger job (the full test set, ~1.7M+ S1 entities). Only run this after the 50k
validation looks good, and expect it to take a long time regardless — it's inherently a bigger
computation, not a bug. Use the same env-var pattern:

```python
os.environ['DATA_DIR'] = '/content/amazon_ml/dataset'
os.environ['OUTPUT_DIR'] = '/content/drive/MyDrive/amazon_ml_challenge/output'  # write results to Drive
os.environ['CKPT_DIR'] = '/content/drive/MyDrive/amazon_ml_challenge/checkpoints'

!python utils/run_test_inference.py
```

It now checkpoints **after every country** to `CKPT_DIR`. If Colab disconnects you, just re-mount
Drive, re-run Steps 3-5, then re-run this cell — it will print
`Resuming from checkpoint: N countries already completed` and skip straight to the remaining ones.

### Step 8 — Keep the session alive / avoid disconnects

- Don't close the browser tab; keep interacting with the notebook occasionally (Colab free/Pro
  disconnects idle sessions).
- Colab Pro/Pro+ gives longer max session length and (on some tiers) background execution — check
  under the "Runtime" menu / the Colab Pro badge in the top-right for what's available on your
  account.
- Because checkpoints now live on Drive, a disconnect is no longer fatal — just reconnect and
  re-run the same cell.

### Step 9 — Download your results

```python
from google.colab import files
files.download('/content/amazon_ml/output/matching_results.tsv')
files.download('/content/amazon_ml/output/candidate_pairs.tsv')
```

(Or just open them from Drive if you pointed `OUTPUT_DIR` there.)

---

## 4. Quick sanity check before a long run

Run a tiny sample first to make sure everything is wired correctly before committing to a 50k or
full-test run:

```python
os.environ['VAL_SAMPLE_SIZE'] = '2000'
!python experiments/run_50k_val.py
```

This should finish in roughly a minute or two. If it doesn't, something environmental (Drive I/O,
missing package, wrong `DATA_DIR`) is wrong — fix that before scaling up, rather than waiting
hours to find out.
