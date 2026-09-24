# AGENTS.md — Entity Resolution Challenge Playbook

> **Read this first.** This is the single source of truth for any agent working on this codebase.

---

## 🎯 Mission

Match business records from Source 2/3 to Source 1 (deduplicated reference).
**Metric: F₀.₅** — precision 2× over recall. False merges are ~4× worse than missed links.

## ⚡ Critical Rules

1. **NO external APIs/data** — no geocoding, no business lookup, no internet augmentation. Disqualification.
2. **≤8B params, MIT/Apache 2.0 license only.**
3. **France is UNSEEN** — train has US + India; test adds France. Never hardcode country lists.
4. **Singletons = free points** — correctly predicting "no match" scores 1.0. One false merge = 0.0.
5. **When in doubt, DON'T match.** Restraint is rewarded.

---

## 🏗️ Pipeline (Every Approach Must Follow This Shape)

```
[1. Preprocess]  →  [2. Block]  →  [3. Score Pairs]  →  [4. Threshold + Post-process]  →  Output
     ↓                  ↓               ↓                        ↓
  Normalize         candidate_      feature vec             matching_
  names/addr        pairs.tsv       or model prob           results.tsv
```

### Stage 1 — Preprocessing
- Unicode NFKD (strip French diacritics: é→e, ç→c, œ→oe)
- Legal suffix extraction → canonical tag (PVT_LTD, SARL, INC, LLC…)
- Abbreviation expansion (Corp→Corporation, Rd→Road, &→and)
- Postal code extraction (US 5-digit, India 6-digit, France 5-digit)
- Landmark removal for Indian addresses

### Stage 2 — Blocking (candidate_pairs.tsv)
- **Goal: recall > 98%.** Missed pairs are irrecoverable.
- Hybrid multi-pass union is the winning strategy:
  - Rule-based: country + postal prefix + soundex
  - Sparse: TF-IDF char 3-4 grams → `sparse_dot_topn` top-K (threshold ~0.40)
  - Dense: sentence-transformer bi-encoder → FAISS HNSW top-K (cosine > 0.70)
- Cap at ~50 candidates per S1 entity after union + dedup

### Stage 3 — Pairwise Scoring
- **Option A (baseline):** ~30 handcrafted features → LightGBM
  - Name: Levenshtein, Jaro-Winkler, token sort/set, Jaccard, Monge-Elkan, Soft TF-IDF, char n-gram Jaccard
  - Phonetic: Soundex, Metaphone, NYSIIS match flags
  - Address: string metrics, postal exact/prefix match, building match
  - Suffix match/conflict, country match, embedding cosine
- **Option B (higher ceiling):** DeBERTa-v3-base cross-encoder (86M params)
- **Option C (hybrid):** Features + cross-encoder score → LightGBM ensemble

### Stage 4 — Threshold & Post-processing
- Tune threshold on **macro F₀.₅** via OOF predictions → expect optimal τ ≈ 0.72–0.85
- Singleton gate: if max(P(match)) < τ → empty list
- Veto rules: country mismatch → reject; postal prefix mismatch → reject
- Optional: reciprocal NN, top1–top2 margin filter

---

## 📐 Validation

```python
# GroupKFold by S1 entity — NEVER random KFold on pairs
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(data, groups=data['source1_entity_id']):
    ...
```

Always run before submit:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 🧰 Core Libraries

| Purpose | Library |
|---|---|
| String metrics | `rapidfuzz` |
| Phonetics | `jellyfish` |
| TF-IDF | `scikit-learn` |
| Sparse blocking | `sparse_dot_topn` |
| Dense blocking | `sentence-transformers` + `faiss-cpu` |
| LSH | `datasketch` |
| Classifier | `lightgbm` |
| Cross-encoder | `transformers` (DeBERTa-v3) |

---

## 🔖 Approach Versioning & Naming Convention

> [!IMPORTANT]
> **Every approach MUST have a versioned name and its own git branch.**
> This is non-negotiable. We need full traceability and easy rollback.

### Naming Format
```
v{N}-{short-descriptive-name}
```

### Examples
| Version | Branch | Description |
|---|---|---|
| `v1-tfidf-lgbm-baseline` | `approach/v1-tfidf-lgbm-baseline` | TF-IDF blocking + 30 features + LightGBM |
| `v2-dense-blocking` | `approach/v2-dense-blocking` | Add FAISS ANN blocking to v1 |
| `v3-deberta-cross-encoder` | `approach/v3-deberta-cross-encoder` | Replace LightGBM with DeBERTa-v3-base |
| `v4-hybrid-ensemble` | `approach/v4-hybrid-ensemble` | Features + DeBERTa score → LightGBM |
| `v5-french-tuning` | `approach/v5-french-tuning` | Country-specific normalization for France |

### Workflow
```bash
# Starting a new approach
git checkout main
git checkout -b approach/v2-dense-blocking

# Working on it...
git add -A && git commit -m "v2: add FAISS HNSW blocking, cosine>0.70"

# Recording results
# Update the results table in this file (see below)

# Merging winner into main
git checkout main
git merge approach/v2-dense-blocking
```

### Directory Structure Per Approach
```
amazon_ml/
├── AGENTS.md                    # ← You are here
├── AGENT_GUIDE.md               # Detailed reference
├── guides/                      # Deep-dive guides
│   ├── 01_blocking_strategies.md
│   ├── 02_feature_engineering.md
│   ├── 03_models_and_approaches.md
│   └── 04_competition_strategies.md
├── dataset/
│   ├── train/                   # train_source{1,2,3}.tsv + ground truth
│   └── test/                    # test_source{1,2,3}.tsv
├── src/                         # Active source code (current best approach)
│   ├── preprocess.py
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   ├── predict.py
│   └── config.py                # Hyperparams, thresholds, model paths
├── experiments/                  # Per-approach experiment logs & configs
│   ├── v1-tfidf-lgbm-baseline/
│   │   ├── config.yaml
│   │   ├── results.json         # {f05_val, precision, recall, threshold}
│   │   └── notes.md
│   └── v2-dense-blocking/
│       ├── config.yaml
│       ├── results.json
│       └── notes.md
├── output/                      # Submission files
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── utils/
│   └── validate_submission.py
└── requirements.txt
```

---

## 📊 Results Tracker

Update this table after every approach. **This is the scoreboard.**

| Version | Approach | Val F₀.₅ | Val Precision | Val Recall | Blocking Recall | Threshold | Notes |
|---|---|---|---|---|---|---|---|
| v1 | — | — | — | — | — | — | — |

---

## 🚫 Anti-Patterns (Hard Rules)

- ❌ Never use `connected_components` / single-linkage clustering — one false edge contaminates entire cluster
- ❌ Never random KFold on pairs — use GroupKFold by S1 entity
- ❌ Never hardcode `country in ['US', 'India']` — France is in test
- ❌ Never optimize for accuracy or log-loss — tune on macro F₀.₅
- ❌ Never skip blocking recall measurement — irrecoverable loss
- ❌ Never submit without running `validate_submission.py`
- ❌ Never use external APIs (geocoding, business lookup) — instant disqualification

---

## 📎 Reference Guides

For deep dives, see:
- [AGENT_GUIDE.md](file:///C:/Users/daksh/Desktop/amazon_ml/AGENT_GUIDE.md) — Full pipeline architecture & tech stack
- [01_blocking_strategies.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/01_blocking_strategies.md) — All blocking methods with code
- [02_feature_engineering.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/02_feature_engineering.md) — 30-feature vector, normalization, phonetics
- [03_models_and_approaches.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/03_models_and_approaches.md) — GBDT vs cross-encoder vs LLM
- [04_competition_strategies.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/04_competition_strategies.md) — F₀.₅ tuning, singletons, precision tricks
