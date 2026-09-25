# Amazon ML Challenge — Business Entity Resolution: Agent Guide

## Problem Summary

**Task:** Match business records across 3 independent sources (S1=deduplicated reference, S2, S3) to resolve real-world entities.
**Metric:** Macro-averaged $F_{0.5}$ (Precision weighted 2× over Recall). Singletons score 1.0 when correctly predicted empty.
**Constraint:** ≤8B params, MIT/Apache 2.0 license, strictly NO external APIs or internet lookups.

## Key Facts

| Aspect | Detail |
|---|---|
| Fields | `entity_id`, `business_name`, `business_address`, `country` |
| Country Setting | Train: US, India. Test: +France (unseen). **Strict open-set string label handling (no hardcoded country conditionals)**. |
| Output | `output/matching_results.tsv` (scored) + `output/candidate_pairs.tsv` (audit) |
| Singletons | S1 entities with 0 matches → empty `matched_entity_ids` → scores 1.0. A single false merge drops score to 0.0. |
| Package Structure | `code/business_entity_resolution/src/pipeline.ipynb`, `requirements.txt`, `README.md`, `Documentation_template.md`, `output/` |

---

## Pipeline Architecture

```
Raw Records (S1, S2, S3)
    │
    ▼
[1. UNIVERSAL PREPROCESSING (COUNTRY-AGNOSTIC)]
    • Unicode NFKD decomposition (strips international accents/ligatures globally: é→e, ç→c, œ→oe)
    • URL / domain cleaning (maurewilliamscolombier.com → maurewilliamscolombier)
    • Universal postal code extraction: \b[1-9]\d{2}\s?\d{3}\b|\b\d{5}(?:-\d{4})?\b
    • Universal landmark removal: near, opp, behind, next to...
    • Universal legal suffix canonicalization: INC, LLC, PVT_LTD, SARL, SAS, SA, LLP...
    • Universal road expansion: rd→road, st→street, ave→avenue, bd→boulevard, rte→route...
    │
    ▼
[2. OPEN-SET CANDIDATE BLOCKING] → candidate_pairs.tsv
    • Dynamic Country Partitioning: df.groupby('country', sort=False) (zero country lists)
    • Dual-Channel TF-IDF (Bounded vocabulary: max_features=50000):
      - Channel A: Character 3-4 grams on Clean Name (top-20, threshold ≥ 0.35)
      - Channel B: Character 3-4 grams on Clean Address (top-10, threshold ≥ 0.40)
    • Fast Sparse COO Union: Merge and deduplicate in sub-seconds; capped at top-35 candidates/entity
    • Blocking Recall target: > 98.5%
    │
    ▼
[3. PAIRWISE FEATURE ENGINEERING]
    • Clean DataFrame merge on entity IDs (robust against indexing mismatches)
    • RapidFuzz string metrics: Levenshtein, Jaro-Winkler, Token Sort, Token Set
    • Address metrics: Normalized Levenshtein, Jaro-Winkler, Token Set
    • Structural features: exact postal match, postal 3-prefix match, building number match
    • Suffix features: legal suffix match (1.0), legal suffix conflict (1.0)
    • Retrieval signals: Name TF-IDF similarity, Address TF-IDF similarity
    │
    ▼
[4. CLASSIFIER & MACRO F₀.₅ OPTIMIZATION]
    • 5-Fold GroupKFold strictly grouped by source1_entity_id (zero leakage)
    • LightGBM GBDT with class imbalance compensation
    • Threshold optimization sweep over OOF probabilities maximizing exact macro F₀.₅
    • Strict singleton preservation gate (max P < τ → empty match set)
    │
    ▼
[5. SUBMISSION & PACKAGING]
    • output/matching_results.tsv (leaderboard) & output/candidate_pairs.tsv (blocking audit)
    • Validated with student_resource/utils/validate_submission.py
    • Packaged with utils/create_submission_zip.py
```

---

## Critical Strategy Principles

1. **Open-Set Country Compliance (No Hardcoded Countries)**:
   The test set introduces unseen jurisdictions (France) and must be treated as an open set of string labels. Never write `if country in ['US', 'India']` or `elif country == 'France'`. Normalization and blocking must be completely country-agnostic.
2. **Precision Over Recall ($F_{0.5}$)**:
   False merges are penalized ~4× more than false negatives. When model confidence is below the optimal threshold, default to singleton (empty match list).
3. **Singletons are High Value**:
   Over 5–15%+ of S1 reference entities are singletons. Predicting an empty match list correctly awards 1.0 points. One false link on a singleton collapses that entity's score to 0.0.
4. **Dual-Channel Blocking is Essential**:
   Name-only blocking achieves only ~82% recall because real-world records often feature noisy names or web URLs with matching addresses. Address-only blocking fails when addresses are missing (`NaN`). Combining both channels via sparse COO union achieves **>99.6% recall**.
5. **No External APIs or Pre-trained Models >8B**:
   Offline execution only. No commercial ER APIs, no external geocoding, and no business registries.

---

## Directory Hierarchy & Submission Specification

```
amazon_ml/
├── output/
│   ├── matching_results.tsv        # Scored on leaderboard
│   └── candidate_pairs.tsv         # Candidate set audit
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── pipeline.ipynb      # Main end-to-end pipeline notebook
│       ├── README.md               # Execution & reproduction guide
│       └── requirements.txt        # Pinned dependencies
├── Documentation_template.md       # Filled methodology write-up at root
├── dataset/                        # Junction/symlink -> student_resource/dataset
│   ├── train/
│   └── test/
├── student_resource/               # Provided challenge utilities & README
│   └── utils/
│       └── validate_submission.py
├── utils/
│   └── create_submission_zip.py    # Automated packaging & validation CLI
├── experiments/                    # Versioned approach logs & configs
│   └── v1-tfidf-lgbm-baseline/
│       ├── config.yaml
│       ├── results.json
│       └── notes.md
├── AGENTS.md                       # Playbook & scoreboard
└── AGENT_GUIDE.md                  # This file
```

---

## Detailed Technique Guides

| Guide | Path | Focus |
|---|---|---|
| Blocking | [guides/01_blocking_strategies.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/01_blocking_strategies.md) | Equi-blocking, Sparse TF-IDF top-K, Dense ANN (FAISS), MinHash LSH |
| Features | [guides/02_feature_engineering.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/02_feature_engineering.md) | RapidFuzz metrics, universal normalization, phonetics, structural features |
| Models | [guides/03_models_and_approaches.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/03_models_and_approaches.md) | GBDT (LightGBM/CatBoost), DeBERTa-v3 cross-encoder, Bi-encoders, LLMs |
| Competition | [guides/04_competition_strategies.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/04_competition_strategies.md) | Macro F₀.₅ tuning, singleton handling, precision veto rules, pitfalls |

---

## Recommended Tech Stack

| Category | Library | Purpose |
|---|---|---|
| String Metrics | `rapidfuzz` | C++ accelerated Levenshtein, Jaro-Winkler, token sort/set |
| Phonetics | `jellyfish` | Metaphone, NYSIIS, Soundex |
| TF-IDF & Vectorization | `scikit-learn` | Character n-gram vectorization with bounded vocabulary |
| Sparse Matrix Math | `sparse_dot_topn`, `scipy` | Sub-second sparse matrix multiplication and COO union |
| Classifier | `lightgbm` | Fast GBDT pairwise scoring with GroupKFold cross-validation |
| Preprocessing | `unicodedata`, `re` | Unicode NFKD decomposition, international regex patterns |
| Packaging & Validation | `zipfile`, `validate_submission.py` | Official submission packaging and local validation gate |
