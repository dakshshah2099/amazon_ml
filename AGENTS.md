# AGENTS.md — Entity Resolution Challenge Playbook

> **Read this first.** This is the single source of truth for any agent working on this codebase.

---

## 🎯 Mission

Match business records from Source 2 and Source 3 to Source 1 (deduplicated reference).
**Metric: Macro F₀.₅** — precision weighted 2× over recall. False merges are ~4× worse than missed links. Singletons correctly predicted empty receive full 1.0 credit.

---

## ⚡ Critical Rules & Strict Compliance

1. **NO external APIs/data** — No geocoding, no business registries, no web lookup. Instant disqualification.
2. **≤8B params, MIT/Apache 2.0 license only.**
3. **OPEN-SET COUNTRY COMPLIANCE (ZERO HARDCODING)**:
   - Train has US and India; test adds France and potentially other unseen countries.
   - **NEVER hardcode country strings** (`US`, `India`, `France`, etc.) in any `if country == '...'`, regex branch, or one-hot encoding.
   - Treat `country` strictly as an open set of string labels.
   - Dynamic country partitioning MUST use `df.groupby('country', sort=False)`.
   - All address normalizations (postal codes, landmarks, road abbreviations) MUST run universally without country conditionals.
4. **Singletons = Free Points**:
   - Singletons represent 5–15%+ of reference entities.
   - Correctly predicting an empty match list (`""`) scores a perfect **1.0**.
   - Predicting a single false match on a singleton drops its score to **0.0**.
5. **Precision First**: When model confidence is uncertain, **DO NOT match**. Restraint maximizes $F_{0.5}$.
6. **Folder Structure Compliance**:
   - Solutions must strictly conform to the Amazon ML Challenge final submission directory hierarchy.

---

## 🏗️ Pipeline Architecture

```
Raw Records (S1, S2, S3)
    │
    ▼
[1. UNIVERSAL PREPROCESSING]
    • Unicode NFKD decomposition (strips accents/ligatures globally: é→e, ç→c, œ→oe)
    • URL/Domain cleaning (maurewilliamscolombier.com → maurewilliamscolombier)
    • Universal postal regex (\b[1-9]\d{2}\s?\d{3}\b|\b\d{5}(?:-\d{4})?\b)
    • Universal landmark removal (near, opp, behind, next to...)
    • Universal legal suffix canonicalization (INC, LLC, PVT_LTD, SARL, SAS...)
    • Universal road expansions (rd→road, st→street, ave→avenue, bd→boulevard...)
    │
    ▼
[2. OPEN-SET CANDIDATE BLOCKING] → candidate_pairs.tsv
    • Dynamic Country Partitioning: df.groupby('country', sort=False) (zero country lists)
    • Dual-Channel TF-IDF:
      - Channel A: Character 3-4 grams on Clean Name (top-20, threshold ≥ 0.35)
      - Channel B: Character 3-4 grams on Clean Address (top-10, threshold ≥ 0.40)
      - Bounded vocabulary: max_features=50000 for sub-second sparse operations
    • Fast Vectorized Union: Sparse COO matrix merge, capped at top-35 candidates/entity
    • Blocking Recall target: > 98.5%
    │
    ▼
[3. VECTORIZED PAIRWISE FEATURES]
    • Clean DataFrame merge on entity IDs (zero positional index mismatch)
    • RapidFuzz string metrics: Levenshtein, Jaro-Winkler, Token Sort, Token Set
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
    • matching_results.tsv (leaderboard) & candidate_pairs.tsv (audit)
    • Validated with student_resource/utils/validate_submission.py
    • Packaged with utils/create_submission_zip.py
```

---

## 📁 Official Submission Package Hierarchy

The submission archive and repository MUST adhere to this exact hierarchy:

```
amazon_ml/
├── output/
│   ├── matching_results.tsv        # Scored on leaderboard
│   └── candidate_pairs.tsv         # Candidate set audit
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── pipeline.ipynb      # Primary executable pipeline notebook
│       ├── README.md               # End-to-end reproduction guide
│       └── requirements.txt        # Pinned dependencies
├── Documentation_template.md       # Filled methodology write-up at root
├── dataset/                        # Junction/symlink -> student_resource/dataset
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
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
├── AGENTS.md                       # ← You are here (Playbook)
└── AGENT_GUIDE.md                  # Comprehensive technical guide
```

---

## 🔖 Versioning & Git Workflow

> [!IMPORTANT]
> **Every approach MUST have a versioned name and its own git branch.**

### Branch & Version Format
`approach/v{N}-{short-descriptive-name}`

### Examples
| Version | Branch | Description |
|---|---|---|
| `v1-tfidf-lgbm-baseline` | `approach/v1-tfidf-lgbm-baseline` | Open-set dual TF-IDF + RapidFuzz + GroupKFold LightGBM |
| `v2-dense-ann-blocking` | `approach/v2-dense-ann-blocking` | Add multilingual BGE-M3 / MiniLM FAISS blocking to v1 |
| `v3-deberta-reranker` | `approach/v3-deberta-reranker` | Cross-encoder pairwise scoring on top candidate pairs |
| `v4-hybrid-ensemble` | `approach/v4-hybrid-ensemble` | Ensemble GBDT + Cross-Encoder probabilities |

### Workflow
```bash
# 1. Start new approach from main
git checkout main
git checkout -b approach/v2-dense-ann-blocking

# 2. Iterate in code/business_entity_resolution/src/pipeline.ipynb
# 3. Log results in experiments/v2-dense-ann-blocking/ and update Results Tracker below
# 4. Commit and merge winner into main
```

---

## 📊 Results Tracker (Scoreboard)

Update this table after every approach.

| Version | Approach | Val F₀.₅ | Val Precision | Val Recall | Blocking Recall | Threshold | Notes |
|---|---|---|---|---|---|---|---|
| `v1` | `v1-tfidf-lgbm-baseline` | **0.1214** | 0.1616 | 0.0851 | **99.60%** | 0.46 | Open-set country-agnostic baseline; official validator PASS |

---

## 🚫 Hard Anti-Patterns (NEVER Do These)

- ❌ **NEVER hardcode country strings** (`if country in ['US', 'India', 'France']`, `if country == 'us'`). Country is an open string set.
- ❌ **NEVER use single-linkage / connected components clustering** — a single false link corrupts the entire cluster.
- ❌ **NEVER use random K-Fold on candidate pairs** — causes massive data leakage. ALWAYS use `GroupKFold` on `source1_entity_id`.
- ❌ **NEVER optimize for accuracy or log-loss** — always tune decision thresholds directly on macro $F_{0.5}$.
- ❌ **NEVER skip blocking recall measurement** — missed pairs at blocking can never be recovered. Target $>98.5\%$.
- ❌ **NEVER diverge from official folder structure** — root `output/`, `code/business_entity_resolution/src/pipeline.ipynb`, root `Documentation_template.md`.
- ❌ **NEVER commit large TSVs, models, or zip files to git** — keep `.gitignore` strictly enforced.
- ❌ **NEVER submit without running validator** — always verify exit code 0 via `validate_submission.py`.

---

## 🛠️ Essential Commands

```powershell
# 1. Run pipeline notebook
jupyter execute code/business_entity_resolution/src/pipeline.ipynb

# 2. Validate submission outputs locally
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

# 3. Build verified submission archive
python utils/create_submission_zip.py --team-name <your_team_name>
```
