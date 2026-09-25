**Team Name:** Kya_hi_kare
**Team Members:** Daksh Shah, Mahir Shah, Hemil Shah, Ansh Patel
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed an open-set, country-agnostic Entity Resolution (ER) pipeline engineered specifically for the precision-weighted macro $F_{0.5}$ metric. The architecture integrates universal multi-jurisdiction preprocessing, open-set country-partitioned dual-channel sparse TF-IDF blocking (achieving 99.55% recall on evaluation matches), vectorized pairwise lexical/structural feature extraction via RapidFuzz with structural conflict vetoes, a 5-Fold GroupKFold LightGBM classifier, and a mathematical 1-to-1 argmax assignment post-processor. On rigorous 5-fold cross-validation, the solution achieves **0.9846 Macro $F_{0.5}$** with **99.15% Precision** and **96.95% Recall**.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across ~12.5M train and ~11.7M test records, we identified critical structural patterns:
- **Intra-Country Invariance**: 100% of validated ground-truth matches occur strictly within the same country label. Partitioning dynamically by country reduces comparison space by 50–70% with zero recall loss.
- **Open-Set Country Challenge**: Training data comprises US and India, while the test set introduces unseen jurisdictions (France). Normalization and blocking must avoid hardcoded country conditionals.
- **Uniqueness Invariant (1-to-1 Assignment)**: Ground truth validation across all 7.6M+ links confirmed that 0 candidate records are assigned to multiple Source 1 entities. Every Source 2/3 record belongs to at most one reference cluster. Any multi-assignment creates guaranteed false positives.
- **Asymmetric Noise Modalities**: Empirical inspection revealed two dominant error patterns:
  1. *Name corruption with clean address*: e.g. web URLs, brand truncation, or OCR errors, while address text remains near-identical.
  2. *Missing address with clean name*: Records where address is null/NaN, but business name contains high lexical overlap. Single-channel blocking on concatenated text fails on these edge cases.
- **Metric Asymmetry & Singletons**: Under macro $F_{0.5}$, precision is weighted 2× over recall, making false merges ~4× more damaging than false negatives. Singletons (entities with 0 matches) represent a substantial portion of reference entities; predicting an empty match list correctly awards a full 1.0 score, while a single false merge collapses the entity score to 0.0.

### 2.2 Solution Strategy
- **Approach Type**: Multi-Stage Hybrid Pipeline (Country-Partitioned Dual Sparse TF-IDF Blocking + Conflict-Aware GBDT + 1-to-1 Argmax Post-Processing).
- **Core Innovation**:
  1. *Universal Country-Agnostic Normalization*: International postal code regex, global prepositional landmark cleaning, and Unicode NFKD decomposition running uniformly across all jurisdictions.
  2. *Dual-Channel Sparse TF-IDF*: Independent top-K sparse matrices on character 3-4 grams for both clean name and clean address, merged via fast COO sparse matrix union.
  3. *1-to-1 Mathematical Argmax Matching*: Filters pairs by threshold $\tau \ge 0.85$, sorts by predicted probability, and deduplicates on `cand_entity_id`, enforcing cluster uniqueness and eliminating cross-cluster collisions.

---

## 3. Candidate Generation (Blocking)

- **Blocking Keys & Algorithms Used**:
  - Open-set dynamic country partitioning via `df.groupby('country', sort=False)`.
  - Channel 1 (Name): Sublinear TF-IDF on character 3-4 grams (`max_features=50000`, top-35 per entity, cosine threshold $\ge 0.25$).
  - Channel 2 (Address): Sublinear TF-IDF on character 3-4 grams (`max_features=50000`, top-20 per entity, cosine threshold $\ge 0.30$).
  - Merge: Vectorized sparse COO matrix union with maximum similarity score tracking, capped at top-50 candidates per entity.
- **Candidate Pairs Generated**: ~39.7 candidates per reference entity (79,530 pairs on 2,000 validation slice; ~68.8M pairs projected across full test set).
- **How True Matches Were Preserved**:
  - Name-only TF-IDF recall was measured at ~83% due to corrupted names or URL variations.
  - Adding the independent Address TF-IDF channel lifted blocking recall to **99.55%** (6,834 / 6,865 matches recalled), successfully recovering matches with distorted or missing name stems.

---

## 4. Matching Model

**Features Used (23 Discriminative Pairwise Features):**
- **Name Similarity**:
  - Normalized Levenshtein similarity (`rapidfuzz.distance.Levenshtein`)
  - Jaro-Winkler similarity (rewards prefix brand consistency)
  - Token Sort Ratio (word permutation invariant)
  - Token Set Ratio (handles subset/superset tokens)
  - Partial Ratio (substring match for brand stems)
  - Exact match flag (`name_exact`)
  - Primary brand-token match flag (`first_token_match`)
  - Proportional length difference (`name_len_diff`)
- **Address Similarity**:
  - Normalized Levenshtein distance
  - Jaro-Winkler similarity
  - Token Sort Ratio
  - Token Set Ratio
  - Partial Ratio
  - Exact match flag (`addr_exact`)
  - Proportional address length difference (`addr_len_diff`)
- **Structural Matches & Conflict Vetoes**:
  - Exact postal code match flag (`postal_exact`)
  - Postal 3-digit prefix match flag (`postal_prefix`)
  - Postal 3-digit conflict flag (`postal_conflict` — powerful negative discriminator)
  - Building/plot number match flag (`bldg_match`)
  - Building/plot number conflict flag (`bldg_conflict` — powerful negative discriminator)
- **Jurisdictional & Suffix Signals**:
  - Canonical legal suffix match flag (e.g., `INC == INC`, `PVT_LTD == PVT_LTD`)
  - Legal suffix conflict flag (e.g., `INC != LLC`)
- **Retrieval Scores**:
  - Name TF-IDF similarity score
  - Address TF-IDF similarity score

**Model Architecture**:
- **Model Type**: LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`).
- **Hyperparameters**: `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=auto`.
- **Cross-Validation**: 5-Fold `GroupKFold` grouped strictly by `source1_entity_id` to eliminate data leakage.
- **Threshold Selection & Argmax Gating**:
  - 1-to-1 argmax assignment per candidate entity.
  - Threshold sweep over OOF probabilities $\tau \in [0.60, 0.95]$.
  - Selected optimal threshold: $\tau^* = 0.91$, achieving optimal trade-off between precision (99.15%) and recall (96.95%).

---

## 5. Results & Error Analysis

- **Validation Metrics (v2 Precision-Engineered GBDT)**:
  - **Macro $F_{0.5}$ Score:** **0.9846**
  - **Macro Precision:** **0.9915** (99.15%)
  - **Macro Recall:** **0.9695** (96.95%)
  - **Blocking Recall:** **99.55%** (6,834 / 6,865 matches recovered)
  - **Optimal Cutoff:** 0.91
- **Comparative Progression**:
  - v1 Baseline: $F_{0.5} = 0.1214$ (evaluation artifact: only 247 of 6,865 matches loaded in validation candidate pool).
  - v2 Full Evaluation Pool + Standard Threshold: $F_{0.5} = 0.9802$ (Precision: 98.60%, Recall: 97.15%).
  - v2 + Enhanced Conflict Features + 1-to-1 Argmax: $F_{0.5} = \mathbf{0.9846}$ (Precision: **99.15%**, Recall: **96.95%**).
- **Residual Error Analysis**:
  - Total False Positives across 2,000 entities: only 47 (0.69% FP rate).
  - Singletons correctly predicted empty: > 98% of true singletons score a perfect 1.0.
  - Residual False Negatives: extreme cases where both name and address differ dramatically (e.g., brand renamings with street-level address shifts).

---

## 6. Conclusion
The v2 pipeline achieves top-tier competitive performance by combining universal normalization, dynamic country partitioning, dual-channel sparse TF-IDF blocking (99.55% recall), conflict-aware GBDT features, and 1-to-1 argmax candidate assignment. With a Macro $F_{0.5}$ of **0.9846** (99.15% Precision), zero hardcoded country conditionals, and 100% pass on official validation checks, the solution is robust, fast, and competition-ready.

---

## Appendix

### A. Code Artefacts
The complete runnable solution is organized per the official competition structure:
```
code/business_entity_resolution/
├── src/
│   └── pipeline.ipynb        # Primary executable end-to-end pipeline notebook
├── README.md                 # Reproduction commands and environment details
└── requirements.txt          # Pinned dependency environment
```
**Entry Point**:
Execute `src/pipeline.ipynb` in Python 3.11 with the `amazon_ml` environment. It reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` end-to-end, followed by running `student_resource/utils/validate_submission.py`.

### B. Additional Results
- Official Submission Validator Output:
  ```
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows
  candidate_pairs.tsv: 1732544 rows
  PASS — no blocking issues found. Safe to submit.
  ```
