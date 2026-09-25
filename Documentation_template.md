# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [To be filled by participant]  
**Team Members:** [To be filled by participant]  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed an open-set, country-agnostic Entity Resolution (ER) pipeline engineered specifically for the precision-weighted macro $F_{0.5}$ metric. The architecture integrates universal multi-jurisdiction preprocessing, open-set country-partitioned dual-channel sparse TF-IDF blocking (achieving 99.60% recall at ~18 candidates/entity), vectorized pairwise lexical/structural feature extraction via RapidFuzz, and a 5-Fold GroupKFold LightGBM classifier with an entity-level singleton preservation threshold optimization.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across ~12.5M train and ~11.7M test records, we identified critical structural patterns:
- **Intra-Country Invariance**: 100% of validated ground-truth matches occur strictly within the same country label. Partitioning dynamically by country reduces comparison space by 50–70% with zero recall loss.
- **Open-Set Country Challenge**: Training data comprises US and India, while the test set introduces unseen jurisdictions (France). Normalization and blocking must avoid hardcoded country conditionals.
- **Asymmetric Noise Modalities**: Empirical inspection revealed two dominant error patterns:
  1. *Name corruption with clean address*: e.g. web URLs, brand truncation, or OCR errors, while address text remains near-identical.
  2. *Missing address with clean name*: Records where address is null/NaN, but business name contains high lexical overlap. Single-channel blocking on concatenated text fails on these edge cases.
- **Metric Asymmetry & Singletons**: Under macro $F_{0.5}$, precision is weighted 2× over recall, making false merges ~4× more damaging than false negatives. Singletons (entities with 0 matches) represent a substantial portion of reference entities; predicting an empty match list correctly awards a full 1.0 score, while a single false merge collapses the entity score to 0.0.

### 2.2 Solution Strategy
- **Approach Type**: Multi-Stage Hybrid Pipeline (Country-Partitioned Dual Sparse TF-IDF Blocking + Vectorized Pairwise GBDT + Macro $F_{0.5}$ Calibrated Thresholding).
- **Core Innovation**:
  1. *Universal Country-Agnostic Normalization*: International postal code regex, global prepositional landmark cleaning, and Unicode NFKD decomposition running uniformly across all jurisdictions.
  2. *Dual-Channel Sparse TF-IDF*: Independent top-K sparse matrices on character 3-4 grams for both clean name and clean address, merged via fast COO sparse matrix union.

---

## 3. Candidate Generation (Blocking)

- **Blocking Keys & Algorithms Used**:
  - Open-set dynamic country partitioning via `df.groupby('country', sort=False)`.
  - Channel 1 (Name): Sublinear TF-IDF on character 3-4 grams (`max_features=50000`, top-20 per entity, cosine threshold $\ge 0.35$).
  - Channel 2 (Address): Sublinear TF-IDF on character 3-4 grams (`max_features=50000`, top-10 per entity, cosine threshold $\ge 0.40$).
  - Merge: Vectorized sparse COO matrix union with maximum similarity score tracking, capped at top-35 candidates per entity.
- **Candidate Pairs Generated**: ~18.3 candidates per reference entity (~36,649 pairs on 2,000 validation slice; ~31.7M pairs projected across full test set).
- **How True Matches Were Preserved**:
  - Name-only TF-IDF recall was measured at 82.24% due to corrupted names or URL variations.
  - Adding the independent Address TF-IDF channel lifted blocking recall to **99.60%**, successfully recovering matches with distorted or missing name stems.

---

## 4. Matching Model

**Features Used (14 Discriminative Pairwise Features):**
- **Name Similarity**:
  - Normalized Levenshtein similarity (`rapidfuzz.distance.Levenshtein`)
  - Jaro-Winkler similarity (rewards prefix brand consistency)
  - Token Sort Ratio (word permutation invariant)
  - Token Set Ratio (handles subset/superset tokens)
- **Address Similarity**:
  - Normalized Levenshtein distance
  - Jaro-Winkler similarity
  - Token Set Ratio
- **Structural Matches**:
  - Exact postal code match flag (`postal_exact`)
  - Postal 3-digit prefix match flag (`postal_prefix`)
  - Building/plot number match flag (`bldg_match`)
- **Jurisdictional & Suffix Signals**:
  - Canonical legal suffix match flag (e.g., `INC == INC`, `PVT_LTD == PVT_LTD`)
  - Legal suffix conflict flag (e.g., `INC != LLC`)
- **Retrieval Scores**:
  - Name TF-IDF similarity score
  - Address TF-IDF similarity score

**Model Architecture**:
- **Model Type**: LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`).
- **Hyperparameters**: `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=auto`.
- **Cross-Validation**: 5-Fold `GroupKFold` grouped strictly by `source1_entity_id` to eliminate data leakage.
- **Threshold Selection**:
  - Grid search over Out-Of-Fold (OOF) predicted probabilities $\tau \in [0.40, 0.90]$.
  - Directly optimizes macro-averaged $F_{0.5}$ including singleton credit (empty prediction on true singleton = 1.0).
  - Selected optimal threshold: $\tau^* = 0.46$.

---

## 5. Results & Error Analysis

- **Validation Metrics (v1 Baseline)**:
  - **Macro $F_{0.5}$ Score:** 0.1214
  - **Macro Precision:** 0.1616
  - **Macro Recall:** 0.0851
  - **Blocking Recall:** 99.60% (246 / 247 testable matches recovered)
- **Common False Positives (Wrong Merges)**:
  - Co-located businesses sharing the exact same street/building address but operating under different trade names.
  - Common franchise/chain names located at different branches where postal codes are missing.
- **Common False Negatives (Missed Matches)**:
  - Highly sparse records where both business name has severe typographical divergence and address field is completely missing (`NaN`).

---

## 6. Conclusion
The initial baseline establishes a high-recall, country-agnostic foundation for the challenge. By combining universal normalization, dynamic country partitioning, dual-channel sparse TF-IDF blocking (99.60% recall), and GroupKFold LightGBM, the pipeline satisfies all challenge constraints, prevents data leakage, and validates cleanly against the official submission format.

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
