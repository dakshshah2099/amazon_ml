# Experiment Notes: v2-precision-engineered-gbdt

## Key Breakthroughs
1. **Resolved v1 Evaluation Artifact**:
   - In v1, `val_s23` broke early after 1 chunk, loading only 247 of 6,865 true matches into memory. This caused an artificially deflated score of 0.1214.
   - v2 reads all chunks for targeted evaluation matches, guaranteeing that 100% of ground-truth matches exist in the validation candidate pool.
2. **1-to-1 Mathematical Argmax Matching**:
   - Analysis of `train_ground_truth.tsv` revealed 0 candidate entities assigned to >1 Source 1 cluster across all 7,638,365 matches.
   - Enforcing 1-to-1 candidate assignment (sorting by predicted probability and deduplicating on `cand_entity_id`) eliminates cross-cluster collisions and dramatically reduces false positives to only 47 across 2,000 entities.
3. **Enhanced Discriminative Features**:
   - Added brand-token matching (`first_token_match`), string length differences (`name_len_diff`, `addr_len_diff`), structural conflicts (`bldg_conflict`, `postal_conflict`), and substring ratios (`name_partial`, `addr_partial`, `name_exact`, `addr_exact`).
   - Structural veto features strongly penalize candidate pairs that disagree on street building numbers or 3-digit postal code prefixes.
4. **Tuned Blocking Parameters**:
   - Name TF-IDF: top_n=35, threshold=0.25
   - Address TF-IDF: top_n=20, threshold=0.30
   - Union capped at 50 candidates per entity.
   - Boosted blocking recall from 99.37% to **99.55%**.

## Final Metrics
- **Macro F₀.₅**: **0.9846**
- **Macro Precision**: **0.9915** (99.15%)
- **Macro Recall**: **0.9695** (96.95%)
- **Blocking Recall**: **0.9955** (99.55%)
- **Optimal Cutoff**: **0.91**
- **Official Validator**: **PASS** (Zero blocking issues, 1,732,544 test rows verified).
