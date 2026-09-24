# Amazon ML Challenge — Business Entity Resolution: Agent Guide

## Problem Summary

**Task:** Match business records across 3 independent sources (S1=reference, S2, S3) to the same real-world entity.
**Metric:** F₀.₅ (precision-heavy, macro-averaged per S1 entity). Precision weighted 2× over recall.
**Constraint:** ≤8B params, MIT/Apache 2.0 license, NO external APIs/data.

## Key Facts

| Aspect | Detail |
|---|---|
| Fields | `entity_id`, `business_name`, `business_address`, `country` |
| Countries | Train: US, India. Test: +France (unseen) |
| Output | `matching_results.tsv` (scored) + `candidate_pairs.tsv` (audit) |
| Singletons | S1 entities with 0 matches → empty `matched_entity_ids` → scores 1.0 |
| False merge on singleton | Scores 0.0 — catastrophic |

## Pipeline Architecture

```
Raw Records (S1, S2, S3)
    │
    ▼
[1. PREPROCESSING & NORMALIZATION]
    • Unicode NFKD decomposition (French diacritics)
    • Legal suffix extraction (Pvt Ltd, SARL, Inc, LLC → canonical tags)
    • DBA/trade name splitting
    • Address parsing: postal code, building number, landmark extraction
    • Abbreviation expansion (Corp→Corporation, Rd→Road)
    • Country-specific normalization (India PIN, France département)
    │
    ▼
[2. BLOCKING / CANDIDATE GENERATION] → candidate_pairs.tsv
    • Branch A: Deterministic rule-blocking (country + postal prefix + soundex)
    • Branch B: TF-IDF char n-gram sparse top-K (sparse_dot_topn)
    • Branch C: Dense bi-encoder ANN (all-MiniLM-L6 or bge-m3 + FAISS)
    • Union all candidates, deduplicate, cap per entity
    │
    ▼
[3. PAIRWISE FEATURE ENGINEERING] (~30 features per pair)
    • String metrics: Levenshtein, Jaro-Winkler, token sort/set, Jaccard
    • Hybrid: Monge-Elkan, Soft TF-IDF
    • Phonetic: Double Metaphone, NYSIIS, Soundex match flags
    • Char n-gram: 3-gram/4-gram Jaccard
    • Address: postal exact/prefix match, building number match
    • Legal suffix: match/conflict flags
    • Semantic: embedding cosine/euclidean/manhattan
    • Country match flag, name length diff ratio
    │
    ▼
[4. CLASSIFICATION]
    • LightGBM / CatBoost binary classifier
    • Threshold tuned for F₀.₅ (~0.72–0.85, NOT default 0.5)
    │
    ▼
[5. POST-PROCESSING]
    • Singleton gate: if max P(match) < τ → emit empty list
    • Reciprocal nearest neighbor verification
    • Output validation via utils/validate_submission.py
    │
    ▼
matching_results.tsv
```

## Critical Strategy Notes

1. **Precision > Recall**: F₀.₅ penalizes false merges ~4× more than missed links. Set high thresholds.
2. **Singletons matter**: Correctly predicting "no match" = 1.0. One false merge on a singleton = 0.0.
3. **France is unseen**: Pipeline must handle French diacritics (NFKD), legal suffixes (SARL, SAS, SA), and 5-digit postal codes without hardcoding country lists.
4. **Blocking recall ceiling**: Any true match missed in blocking can NEVER be recovered. Aim for PC > 98%.
5. **No external APIs**: All normalization/embedding must be offline. Disqualification otherwise.

## Detailed Guides

| Guide | Path | Contents |
|---|---|---|
| Blocking | [guides/01_blocking_strategies.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/01_blocking_strategies.md) | Candidate generation: rule-based, TF-IDF, dense ANN, hybrid, LSH |
| Features | [guides/02_feature_engineering.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/02_feature_engineering.md) | String similarity, name/address normalization, phonetics, embeddings |
| Models | [guides/03_models_and_approaches.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/03_models_and_approaches.md) | DITTO, DeBERTa cross-encoder, contrastive learning, LLM fine-tuning |
| Competition | [guides/04_competition_strategies.md](file:///C:/Users/daksh/Desktop/amazon_ml/guides/04_competition_strategies.md) | F₀.₅ tuning, singleton handling, precision tricks, pitfalls |

## Recommended Tech Stack

| Category | Library | Purpose |
|---|---|---|
| String metrics | `rapidfuzz` | C++ accelerated Levenshtein, Jaro-Winkler, token sort |
| Phonetics | `jellyfish` | Metaphone, NYSIIS, Soundex |
| TF-IDF | `scikit-learn` | Token & char n-gram vectorization |
| Sparse blocking | `sparse_dot_topn` | Fast sparse matrix top-K multiplication |
| Dense blocking | `sentence-transformers` + `faiss-cpu` | Bi-encoder embeddings + ANN search |
| LSH | `datasketch` | MinHash LSH for Jaccard blocking |
| Address parsing | Custom regex / `deepparse` | Postal code, building number extraction |
| Name cleaning | Custom + `cleanco` | Legal suffix stripping |
| Unicode | `anyascii` / `unicodedata` | Transliteration, diacritics removal |
| Classifier | `lightgbm` / `catboost` | Gradient boosted trees for pairwise classification |
| Validation | `utils/validate_submission.py` | Pre-submit format check |
