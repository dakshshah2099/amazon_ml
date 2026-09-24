# Experiment Notes: v1-tfidf-lgbm-baseline

## Summary
Initial baseline implementation executed inside `pipeline.ipynb` across all 5 phases:
1. Multi-jurisdiction preprocessing (Unicode NFKD, legal suffix canonicalization, postal extraction).
2. Country-partitioned dual-channel sparse TF-IDF blocking (char 3-4 grams on clean name and clean address).
3. Fast vectorized pairwise feature extraction via RapidFuzz.
4. 5-Fold GroupKFold LightGBM classifier with threshold optimization for macro F0.5.
5. Submission generation structure and validation script check.

## Key Findings
- Matches are strictly 100% intra-country (zero cross-country matches in inspection). Country-partitioned blocking reduces candidate search space by 50–70% with zero loss of recall.
- Dual-channel blocking (Name top-20 + Address top-10) achieves **99.6% blocking recall** on testable ground truth matches with only ~18 candidates per S1 entity.
- Fast COO sparse matrix merge runs in sub-seconds.
- Baseline macro F0.5 achieved: 0.1214 with optimal threshold 0.46.

## Next Optimizations (v2)
- Include Soft TF-IDF and phonetic codes (Double Metaphone / NYSIIS).
- Scale training data to full training set chunks.
- Add reciprocal nearest neighbor / top-1 margin filter for precision boost.
