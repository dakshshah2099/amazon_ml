# 12 — Validate training on train split (sanity check)

**What to build:** Run `predict_matches_v2.py` (ticket 13) on the train split as a sanity check. Compare the resulting F0.5 against the OOF F0.5 from training. This is NOT the final evaluation — it's a smoke test that the inference path produces consistent results with what training reported.

**Blocked by:** 11 (train_matcher_v2 complete), 13 (predict_matches_v2 exists)

**Status:** ready-for-agent

## Detailed instructions

This ticket is sequenced after ticket 13 (which creates predict_matches_v2.py) even though it runs on train data, because it needs the prediction script to exist.

### Steps

1. Run prediction on train split:
   ```powershell
   cd code/business_entity_resolution/src
   python predict_matches_v2.py --split train
   ```

2. Evaluate against ground truth:
   ```python
   import csv

   # Load GT
   gt = {}
   with open('dataset/train/train_ground_truth.tsv', 'r') as f:
       reader = csv.reader(f, delimiter='\t')
       next(reader)
       for row in reader:
           if len(row) >= 2 and row[1].strip():
               gt[row[0].strip()] = {m.strip() for m in row[1].split(',') if m.strip()}

   # Load predictions
   preds = {}
   with open('output_v2/train_matching_results.tsv', 'r') as f:
       reader = csv.reader(f, delimiter='\t')
       next(reader)
       for row in reader:
           if len(row) >= 2 and row[1].strip():
               preds[row[0].strip()] = {m.strip() for m in row[1].split(',') if m.strip()}

   # Compute F0.5
   tp = sum(len(gt.get(k, set()) & preds.get(k, set())) for k in gt)
   fp = sum(len(preds.get(k, set()) - gt.get(k, set())) for k in gt)
   fn = sum(len(gt.get(k, set()) - preds.get(k, set())) for k in gt)
   p = tp / (tp + fp) if (tp + fp) > 0 else 0
   r = tp / (tp + fn) if (tp + fn) > 0 else 0
   f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0

   print(f"Train-split F0.5: {f05*100:.2f}% (P={p*100:.2f}%, R={r*100:.2f}%)")
   print("NOTE: This is on TRAINING data, not the real test set.")
   ```

3. Compare with OOF F0.5 from ticket 11. They should be "reasonably close" — not wildly different. The train-split number may be slightly higher (since final models are trained on ALL data including this split). If it's dramatically different, something is wrong in the inference path.

- [ ] Prediction on train split completes
- [ ] Train-split F0.5 computed and printed
- [ ] F0.5 labeled explicitly as "on TRAINING data"
- [ ] F0.5 is reasonably close to OOF F0.5 from training (not wildly different)
