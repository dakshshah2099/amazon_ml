# 15 — Report results and diagnose gap honestly

**What to build:** Compare the real Unstop test F0.5 against the OOF F0.5 from training. Report the gap plainly. If the gap is large (> a few points), diagnose likely causes and recommend next steps.

**Blocked by:** 14 (Test submission complete), 12 (Train sanity check gives OOF baseline)

**Status:** ready-for-agent

## Detailed instructions

### Step 1: Record numbers

Create a results summary with these numbers:
- **OOF F0.5** from ticket 11 (training, out-of-fold)
- **Train-split F0.5** from ticket 12 (sanity check)
- **Real test F0.5** from ticket 14 (Unstop submission)

### Step 2: State which split each number came from

Every F0.5 number printed must state its split:
- "OOF F0.5 (training distribution, 5-fold cross-validation): X.XX%"
- "Train-split inference F0.5 (training data, full model): X.XX%"
- "Test F0.5 (Unstop held-out test set): X.XX%"

### Step 3: If test F0.5 < 99%

State explicitly: "The target F0.5 > 99% was not achieved. This is an aggressive target for a cross-source, multi-country, partially-transliterated entity-matching task, and 99%+ generalizing F0.5 is not guaranteed by the techniques used."

### Step 4: Diagnose gap (if > few points)

If OOF–test gap is large, investigate in this order of likelihood:

1. **Country coverage gap**: A country in test underrepresented or absent in train. Check per-country prediction counts and investigate.

2. **Blocking recall gap for specific segment**: Re-run `tune_blocking_v2.py` with larger sample (10,000+) focusing on the weakest country/source combination.

3. **True distribution shift**: The contest test set may have deliberately harder negative pairs. If so, state this plainly rather than chasing it with more engineering.

### Step 5: Do NOT present OOF as final result

The real answer to "did this work" is only the Unstop test F0.5. OOF is context, not the answer.

- [ ] All three F0.5 numbers recorded with explicit split labels
- [ ] Gap between OOF and test F0.5 stated plainly
- [ ] If target missed: honest statement about aggressive target
- [ ] If gap large: ordered diagnosis of likely causes
- [ ] OOF never presented as final result
