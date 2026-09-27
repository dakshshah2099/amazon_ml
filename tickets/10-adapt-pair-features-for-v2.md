# 10 — Adapt `compute_pair_features()` for v2 blocking output

**What to build:** Update `compute_pair_features()` to handle the new blocking_v2 output format (single `embed_score` instead of `signal_type + name_score + addr_score`). The 5 old signal-dependent features (`sig_both`, `sig_name`, `name_sim`, `addr_sim`, `score_gap`) must be redefined to use the single embedding score instead.

**Blocked by:** 08 (Embedding features defined), 06 (blocking_v2 output format known)

**Status:** ready-for-agent

## Detailed instructions

### Problem

The old `compute_pair_features()` in `feature_utils.py` takes these blocking-related params:
```python
def compute_pair_features(s1_name, s1_addr, s1_pc, s2_name, s2_addr, s2_pc,
                          signal_type='', name_score=0.0, addr_score=0.0,
                          s1_has_state=False, s2_has_state=False,
                          s1_needs_trans=False, s2_needs_trans=False):
```

The old blocking produced per-pair: `signal_type` (one of 'both', 'name', 'address'), `name_score`, `addr_score`.

The new blocking_v2 produces per-pair: `embed_score` (single float, cosine similarity of joint name+addr embeddings).

### Features to adapt (positions 16-20 in FEATURE_NAMES)

| Index | Old Name | Old Computation | New Computation |
|-------|----------|----------------|-----------------|
| 16 | `sig_both` | `1.0 if signal_type == 'both'` | `1.0 if embed_score >= 0.70` (high-confidence proxy) |
| 17 | `sig_name` | `1.0 if signal_type == 'name'` | **Remove or set to 0.0** — no separate name signal exists |
| 18 | `name_sim` | `name_score` | `embed_score` (reuse as the blocking similarity) |
| 19 | `addr_sim` | `addr_score` | `embed_score` (same value — there's only one score now) |
| 20 | `score_gap` | `name_score - addr_score` | `0.0` (no gap to compute with single score) |

### Implementation

**Option A (recommended — simpler):** Change the function signature to accept `embed_score` instead of `signal_type/name_score/addr_score`:

```python
def compute_pair_features(s1_name, s1_addr, s1_pc, s2_name, s2_addr, s2_pc,
                          embed_score=0.0,
                          s1_has_state=False, s2_has_state=False,
                          s1_needs_trans=False, s2_needs_trans=False):
```

Then update features 16-20:
```python
features.append(1.0 if embed_score >= 0.70 else 0.0)  # sig_both -> high_embed_proxy
features.append(0.0)                                     # sig_name -> deprecated, always 0
features.append(embed_score)                              # name_sim -> embed_sim
features.append(embed_score)                              # addr_sim -> embed_sim (same)
features.append(0.0)                                      # score_gap -> no gap
```

**Keep `FEATURE_NAMES` list unchanged** (still 27 items, same names) to avoid breaking the concatenation in `all_features.py`. The names become slightly misleading but the positions are stable.

### Update `feature_utils.py`'s `compute_pair_features` implementation

1. Change signature: replace `signal_type='', name_score=0.0, addr_score=0.0` with `embed_score=0.0`.
2. Update the 5 feature computations at positions 16-20 as shown above.
3. All other 22 features (positions 0-15, 21-26) remain completely unchanged.

### Do NOT update callers in this ticket

The callers (`train_matcher_v2.py`, `predict_matches_v2.py`) are created in later tickets (11, 13). This ticket only changes the function itself.

- [x] `compute_pair_features()` signature changed: `signal_type/name_score/addr_score` → `embed_score`
- [x] Features 16-20 adapted to use single `embed_score`
- [x] Features 0-15 and 21-26 completely unchanged
- [x] `FEATURE_NAMES` list unchanged (27 items, same names)
- [x] Function still returns exactly 27 floats
