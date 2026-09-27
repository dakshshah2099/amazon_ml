# 04 — Update `clean_batch()` and `preprocess.py` for 12-column output

**What to build:** Extend `clean_batch()` to call `clean_for_embedding()` per record, appending 2 new fields to the output tuple (10→12 elements). Update `preprocess.py`'s `OUTPUT_COLS` to include the new columns. Validate on a small sample, then run the full preprocessing job.

**Blocked by:** 03 (Add `clean_for_embedding`)

**Status:** ready-for-agent

## Detailed instructions

### Step 1: Modify `clean_batch()` in `code/business_entity_resolution/src/cleaning_utils.py`

The current `clean_batch()` function:
- Takes a list of `(entity_id, country, business_name, business_address)` tuples.
- Calls `clean_record(raw_name, raw_addr)` which returns a 6-tuple.
- Builds a 10-element result tuple: `(entity_id, country, raw_name, raw_addr, *clean_record_output)`.

**Modification:** After calling `clean_record()`, also call `clean_for_embedding(raw_name, raw_addr)` and append the 2 returned values to the result tuple, making it 12 elements:

```python
def clean_batch(batch_records):
    results = []
    for entity_id, country, raw_name, raw_addr in batch_records:
        clean_name, clean_addr, postal_code, has_state, needs_trans_name, needs_trans_addr = clean_record(raw_name, raw_addr)
        embed_name, embed_addr = clean_for_embedding(raw_name, raw_addr)
        results.append((
            entity_id, country, raw_name, raw_addr,
            clean_name, clean_addr, postal_code, has_state,
            needs_trans_name, needs_trans_addr,
            embed_name, embed_addr  # NEW: positions 10 and 11
        ))
    return results
```

### Step 2: Update `OUTPUT_COLS` in `code/business_entity_resolution/src/preprocess.py`

Current `OUTPUT_COLS` (10 items):
```python
OUTPUT_COLS = [
    'entity_id', 'country', 'business_name', 'business_address',
    'business_name_clean', 'business_address_clean', 'postal_code',
    'has_state', 'needs_transliteration_name', 'needs_transliteration_address'
]
```

Change to (12 items):
```python
OUTPUT_COLS = [
    'entity_id', 'country', 'business_name', 'business_address',
    'business_name_clean', 'business_address_clean', 'postal_code',
    'has_state', 'needs_transliteration_name', 'needs_transliteration_address',
    'business_name_for_embedding', 'business_address_for_embedding'
]
```

### Step 3: Verify no code assumes tuple length == 10

Scan `preprocess.py` for any code that:
- Indexes into the result tuple beyond position 9 (there shouldn't be any, but check).
- Uses `len(result) == 10` or similar assertions.
- The stats aggregation loop uses `r[6]` through `r[9]` — those indices are unaffected since the new fields are appended at positions 10/11.

### Step 4: Validate on a small sample

Before running the full job, test on 10,000 rows:
1. Temporarily modify `FILES` in `preprocess.py` to process only one file, or create a truncated TSV with 10k rows:
   ```powershell
   Get-Content "dataset/train/train_source1.tsv" -TotalCount 10001 | Set-Content "dataset/train/train_source1_sample.tsv"
   ```
2. Run preprocessing on that sample.
3. Verify the output parquet has 12 columns by:
   ```python
   import pandas as pd
   df = pd.read_parquet('dataset/train/train_source1_clean.parquet')
   print(df.columns.tolist())  # should show all 12
   print(df[['business_name_for_embedding', 'business_address_for_embedding']].head())
   ```
4. Find 5 rows where `needs_transliteration_name == True` and confirm `business_name_for_embedding` is Latin script (not Devanagari/Tamil/etc.).

### Step 5: Run full preprocessing

Once sample validated, run the full `preprocess.py` on all 6 files (train + test, sources 1-3). This is a long job — let it run to completion.

```powershell
cd code/business_entity_resolution/src
python preprocess.py
```

The output parquets go to `dataset/train/` and `dataset/test/`.

- [x] `clean_batch()` returns 12-element tuples (positions 10/11 are embed_name, embed_addr)
- [x] `OUTPUT_COLS` in `preprocess.py` has 12 items, ending with `business_name_for_embedding` and `business_address_for_embedding`
- [x] No existing index-based code broken by the tuple extension
- [x] Sample run: output parquet has 12 columns
- [x] Sample run: 5 transliterated rows show Latin script in `business_name_for_embedding`
- [x] Full preprocessing run completes for all 6 files (train S1/S2/S3 + test S1/S2/S3)
