# Business Entity Resolution — Preprocessing & Blocking Pipeline
## Design Spec + Agent Implementation Prompt

This document has two parts:
1. **SPEC** — the locked design decisions, for human reference.
2. **AGENT PROMPT** — paste this into your coding agent (Claude Code or similar) verbatim.

---

# PART 1: SPEC (reference only)

## Scope
This covers preprocessing + blocking (candidate generation) ONLY. Matching (LightGBM
classifier) is a separate later stage and is NOT part of this prompt.

## Data scale
- Source 1: ~2.2M rows (train), ~1.2M (test, incl. France)
- Source 2: ~5.0M rows (train), ~4.9M (test)
- Source 3: ~5.3M rows (train), ~5.1M (test)
- Countries: US, India (train); US, India, France (test)

## Hardware target
MacBook M4 Pro, 64GB unified memory, CPU-only FAISS (no CUDA on Mac).

## Locked decisions
1. **Country is a hard blocking filter.** Never compare records across different
   countries. Country strings are already clean in this dataset (verified: no
   casing/whitespace variants), so exact string match is safe.
2. **Separate blocking passes for Source 2 and Source 3.** Do not merge them into
   one pool before blocking — build independent indexes for S1-vs-S2 and S1-vs-S3.
3. **Two parallel blocking signals per (country, source) pair**: business_name-based
   and business_address-based. Built as separate FAISS indexes.
4. **Always union both signals' candidates** (do not use address as a fallback-only
   — always retrieve top-K from both name index and address index, then union and
   dedupe by entity_id).
5. **K = 30 nearest neighbors per signal** (tunable later; start here). After union
   + dedupe, expect ~20-60 candidates per source1 entity per source (S2, S3 each).
6. **Transliteration is applied from the start** (not deferred), to both
   business_name AND business_address fields, for any text containing non-Latin
   script. This runs BEFORE lowercasing/normalization.
7. **Landmark phrases are fully removed** from address text used for blocking
   (not down-weighted, not kept). "Near X", "Opp X", "Opposite X", "Behind X" and
   the landmark phrase following them are stripped entirely.
8. **candidate_pairs.tsv is allowed to be large for now.** Do not prematurely
   optimize candidate set size. K will be tuned down later once recall/precision
   tradeoffs are measured. Do not sacrifice recall for a smaller file at this stage.
9. **FAISS CPU with HNSW index** (IndexHNSWFlat), biased toward HIGH RECALL since
   this is a one-time offline batch pass, not a latency-sensitive live system:
   - M = 48 (connections per node — higher = better recall, more memory/build time)
   - efConstruction = 300
   - efSearch = 300 (at query time)
   These are starting points; log actual recall on a validation slice and note if
   they need raising further.
10. **Dimensionality reduction via TruncatedSVD to ~256 dimensions** before FAISS
    indexing (applied per country, per field-type — i.e. a separate SVD fit for
    name-vectors and address-vectors, per country). This is for index quality/speed,
    not memory necessity (64GB is enough headroom to skip this, but empirically
    HNSW performs better in moderate dimensionality than on raw high-dim sparse-
    derived vectors).
11. **PIN/ZIP codes**: extract into a separate field via regex, remove from the
    blocking text entirely. Store alongside cleaned records for later use as a
    matcher feature (not used in blocking itself).
12. **has_state flag**: compute and store as a boolean per address record (for
    later matcher feature), but do NOT let it affect blocking.

## Preprocessing pipeline order (must be applied in this exact order)

For BOTH business_name and business_address fields, per record:

1. **Script detection**: regex-check for non-Latin unicode ranges (Devanagari
   U+0900–U+097F, Telugu U+0C00–U+0C7F, Tamil U+0B80–U+0BFF, and other Indic
   ranges as needed). Flag record field as needs_transliteration = True/False.
2. **Transliteration**: if flagged, transliterate using a LOCAL library (e.g.
   `indic_transliteration` or `ai4bharat`/`indic-nlp-library` — must run fully
   offline, no API calls, no internet lookups — this is exploratory/normalization
   tooling on the given dataset only, not an external entity lookup, so it is
   within contest fair-play rules). Convert to Latin phonetic representation.
3. **Lowercase** the entire string.
4. **Punctuation normalization**: strip `[`, `]`, `#`, `*`, `@`, `!`, `>`, `<`,
   `_`, `:`, `+`, `|` entirely (replace with space). Keep `,` and `-` as they can
   delimit meaningful address components — but strip them too for the FINAL
   n-gram blocking text (keep a pre-strip copy of the raw cleaned string for
   potential future feature use, e.g. component splitting). For business_name,
   also normalize `&` — see step 5.
5. **Suffix / abbreviation expansion — business_name only.** Apply as whole-word
   token replacement (not substring replacement, to avoid corrupting unrelated
   text) using this exact mapping (case-insensitive, applied after lowercasing):
   ```
   ltd        -> limited
   pvt        -> private
   corp       -> corporation
   inc        -> incorporated
   co         -> company        (ONLY as a standalone token, not inside other words)
   &          -> and
   llc        -> llc            (no change, keep as-is, do not expand)
   llp        -> llp            (no change)
   pllc       -> pllc           (no change)
   plc        -> plc            (no change)
   sarl       -> sarl           (no change; French)
   sasu       -> sasu           (no change; French)
   eurl       -> eurl           (no change; French)
   ```
   Rationale: normalize toward the FULL form (limited, private, corporation,
   incorporated, company, and) since that gives a consistent canonical token
   regardless of which abbreviation each source happened to use. Do NOT expand
   llc/llp/pllc/plc/sarl/sasu/eurl — these don't have a common full-form
   counterpart in the data and are left as distinguishing tokens.
6. **Address abbreviation expansion — business_address only.** Apply as whole-word
   token replacement, normalize toward full form:
   ```
   rd    -> road
   st    -> street
   ave   -> avenue
   dr    -> drive
   fl    -> floor
   apt   -> apartment
   ste   -> suite
   blvd  -> boulevard
   ```
   Do not alter: nagar, colony, sector, rue, and other non-abbreviated
   country-specific tokens.
7. **PIN/ZIP extraction — business_address only.** Regex `\b\d{5,6}\b` — extract
   first match into a separate `postal_code` field. Remove the matched digits
   from the address text used for blocking (they add noise to n-grams and are
   handled as a separate exact-match feature later).
8. **has_state flag — business_address only.** Boolean: does the address contain
   a recognizable state name or abbreviation from a lookup list covering US
   states (+ DC, 2-letter codes), Indian states/UTs (+ common abbreviations:
   MH, DL, KA, TN, WB, TS, AP, UP, etc.), and French regions. Store as a
   separate field, do not modify the address text based on this.
9. **Landmark phrase removal — business_address only.** Regex to find and
   remove: `\b(near|opp|opposite|behind)\.?\s+[^,]+` (i.e. the keyword plus
   everything up to the next comma or end of string). Remove entirely from
   blocking text. Case-insensitive.
10. **Whitespace collapse**: collapse multiple spaces to one, strip leading/
    trailing whitespace. This is the FINAL cleaned text used for n-gram
    vectorization / blocking.

## Output of preprocessing stage
For each source file (source1, source2, source3 — train and test), produce a
cleaned parquet or TSV with columns:
```
entity_id, country, business_name_clean, business_address_clean,
postal_code, has_state, needs_transliteration_name, needs_transliteration_address
```
Keep original business_name/business_address columns too (don't discard raw data).

## Blocking stage design

For each country C in {US, India, France} (France only exists in test):
  For each source pair in {(S1, S2), (S1, S3)}:
    For each signal in {name, address}:
      1. Build char n-gram (3-4 gram, i.e. ngram_range=(3,4), analyzer='char_wb')
         TF-IDF vectorizer, FIT ONLY on the combined vocabulary of S1 + the
         relevant source (S2 or S3) records within country C. Cap max_features
         at a reasonable vocab size (e.g. 20,000) to keep SVD/indexing tractable
         — use TF-IDF's built-in max_features parameter, selecting by document
         frequency.
      2. Transform both S1 and the source's records within country C into TF-IDF
         sparse vectors.
      3. Fit TruncatedSVD(n_components=256) on the source-side vectors (S2 or S3),
         transform both S1 query vectors and source vectors into 256-dim dense
         space using the SAME fitted SVD.
      4. L2-normalize all vectors (required for cosine similarity via inner
         product in FAISS).
      5. Build FAISS IndexHNSWFlat(256, M=48) with metric_type=METRIC_INNER_PRODUCT,
         set hnsw.efConstruction = 300, add all source-side vectors (S2 or S3,
         within country C).
      6. Set index.hnsw.efSearch = 300. Query with all S1 vectors (within country
         C), k=30. This returns top-30 nearest source-side entity_ids + similarity
         scores per S1 query.
      7. Save results as (source1_entity_id, candidate_entity_id, similarity_score,
         signal_type) rows.

After all indexes for a given (country, source_pair) are queried for BOTH signals:
  - Union the name-signal and address-signal candidate lists per source1_entity_id
    (dedupe by candidate_entity_id; if a candidate appears in both signals, keep
    the max similarity score and note both signals matched — this is useful
    information for the matcher stage later, so keep a boolean flag
    matched_by_name / matched_by_address per candidate row, not just a merged list).

After processing (S1,S2) and (S1,S3) separately, per source1_entity_id, further
union across BOTH sources into the final candidate list for that entity (S2 and
S3 candidates both belong in the same row's candidate list per the file format
spec, since candidate_pairs.tsv has one row per source1 entity covering both
sources).

## Output of blocking stage
1. `candidate_pairs_detailed.tsv` (internal, keep all metadata — NOT the final
   submission file): source1_entity_id, candidate_entity_id, source
   (S2 or S3), signal_type (name/address/both), similarity_score
2. `candidate_pairs.tsv` (matches the contest's required format exactly):
   source1_entity_id [tab] candidate_entity_ids (comma-separated, no quoting,
   S2/S3 IDs only, no duplicates, empty string if no candidates found). One row
   per Source 1 entity, EVERY source1 entity must appear even if it has zero
   candidates.

## Validation to run after blocking
Using train_ground_truth.tsv, measure BLOCKING RECALL: for each source1 entity
with true matches, what fraction of true matched_entity_ids actually appear in
that entity's candidate list? Report:
- Overall recall (macro-averaged per entity, and micro/pooled)
- Recall broken down by country
- Recall broken down by signal (name-only candidates vs address-only candidates
  vs union) — i.e. ablation: what recall would we get with name-signal alone,
  address-signal alone, vs union — to justify or challenge the always-union
  decision
- Average candidate list size per entity (to understand the recall/size tradeoff)
- List the entities where blocking FAILED to find a true match at all (true
  match missing from candidates) — sample 20 of these and print their
  business_name/business_address (source1 and the missed true match) so we can
  diagnose WHY blocking missed them (still-cross-script after transliteration?
  extreme DBA name mismatch? something else?)

This recall number is the ceiling for the entire pipeline's F0.5 score — report
it clearly and do not proceed to matcher-stage work until this is reported back.

---