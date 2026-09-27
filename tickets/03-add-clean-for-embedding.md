# 03 — Add `clean_for_embedding()` to cleaning_utils.py

**What to build:** Add a lighter cleaning function alongside (NOT replacing) `clean_record()` that produces embedding-friendly text: transliterated to Latin script but preserving case, word order, and abbreviations.

**Blocked by:** None — can start immediately (no GPU dependency).

**Status:** ready-for-agent

## Detailed instructions

### File to modify: `code/business_entity_resolution/src/cleaning_utils.py`

1. **Do NOT modify** `clean_record()` or `clean_batch()` yet (that's ticket 04). Only ADD a new function.

2. Add this function after the existing `clean_record()` function:

   ```python
   def clean_for_embedding(raw_name, raw_addr):
       """
       Minimal cleaning for embedding models: transliterate non-Latin scripts
       to Latin, strip only control/junk punctuation, but preserve case,
       word order, and abbreviations.
       """
       raw_name_str = '' if raw_name is None else str(raw_name).strip()
       raw_addr_str = '' if raw_addr is None else str(raw_addr).strip()

       if raw_name_str.lower() in SENTINELS:
           name_out = ''
       else:
           name_out = transliterate_if_needed(raw_name_str) if NON_LATIN_INDIC_REGEX.search(raw_name_str) else raw_name_str
           name_out = WHITESPACE_REGEX.sub(' ', name_out).strip()

       if raw_addr_str.lower() in SENTINELS:
           addr_out = ''
       else:
           addr_out = transliterate_if_needed(raw_addr_str) if NON_LATIN_INDIC_REGEX.search(raw_addr_str) else raw_addr_str
           addr_out = WHITESPACE_REGEX.sub(' ', addr_out).strip()

       return name_out, addr_out
   ```

3. This function reuses existing constants already defined in the file:
   - `SENTINELS` — set of null-like strings
   - `NON_LATIN_INDIC_REGEX` — regex `[\u0900-\u0D7F]`
   - `WHITESPACE_REGEX` — regex `\s+`
   - `transliterate_if_needed()` — existing function

4. Quick validation: import and call it manually:
   ```python
   from cleaning_utils import clean_for_embedding
   # Test with a Devanagari string
   n, a = clean_for_embedding("आईसीआईसीआई बैंक", "मुंबई, महाराष्ट्र")
   print(repr(n), repr(a))  # should be Latin transliterated
   # Test with English — should pass through mostly unchanged
   n2, a2 = clean_for_embedding("ICICI Bank Ltd.", "123 Main St, Mumbai")
   print(repr(n2), repr(a2))  # should preserve case and abbreviations
   ```

- [x] `clean_for_embedding()` added to `cleaning_utils.py` without modifying existing functions
- [x] Function uses existing `SENTINELS`, `NON_LATIN_INDIC_REGEX`, `WHITESPACE_REGEX`, `transliterate_if_needed`
- [x] Devanagari input produces Latin-script output
- [x] English input preserves case and abbreviations (not lowercased, "Ltd." not expanded)
