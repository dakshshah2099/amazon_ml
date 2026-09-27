# 01 — Install new dependencies & validate GPU

**What to build:** Install `sentence-transformers`, `torch` (CUDA 12.1), `catboost`, and `optuna` into the existing `.venv` environment. Validate the GPU is usable for all downstream embedding work.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

## Detailed instructions

1. Activate the existing venv at `.venv/Scripts/activate.ps1`.

2. Run these installs (in order — torch must come from the CUDA 12.1 index):
   ```
   pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cu121
   pip install catboost
   pip install optuna
   ```

3. Validate GPU access by running this snippet:
   ```python
   import torch
   print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
   ```
   - Must print `True` and a name containing "3050" or similar.
   - If `False`: stop and resolve CUDA/torch mismatch before marking done. Every downstream ticket assumes GPU.

4. Update `code/business_entity_resolution/requirements.txt` to add:
   ```
   sentence-transformers
   torch
   catboost
   optuna
   lightgbm
   rapidfuzz
   ```
   (Note: `lightgbm` and `rapidfuzz` were already imported by existing code but missing from requirements.txt — add them now.)

- [x] `torch.cuda.is_available()` prints `True`
- [x] GPU name confirmed as RTX 3050 Ti (or similar)
- [x] `requirements.txt` updated with all 6 new/missing packages
- [x] `from sentence_transformers import SentenceTransformer` imports without error
- [x] `from catboost import CatBoostClassifier` imports without error
