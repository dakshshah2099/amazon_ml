# Google Colab Setup Guide

## 1. Environment & GPU/CPU Configuration
In Colab:
- Navigate to: **Runtime** -> **Change runtime type**.
- Select **High-RAM CPU** or **T4 GPU / A100 GPU**. (Note: if using GPU, `faiss-gpu` or `faiss-cpu` work seamlessly).

---

## 2. Directory Structure Setup in Colab
Run this in a code cell:
```bash
!mkdir -p dataset/train dataset/test output code/business_entity_resolution/src utils tickets
```

Upload your files or connect Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
# Copy datasets from Drive if stored there:
# !cp -r /content/drive/MyDrive/amazon_ml/dataset/* dataset/
```

---

## 3. Package Installation
Run in a code cell:
```bash
!pip install -q faiss-cpu indic-transliteration scikit-learn pandas pyarrow tqdm pytest
```

---

## 4. End-to-End Execution Sequence

### Step 1: Preprocessing (Stage 1)
```bash
python code/business_entity_resolution/src/preprocess.py
```
*Outputs cleaned Parquet files to `dataset/train/` and `dataset/test/`.*

### Step 2: Run Automated Cleaner Tests
```bash
pytest code/business_entity_resolution/tests/test_cleaner.py -v
```

### Step 3: Blocking (Stage 2)
```bash
# For Train Split (Validation Run):
python code/business_entity_resolution/src/blocking.py --split train --out-dir output --dims 128

# For Test Split (Competition Submission Candidate Generation):
python code/business_entity_resolution/src/blocking.py --split test --out-dir output --dims 128
```

### Step 4: Validate Blocking Recall (Stage 3)
```bash
python code/business_entity_resolution/src/validate_blocking.py
```
*Generates `blocking_validation_report.md` with recall scores and error analysis.*

### Step 5: Submission Format Validation
```bash
python utils/validate_submission.py \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## 5. Artifacts to Download from Colab
Once finished, download:
1. `output/candidate_pairs.tsv`
2. `output/candidate_pairs_detailed.tsv`
3. `blocking_validation_report.md`
4. Clean Parquet files from `dataset/` (optional, to avoid recomputing Stage 1).
