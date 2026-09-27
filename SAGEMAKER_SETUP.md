# AWS SageMaker AI Setup & Inference Guide

Repository: `https://github.com/dakshshah2099/amazon_ml.git` (Branch: `main`)

---

## 1. Recommended SageMaker Instance Type
- **Instance:** `ml.g4dn.xlarge` (T4 GPU, 16GB VRAM, 4 vCPUs, 16GB RAM) or `ml.g5.xlarge` (A10G GPU, 24GB VRAM).
- **Storage:** EBS Volume $\ge 100$ GB (for full 12.5M test set embeddings and parquets).
- **Environment:** PyTorch 2.x (CUDA 12.1), Python 3.10 / 3.11.

---

## 2. Setup on SageMaker Studio / Notebook Terminal

### Step A: Clone Repository
```bash
git clone https://github.com/dakshshah2099/amazon_ml.git
cd amazon_ml
```

### Step B: Install Dependencies
```bash
pip install -r code/business_entity_resolution/requirements.txt
# Ensure GPU support for PyTorch and FAISS
pip install faiss-gpu-cu12 sentence-transformers torch --upgrade
```

### Step C: Sync Data & Trained Models from S3
Upload your test data and trained sample/train models to S3 first from local:
```bash
# In SageMaker terminal:
aws s3 sync s3://<your-bucket>/dataset/test/ dataset/test/
aws s3 sync s3://<your-bucket>/models/ models/
```

---

## 3. End-to-End Inference Commands

### Step 1: Preprocess Test TSVs to Parquet
```bash
python code/business_entity_resolution/src/preprocess.py --split test
```

### Step 2: Generate Test Dense Embeddings (GPU Accelerated)
On SageMaker GPU, batch size 256/512 finishes full test set in ~20-30 minutes:
```bash
python code/business_entity_resolution/src/embed_entities.py --split test --batch-size 256
```

### Step 3: Run Hybrid Blocking (FAISS Exact + Multi-Key Inverted Index)
```bash
python code/business_entity_resolution/src/blocking.py \
  --split test \
  --k 20 \
  --min-score 0.40 \
  --max-bucket 30 \
  --out-dir output
```

### Step 4: Run Inference with Trained Models
```bash
python code/business_entity_resolution/src/predict_matches.py \
  --split test \
  --cand-dir output \
  --model-lgb models/lgb_matcher.txt \
  --model-cb models/cb_matcher.cbm \
  --meta models/matcher_metadata.pkl \
  --out output/matching_results.tsv \
  --workers 8
```

### Step 5: Upload Results to S3
```bash
aws s3 cp output/matching_results.tsv s3://<your-bucket>/submissions/matching_results.tsv
```
