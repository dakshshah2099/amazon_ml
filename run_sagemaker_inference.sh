#!/bin/bash
set -e

echo "=========================================================="
echo "AMAZON ML SENSING / ENTITY RESOLUTION - SAGEMAKER RUNNER"
echo "=========================================================="
date

BUCKET="s3://amazon-ml-648426766204-ap-south-1"

# 1. Install Dependencies
echo -e "\n>>> Step 1: Installing Requirements..."
pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt

# 2. Sync Datasets and Trained Models from S3
echo -e "\n>>> Step 2: Syncing test TSVs and models from S3..."
mkdir -p dataset/test models output
aws s3 sync ${BUCKET}/dataset/test/ dataset/test/
aws s3 sync ${BUCKET}/models/ models/

# 3. Preprocess Test TSVs to Clean Parquet
echo -e "\n>>> Step 3: Preprocessing test TSVs to clean Parquet..."
python code/business_entity_resolution/src/preprocess.py --split test --workers 4

# 4. Generate Embeddings (Dense Semantic Representation on GPU with Tensor Core FP16)
echo -e "\n>>> Step 4: Generating embeddings for test entities on GPU..."
python code/business_entity_resolution/src/embed_entities.py --split test --batch-size 256

# 5. Hybrid Blocking (FAISS Exact + Multi-Key Inverted Index)
echo -e "\n>>> Step 5: Running Hybrid Blocking..."
python code/business_entity_resolution/src/blocking.py \
  --split test \
  --k 20 \
  --min-score 0.40 \
  --max-bucket 30 \
  --out-dir output

# 6. Predict Matches with Trained Ensemble
echo -e "\n>>> Step 6: Running Matcher Prediction..."
python code/business_entity_resolution/src/predict_matches.py \
  --split test \
  --cand-dir output \
  --model-lgb models/lgb_matcher.txt \
  --model-cb models/cb_matcher.cbm \
  --meta models/matcher_metadata.pkl \
  --out output/matching_results.tsv \
  --min-cand-score 0.52 \
  --threshold 0.92 \
  --workers 4

# 7. Upload Submission TSV to S3
echo -e "\n>>> Step 7: Uploading final submission to S3..."
aws s3 cp output/matching_results.tsv ${BUCKET}/submissions/matching_results.tsv

echo "=========================================================="
echo "RUN COMPLETED SUCCESSFULLY! Submission uploaded to:"
echo "${BUCKET}/submissions/matching_results.tsv"
echo "=========================================================="

# 8. Auto-stop notebook instance to avoid compute charges
echo -e "\n>>> Step 8: Auto-stopping notebook instance to avoid billing..."
aws sagemaker stop-notebook-instance --notebook-instance-name amazon-ml-gpu || true

