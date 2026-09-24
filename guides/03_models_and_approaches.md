# ML Models & Approaches for Entity Resolution

## Overview

Modern ER uses a multi-stage architecture. This guide covers all viable model choices.

---

## 1. Architecture: Two-Stage Pipeline

```
[Raw Records]
      │
      ▼
[Stage 1: Blocking (Bi-Encoder)]     → High recall, sub-quadratic
  • Sentence-Transformers / TF-IDF   → Candidate pairs
      │
      ▼
[Stage 2: Matching (Cross-Encoder OR GBDT)]  → High precision
  • DeBERTa-v3 cross-encoder, OR
  • LightGBM on handcrafted features
      │
      ▼
[Stage 3: Post-Processing]
  • F₀.₅ threshold tuning
  • Singleton gate
  • Reciprocal NN verification
```

---

## 2. GBDT Classifier (RECOMMENDED for this challenge)

Best option given constraints (≤8B params, reproducibility, speed).

```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

# Train features: X_train (N_pairs × 28 features), y_train (binary)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 7,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': neg_count / pos_count,  # Handle imbalance
    'verbose': -1,
}

model = lgb.LGBMClassifier(**params, n_estimators=1000)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]
```

### Why GBDT over Deep Learning here?
- Handles missing values natively (some addresses lack postal codes)
- Captures non-linear feature interactions (postal match + high name JW → strong match)
- Fast training/inference, no GPU needed
- Easily interpretable feature importances
- Works well with ~30 handcrafted features

---

## 3. Cross-Encoder (DeBERTa-v3) — Higher Accuracy Option

Serializes both records into a single sequence with full cross-attention.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "microsoft/deberta-v3-base"  # 86M params, MIT license
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def serialize_pair(rec1, rec2):
    """DITTO-style serialization with COL/VAL markers."""
    s1 = f"[COL] name [VAL] {rec1['name']} [COL] address [VAL] {rec1['address']} [COL] country [VAL] {rec1['country']}"
    s2 = f"[COL] name [VAL] {rec2['name']} [COL] address [VAL] {rec2['address']} [COL] country [VAL] {rec2['country']}"
    return s1, s2

# Fine-tuning
inputs = tokenizer(text_a, text_b, padding=True, truncation=True, max_length=256, return_tensors="pt")
outputs = model(**inputs, labels=labels)
loss = outputs.loss
```

### Why DeBERTa-v3 is best cross-encoder for ER:
- **Disentangled attention**: Separate content + position vectors → better at structured text
- **Enhanced Mask Decoder**: Better at tabular serialization patterns
- `deberta-v3-base` (86M) or `deberta-v3-large` (435M) — both within 8B limit

---

## 4. Bi-Encoder (Sentence-Transformers) — For Blocking

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

# Fine-tune with contrastive loss on training matches
train_examples = [
    InputExample(texts=[serialize(rec_s1), serialize(rec_match)], label=1.0),
    InputExample(texts=[serialize(rec_s1), serialize(rec_nonmatch)], label=0.0),
]

train_dl = DataLoader(train_examples, batch_size=64, shuffle=True)
loss_fn = losses.ContrastiveLoss(model)

model.fit(
    train_objectives=[(train_dl, loss_fn)],
    epochs=3,
    warmup_steps=100,
)
```

---

## 5. Key Papers & Frameworks

### DITTO (PVLDB 2021)
- **Idea**: Cross-encoder ER using RoBERTa with domain knowledge injection, TF-IDF summarization, and MixDA augmentation
- **Repo**: `megagonlabs/ditto`
- **License**: Apache 2.0
- **F1**: 85–99% across Magellan benchmarks

### PromptEM (ACL 2022)
- **Idea**: Cloze-style prompt tuning instead of classification head. High sample efficiency (<100 labels)
- **Best for**: Few-shot settings

### ComEM (COLING 2025)
- **Idea**: Compound Match-Compare-Select paradigm. Avoids pairwise explosion and transitive inconsistency
- **Models**: Llama 3 8B, Mistral 7B, Qwen 2.5
- **License**: Open source

### AvengER (ESWC 2025)
- **Idea**: 8B base matcher + 32B judge for ambiguous cases (~12% traffic). 99% of large-model F1 at 15% compute
- **License**: Apache 2.0

### Sudowoodo (VLDB 2023)
- **Idea**: Contrastive self-supervised learning with tabular augmentations (cell dropping, typo injection)
- **Best for**: Unsupervised / few-label settings

---

## 6. LLM Fine-Tuning (≤8B, Open License)

### Viable Models

| Model | Params | License | Notes |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Llama 3.1 Community | Top choice for LoRA fine-tuning |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Apache 2.0 | Strong multilingual |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Apache 2.0 | Good balance |
| `google/gemma-2-9b-it` | 9B | Gemma | ⚠️ Check license compliance |

### LoRA Fine-Tuning Example

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", load_in_4bit=True)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training prompt format
PROMPT = """Determine if these two business records refer to the same entity.
Record A: Name: {name1} | Address: {addr1} | Country: {country1}
Record B: Name: {name2} | Address: {addr2} | Country: {country2}
Answer (yes/no):"""
```

### Zero-Shot vs Fine-Tuned Performance

| Approach | Accuracy | Latency | Cost |
|---|---|---|---|
| Zero-shot GPT-4 | Moderate | 500-2500ms/pair | High |
| Zero-shot 8B | Low-Moderate | 50-200ms/pair | Medium |
| **Fine-tuned 8B (LoRA)** | **Highest** | **15-80ms/pair** | **Low** |
| **DeBERTa-v3 cross-encoder** | **Highest** | **5-20ms/pair** | **Low** |

---

## 7. Recommended Approach for This Challenge

### Option A: GBDT Pipeline (Simpler, Proven)
```
TF-IDF + Dense Blocking → 30 handcrafted features → LightGBM → F₀.₅ threshold
```
**Pros**: Fast, reproducible, interpretable, no GPU needed
**Cons**: May miss deep semantic patterns

### Option B: Cross-Encoder Pipeline (Higher Ceiling)
```
Dense Blocking → DeBERTa-v3-base cross-encoder → F₀.₅ threshold
```
**Pros**: Higher accuracy on ambiguous cases
**Cons**: Needs GPU, slower inference

### Option C: Hybrid (Best of Both)
```
Hybrid Blocking → Feature extraction + DeBERTa scores → LightGBM ensemble → F₀.₅ threshold
```
**Pros**: Combines handcrafted + learned features
**Cons**: Most complex to implement

> [!TIP]
> Start with Option A (GBDT) to establish a strong baseline, then layer in Option B/C for marginal gains.

---

## 8. Open-Source Library Status (2025-2026)

| Library | License | Status | Backend | Best For |
|---|---|---|---|---|
| **Splink 4** | MIT | ✅ Active | DuckDB/Spark | Probabilistic ER at scale |
| **pyJedAI** | Apache 2.0 | ✅ Active | In-memory/FAISS | Research, hybrid pipelines |
| **Zingg** | AGPL-3.0 | ✅ Active | Spark | Active learning ER |
| dedupe | MIT | ⚠️ Maintenance | Python only | Small datasets |
| DeepMatcher | BSD-3 | ❌ Archived | PyTorch legacy | Historical reference only |
| Magellan | BSD-3 | ❌ Archived | Pandas | Benchmark datasets only |
