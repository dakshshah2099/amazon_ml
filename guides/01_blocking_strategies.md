# Blocking & Candidate Generation Strategies

## Overview

Blocking reduces the O(N²) comparison space to O(N·k) candidates while maintaining high recall.
**Goal**: Pairs Completeness (Recall) > 98%, Reduction Ratio > 99.9%.

> [!CAUTION]
> Any true match missed in blocking can NEVER be recovered by downstream models.

---

## 1. Standard Equi-Blocking (Inverted Index)

**Approach**: Hash records by exact blocking key value (BKV). Only records with same BKV are compared.

```python
# Example BKVs for business ER:
# BKV1: country + postal_code_prefix_3 + soundex(first_word_of_name)
# BKV2: country + first_4_chars(clean_name)
# BKV3: country + exact_postal_code
```

| Metric | Value |
|---|---|
| Recall (clean fields) | 95–99% |
| Recall (noisy fields) | 50–75% ⚠️ |
| Cost | O(N) build |
| Vulnerability | Typos in key → total miss; block skew on common keys |

**Libraries**: `Splink 4`, `recordlinkage`, `pyJedAI`

---

## 2. TF-IDF Sparse Top-K (RECOMMENDED PRIMARY)

Best single blocking method for text-heavy business ER. Char n-grams are typo-resistant.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# Vectorize using char n-grams (robust to typos)
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 4),
    min_df=2,
    sublinear_tf=True
)

# Fit on all records (S1 + S2 + S3), transform separately
all_text = pd.concat([s1['full_text'], s2['full_text'], s3['full_text']])
vectorizer.fit(all_text)

s1_tfidf = vectorizer.transform(s1['full_text'])
s2s3_tfidf = vectorizer.transform(s2s3['full_text'])

# Sparse top-K: finds top 20 candidates per S1 entity with cosine > 0.4
candidates = sp_matmul_topn(
    A=s1_tfidf,
    B=s2s3_tfidf.T,
    top_n=20,
    threshold=0.40,
    sort=True
)
```

| Metric | Value |
|---|---|
| Recall | 93–98% |
| Reduction Ratio | 99.8–99.99% |
| Speed | 1M records in <60s on CPU |
| Vulnerability | Misses semantic synonyms, cross-language |

**Libraries**: `sparse_dot_topn`, `scikit-learn`

---

## 3. Dense Bi-Encoder ANN (Semantic Blocking)

Overcomes lexical mismatch (abbreviations, synonyms, translations).

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Serialize entities
s1_texts = [f"Name: {n} | Address: {a} | Country: {c}" 
            for n, a, c in zip(s1.name, s1.address, s1.country)]
s2s3_texts = [f"Name: {n} | Address: {a} | Country: {c}" 
              for n, a, c in zip(s2s3.name, s2s3.address, s2s3.country)]

# Encode
s1_emb = model.encode(s1_texts, normalize_embeddings=True, batch_size=256)
s2s3_emb = model.encode(s2s3_texts, normalize_embeddings=True, batch_size=256)

# Build FAISS HNSW index on S2+S3
d = s2s3_emb.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efSearch = 64
index.hnsw.efConstruction = 128
index.add(s2s3_emb.astype(np.float32))

# Query S1 against index
k = 25
distances, indices = index.search(s1_emb.astype(np.float32), k)
```

### Recommended Models (offline, open license)

| Model | Dims | License | Multilingual | Size |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | MIT | No (English) | 80MB |
| `BAAI/bge-m3` | 1024 | Apache 2.0 | Yes (100+ langs) | ~2GB |
| `intfloat/multilingual-e5-small` | 384 | MIT | Yes | ~450MB |
| `BAAI/bge-small-en-v1.5` | 384 | MIT | No | ~130MB |

| Metric | Value |
|---|---|
| Recall | 93–99% |
| Speed | Encoding is GPU bottleneck; search is O(log N) |
| Vulnerability | Misses exact alphanumeric IDs; needs GPU for encoding |

**Libraries**: `sentence-transformers`, `faiss-cpu`, `hnswlib`

---

## 4. MinHash LSH (Jaccard Similarity)

Hash-based sub-linear candidate generation. Good for very large datasets.

```python
from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.4, num_perm=128)

def text_to_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    # Use char 3-grams as shingles
    text = text.lower().strip()
    for i in range(len(text) - 2):
        m.update(text[i:i+3].encode('utf8'))
    return m

# Insert S2+S3 into LSH
for idx, text in enumerate(s2s3_texts):
    m = text_to_minhash(text)
    lsh.insert(f"s2s3_{idx}", m)

# Query S1
for idx, text in enumerate(s1_texts):
    m = text_to_minhash(text)
    candidates = lsh.query(m)
```

| Metric | Value |
|---|---|
| Recall | 85–96% |
| Speed | O(N) build, sub-linear query |
| Vulnerability | Threshold boundary false negatives |

---

## 5. Hybrid Multi-Pass (RECOMMENDED STRATEGY)

Union of multiple blocking methods maximizes recall. **This is the winning approach.**

```python
# Final candidate set = Union of all methods
candidates_final = set()

# Pass 1: Deterministic rule blocking
for s1_id in s1_ids:
    for s2s3_id in s2s3_ids:
        if same_country(s1_id, s2s3_id) and same_postal_prefix(s1_id, s2s3_id, n=3):
            if soundex_match(s1_id, s2s3_id):
                candidates_final.add((s1_id, s2s3_id))

# Pass 2: TF-IDF sparse top-K
candidates_final |= tfidf_candidates

# Pass 3: Dense ANN top-K  
candidates_final |= dense_candidates

# Pass 4: MinHash LSH (optional, for coverage)
candidates_final |= lsh_candidates

# Deduplicate and cap at max 50 per S1 entity
```

| Metric | Value |
|---|---|
| Recall | **97–99.8%** |
| Reduction Ratio | 99.9–99.99% |

---

## 6. Country-Aware Blocking

### Partition by Country First
```python
# Block within country to shrink search space ~50-200×
for country in ['US', 'India', 'France']:
    s1_country = s1[s1.country == country]
    s2s3_country = s2s3[s2s3.country == country]
    # Run blocking within this partition
```

### Country-Specific Postal Anchoring
- **US**: ZIP 5-digit → first 3 digits = sectional center
- **India**: PIN 6-digit → first 3 digits = sorting district  
- **France**: 5-digit → first 2 digits = département

### Multilingual Considerations
- Strip French diacritics via Unicode NFKD
- Handle Indian transliteration variants (Shri/Shree/Sri, Agarwal/Aggarwal/Agrawal)
- Use `anyascii` for Unicode→ASCII transliteration
- Use multilingual embeddings (`bge-m3`) for cross-script matching

---

## 7. Block Cleaning & Meta-Blocking

After generating candidates, clean to remove noise:

1. **Block Purging**: Remove blocks with >500 records (common-name blocks)
2. **Block Filtering**: For each entity, keep only smallest (most informative) blocks
3. **Cardinality Capping**: Max 50 candidates per S1 entity
4. **Comparison Propagation**: Deduplicate (min(i,j), max(i,j)) across passes

---

## Quick Reference: Strategy Comparison

| Strategy | Recall | RR | Speed | Best For |
|---|---|---|---|---|
| Equi-Blocking | 60–95% | 99.0–99.9% | O(N) | Clean structured data |
| Sorted Neighborhood | 75–92% | 99.5–99.99% | O(N log N) | Fixed compute budget |
| Q-Gram | 92–98% | 98.0–99.5% | O(N·L) | Heavy typo corruption |
| **TF-IDF Sparse Top-K** | **93–98%** | **99.8–99.99%** | **Very fast** | **Primary method** |
| **Dense ANN** | **94–99%** | **99.5–99.99%** | GPU needed | **Semantic matching** |
| MinHash LSH | 85–96% | 99.5–99.99% | O(N) | Very large datasets |
| **Hybrid Multi-Pass** | **97–99.8%** | **99.9–99.99%** | Sum of parts | **Competition winner** |
