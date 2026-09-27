# 05 — Create `embed_entities.py` for batch embedding generation

**What to build:** A new script that loads the 12-column clean parquets, concatenates `business_name_for_embedding + " | " + business_address_for_embedding` per entity, encodes with the MiniLM model on GPU, and saves L2-normalized 384-dim embeddings as `.npy` files aligned by row order to the parquets.

**Blocked by:** 02 (Download embedding model), 04 (12-column parquets exist)

**Status:** ready-for-agent

## Detailed instructions

### Create file: `code/business_entity_resolution/src/embed_entities.py`

```python
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def embed_split(split='train', batch_size=256, max_seq_length=64):
    """
    Generates L2-normalized 384-dim embeddings for all entities in a split.
    Saves as .npy files aligned by row order to clean parquet files.
    """
    log("Loading embedding model on GPU...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
    model.max_seq_length = max_seq_length

    for source_num in [1, 2, 3]:
        parquet_path = f"dataset/{split}/{split}_source{source_num}_clean.parquet"
        out_path = f"dataset/{split}/{split}_source{source_num}_embeddings.npy"
        ids_path = f"dataset/{split}/{split}_source{source_num}_embed_ids.npy"

        log(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path, columns=[
            'entity_id', 'business_name_for_embedding', 'business_address_for_embedding'
        ])

        combined = (
            df['business_name_for_embedding'].fillna('') + ' | ' +
            df['business_address_for_embedding'].fillna('')
        ).tolist()

        log(f"Encoding {len(combined):,} records (batch_size={batch_size})...")
        t0 = time.time()
        embeddings = model.encode(
            combined,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device='cuda'
        ).astype(np.float32)
        log(f"Encoded in {time.time()-t0:.1f}s")

        np.save(out_path, embeddings)
        np.save(ids_path, df['entity_id'].to_numpy())
        log(f"Saved {out_path} ({embeddings.shape}) and {ids_path}")

        del df, combined, embeddings
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--batch-size', type=int, default=256,
                        help='If CUDA OOM, halve to 128 then 64. Do NOT switch to CPU.')
    args = parser.parse_args()
    embed_split(split=args.split, batch_size=args.batch_size)
```

### Important constraints

- `batch_size=256` is the default for 4GB VRAM RTX 3050 Ti. If CUDA OOM occurs, halve to 128, then 64. Do NOT increase beyond 256 and do NOT fall back to CPU.
- `max_seq_length=64` — business names/addresses are short strings; capping saves VRAM.
- `normalize_embeddings=True` — critical: L2-normalizes at source so dot product = cosine similarity in FAISS later.
- Save entity IDs alongside embeddings to enable order-alignment verification downstream.

### Output file layout

For each source in each split, two files are created:
```
dataset/{split}/{split}_source{N}_embeddings.npy   — shape (num_entities, 384), float32
dataset/{split}/{split}_source{N}_embed_ids.npy    — shape (num_entities,), entity_id strings
```

### Run and validate

1. Run train split first (smaller risk):
   ```powershell
   cd code/business_entity_resolution/src
   python embed_entities.py --split train
   ```

2. Validate:
   ```python
   import numpy as np
   emb = np.load('dataset/train/train_source1_embeddings.npy')
   ids = np.load('dataset/train/train_source1_embed_ids.npy')
   print(f"Shape: {emb.shape}")          # (N, 384)
   print(f"IDs shape: {ids.shape}")       # (N,)
   print(f"L2 norm of row 0: {np.linalg.norm(emb[0]):.6f}")  # should be ~1.0
   assert emb.shape[1] == 384
   assert np.allclose(np.linalg.norm(emb[0]), 1.0, atol=1e-4)
   ```

3. Then run test split:
   ```powershell
   python embed_entities.py --split test
   ```

- [ ] Script created at `code/business_entity_resolution/src/embed_entities.py`
- [ ] Train split: 3 embedding .npy files + 3 id .npy files created under `dataset/train/`
- [ ] Test split: 3 embedding .npy files + 3 id .npy files created under `dataset/test/`
- [ ] All embedding arrays have shape `(N, 384)` with float32 dtype
- [ ] L2 norms ≈ 1.0 (atol 1e-4) for spot-checked rows
- [ ] No CUDA OOM errors
