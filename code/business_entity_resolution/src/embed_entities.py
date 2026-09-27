import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def embed_split(split='train', batch_size=256, max_seq_length=64, data_dir=None, prefix=None):
    """
    Generates L2-normalized 384-dim embeddings for all entities in a split.
    Saves as .npy files aligned by row order to clean parquet files.
    """
    if data_dir is None:
        data_dir = f"dataset/{split}"
    if prefix is None:
        prefix = split

    log("Loading embedding model on GPU...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
    model.max_seq_length = max_seq_length

    for source_num in [1, 2, 3]:
        parquet_path = f"{data_dir}/{prefix}_source{source_num}_clean.parquet"
        out_path = f"{data_dir}/{prefix}_source{source_num}_embeddings.npy"
        ids_path = f"{data_dir}/{prefix}_source{source_num}_embed_ids.npy"

        if os.path.exists(out_path) and os.path.exists(ids_path):
            log(f"Skipping {out_path} — already generated.")
            continue

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
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='If CUDA OOM, halve to 128 then 64. Do NOT switch to CPU.')
    args = parser.parse_args()
    embed_split(
        split=args.split,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        prefix=args.prefix
    )
