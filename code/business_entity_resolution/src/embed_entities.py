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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        num_threads = min(os.cpu_count() or 4, 8)
        torch.set_num_threads(num_threads)
        log(f"CUDA not detected. Running SentenceTransformer on CPU with {num_threads} threads...")
    else:
        torch.backends.cudnn.benchmark = True
        log(f"CUDA detected ({torch.cuda.get_device_name(0)}). Running SentenceTransformer with Tensor Core FP16 acceleration...")

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    model.max_seq_length = max_seq_length
    if device == 'cuda':
        model = model.half()

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
        entity_ids = df['entity_id'].to_numpy()
        np.save(ids_path, entity_ids)

        combined = (
            df['business_name_for_embedding'].fillna('') + ' | ' +
            df['business_address_for_embedding'].fillna('')
        ).tolist()

        del df, entity_ids
        import gc
        gc.collect()

        n_total = len(combined)
        chunk_step = 100_000
        total_chunks = (n_total + chunk_step - 1) // chunk_step
        log(f"Encoding {n_total:,} records across {total_chunks} memory-safe chunks (batch_size={batch_size}, FP16={device == 'cuda'})...")
        t0 = time.time()

        embeddings = np.empty((n_total, 384), dtype=np.float32)

        with torch.inference_mode():
            for chunk_idx, c_start in enumerate(range(0, n_total, chunk_step), 1):
                t_chunk = time.time()
                c_end = min(c_start + chunk_step, n_total)
                sub_texts = combined[c_start:c_end]
                pct = (c_end / n_total) * 100
                log(f"  [SOURCE{source_num} Chunk {chunk_idx}/{total_chunks}] Encoding records {c_start+1:,} to {c_end:,} ({pct:.1f}%)...")
                sub_emb = model.encode(
                    sub_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=device
                )
                embeddings[c_start:c_end] = sub_emb.astype(np.float32)
                chunk_time = time.time() - t_chunk
                elapsed = time.time() - t0
                rate = c_end / elapsed if elapsed > 0 else 0
                eta_s = (n_total - c_end) / rate if rate > 0 else 0
                log(f"  [SOURCE{source_num} Chunk {chunk_idx}/{total_chunks}] Done in {chunk_time:.1f}s | Avg rate: {rate:,.0f} rec/s | ETA: {eta_s/60:.1f} min")
                del sub_texts, sub_emb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        elapsed = time.time() - t0
        log(f"Encoded {n_total:,} in {elapsed:.1f}s ({n_total/elapsed:.0f} records/s)")

        np.save(out_path, embeddings)
        log(f"Saved {out_path} ({embeddings.shape}) and {ids_path}")

        del combined, embeddings
        gc.collect()
        if torch.cuda.is_available():
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
