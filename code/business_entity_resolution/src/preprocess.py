import os
import sys
import time
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from cleaning_utils import clean_batch

FILES = [
    ('dataset/train/train_source1.tsv', 'dataset/train/train_source1_clean.parquet'),
    ('dataset/train/train_source2.tsv', 'dataset/train/train_source2_clean.parquet'),
    ('dataset/train/train_source3.tsv', 'dataset/train/train_source3_clean.parquet'),
    ('dataset/test/test_source1.tsv', 'dataset/test/test_source1_clean.parquet'),
    ('dataset/test/test_source2.tsv', 'dataset/test/test_source2_clean.parquet'),
    ('dataset/test/test_source3.tsv', 'dataset/test/test_source3_clean.parquet'),
]

OUTPUT_COLS = [
    'entity_id',
    'country',
    'business_name',
    'business_address',
    'business_name_clean',
    'business_address_clean',
    'postal_code',
    'has_state',
    'needs_transliteration_name',
    'needs_transliteration_address',
    'business_name_for_embedding',
    'business_address_for_embedding'
]

def process_file(in_path, out_path, chunksize=100000, batch_size=5000, max_workers=10):
    print(f"\n==================================================")
    print(f"Processing: {in_path} -> {out_path}")
    print(f"==================================================")
    start_time = time.time()
    
    total_rows = 0
    total_trans_name = 0
    total_trans_addr = 0
    total_postal = 0
    total_has_state = 0
    
    collected_examples = []
    trans_examples = []
    
    # Use PyArrow ParquetWriter to write chunks sequentially
    writer = None
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for chunk_idx, df_chunk in enumerate(pd.read_csv(in_path, sep='\t', chunksize=chunksize, dtype=str, keep_default_na=False)):
                records = list(zip(
                    df_chunk['entity_id'],
                    df_chunk['country'],
                    df_chunk['business_name'],
                    df_chunk['business_address']
                ))
                
                # Split chunk into sub-batches for executor
                sub_batches = [records[i:i+batch_size] for i in range(0, len(records), batch_size)]
                cleaned_sub_batches = list(executor.map(clean_batch, sub_batches))
                
                # Flatten
                flat_rows = [r for sub in cleaned_sub_batches for r in sub]
                
                # Aggregate stats
                for r in flat_rows:
                    total_rows += 1
                    if r[6]:  # postal_code
                        total_postal += 1
                    if r[7]:  # has_state
                        total_has_state += 1
                    if r[8]:  # needs_trans_name
                        total_trans_name += 1
                    if r[9]:  # needs_trans_addr
                        total_trans_addr += 1
                    
                    # Collect transliterated examples
                    if (r[8] or r[9]) and len(trans_examples) < 10:
                        trans_examples.append(r)
                    
                    # Reservoir/simple sampling for general examples
                    if len(collected_examples) < 100:
                        collected_examples.append(r)
                    elif random.random() < 0.0005 and len(collected_examples) < 200:
                        collected_examples.append(r)
                
                # Build DataFrame and write to parquet
                chunk_df = pd.DataFrame(flat_rows, columns=OUTPUT_COLS)
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                writer.write_table(table)
                
                elapsed = time.time() - start_time
                print(f"  Chunk {chunk_idx + 1} processed: {total_rows} rows so far ({total_rows/elapsed:.0f} rows/s)")
    finally:
        if writer is not None:
            writer.close()
            
    elapsed = time.time() - start_time
    print(f"\nCompleted {in_path} in {elapsed:.2f}s ({total_rows/elapsed:.0f} rows/s)")
    print(f"Summary Stats:")
    print(f"  Total row count:                  {total_rows:,}")
    print(f"  Needs transliteration (name):     {total_trans_name:,} ({total_trans_name/total_rows*100:.2f}%)")
    print(f"  Needs transliteration (address):  {total_trans_addr:,} ({total_trans_addr/total_rows*100:.2f}%)")
    print(f"  Has postal_code extracted:        {total_postal:,} ({total_postal/total_rows*100:.2f}%)")
    print(f"  Has state identified (has_state): {total_has_state:,} ({total_has_state/total_rows*100:.2f}%)")
    
    # Select 10 random before/after examples, ensuring at least 3 transliteration examples if available
    chosen_examples = []
    num_trans_to_pick = min(len(trans_examples), 3)
    if num_trans_to_pick > 0:
        chosen_examples.extend(random.sample(trans_examples, num_trans_to_pick))
    
    remaining_slots = 10 - len(chosen_examples)
    non_trans_candidates = [e for e in collected_examples if e not in chosen_examples]
    if len(non_trans_candidates) >= remaining_slots:
        chosen_examples.extend(random.sample(non_trans_candidates, remaining_slots))
    else:
        chosen_examples.extend(non_trans_candidates)
        
    print(f"\n--- 10 Random Before / After Examples for {os.path.basename(in_path)} ---")
    for idx, ex in enumerate(chosen_examples, 1):
        print(f"\nExample {idx}: [ID: {ex[0]}, Country: {ex[1]}]")
        print(f"  RAW NAME:     {ex[2]}")
        print(f"  CLEAN NAME:   {ex[4]}")
        print(f"  RAW ADDR:     {ex[3]}")
        print(f"  CLEAN ADDR:   {ex[5]}")
        print(f"  POSTAL CODE:  '{ex[6]}'")
        print(f"  HAS STATE:    {ex[7]}")
        print(f"  TRANS NAME:   {ex[8]} | TRANS ADDR: {ex[9]}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'test'],
                        help='Split to process: train, test, or all')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes')
    args = parser.parse_args()

    random.seed(42)
    overall_start = time.time()

    if args.split == 'train':
        target_files = [f for f in FILES if 'train' in f[0]]
    elif args.split == 'test':
        target_files = [f for f in FILES if 'test' in f[0]]
    else:
        target_files = FILES

    for in_path, out_path in target_files:
        if not os.path.exists(in_path):
            print(f"Error: {in_path} does not exist!")
            sys.exit(1)
        
        # Check if already processed with 12 columns
        if os.path.exists(out_path):
            try:
                schema = pq.read_schema(out_path)
                if len(schema.names) == len(OUTPUT_COLS) and schema.names[-1] == OUTPUT_COLS[-1]:
                    print(f"Skipping {out_path} — already processed with {len(OUTPUT_COLS)} columns.")
                    continue
            except Exception:
                pass

        process_file(in_path, out_path, max_workers=args.workers)
    print(f"\n==================================================")
    print(f"ALL REQUESTED FILES PROCESSED SUCCESSFULLY in {time.time() - overall_start:.2f}s!")
    print(f"==================================================")

if __name__ == '__main__':
    main()
