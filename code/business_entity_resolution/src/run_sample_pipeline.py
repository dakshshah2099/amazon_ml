"""
End-to-end rapid sample runner (50k sample).
Runs blocking, tuning, matcher training (5-fold CV), inference, and evaluation.
"""
import os
import sys
import time

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def main():
    t_start = time.time()
    log("==================================================")
    log("RUNNING RAPID 50K SAMPLE PIPELINE END-TO-END")
    log("==================================================")

    data_dir = 'dataset/sample50k'
    prefix = 'sample'
    out_dir = 'output_sample50k'
    models_dir = 'models_sample50k'
    gt_path = 'dataset/sample50k/sample_ground_truth.tsv'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Tune blocking thresholds on sample
    log("Step 1: Running blocking threshold tuning on sample...")
    from blocking import tune_blocking
    tune_blocking(
        sample_size=5000,
        score_grid=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
        k=30,
        data_dir=data_dir,
        prefix=prefix,
        gt_path=gt_path
    )

    # 2. Run blocking on sample
    log("\nStep 2: Running blocking on sample...")
    from blocking import run_blocking
    run_blocking(
        split='sample',
        out_dir=out_dir,
        k=20,
        min_score=0.40,
        max_bucket_size=30,
        data_dir=data_dir,
        prefix=prefix
    )

    # 3. Train Matcher with 5-fold CV
    log("\nStep 3: Training Matcher (5-fold CV) on sample...")
    from train_matcher import train_matcher
    train_matcher(
        gt_path=gt_path,
        cand_detailed_path=os.path.join(out_dir, 'sample_candidate_pairs_detailed.tsv'),
        num_entities_sample=35_000,
        neg_ratio=3,
        n_folds=5,
        random_seed=None,
        data_dir=data_dir,
        prefix=prefix,
        models_dir=models_dir
    )

    # 4. Predict Matches on Sample & Evaluate
    log("\nStep 4: Running predict_matches on sample...")
    from predict_matches import predict_matches, evaluate
    pred_out = os.path.join(out_dir, 'sample_matching_results.tsv')
    predict_matches(
        split='sample',
        cand_dir=out_dir,
        model_lgb_path=os.path.join(models_dir, 'lgb_matcher.txt'),
        model_cb_path=os.path.join(models_dir, 'cb_matcher.cbm'),
        meta_path=os.path.join(models_dir, 'matcher_metadata.pkl'),
        out_path=pred_out,
        num_workers=8,
        data_dir=data_dir,
        prefix=prefix,
        eval_gt_path=gt_path
    )

    # 5. Report summary
    p, r, f05 = evaluate(pred_out, gt_path, split_name='sample_50k_eval')

    elapsed = time.time() - t_start
    log(f"\nRAPID SAMPLE PIPELINE COMPLETED IN {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log(f"FINAL SAMPLE RESULTS -> Precision: {p*100:.2f}%, Recall: {r*100:.2f}%, F0.5: {f05*100:.2f}%")

if __name__ == '__main__':
    main()
