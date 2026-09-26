import os, time, re, unicodedata, gc, warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz, distance
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import joblib

warnings.filterwarnings('ignore', category=UserWarning)

# DATA_DIR/CKPT_DIR can be overridden with env vars so this runs unmodified on Colab
# (e.g. `DATA_DIR=/content/dataset` after copying the dataset off Google Drive onto local disk).
DATA_DIR = os.environ.get('DATA_DIR', 'dataset')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
CKPT_DIR = os.environ.get('CKPT_DIR', 'experiments/checkpoints')
VAL_SAMPLE_SIZE = int(os.environ.get('VAL_SAMPLE_SIZE', 50000))
os.makedirs('experiments/models', exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
t_run_start = time.time()

# 1. Universal Country-Agnostic Normalizer
LEGAL_SUFFIXES = {
    r'\b(pvt\.?\s*ltd\.?|private\s+limited)\b': ' PVT_LTD ',
    r'\b(ltd\.?|limited)\b': ' LTD ',
    r'\b(llp|limited\s+liability\s+partnership)\b': ' LLP ',
    r'\b(opc|one\s+person\s+company)\b': ' OPC ',
    r'\b(inc\.?|incorporated)\b': ' INC ',
    r'\b(corp\.?|corporation)\b': ' CORP ',
    r'\b(llc|l\.l\.c\.)\b': ' LLC ',
    r'\b(co\.?|company)\b': ' CO ',
    r'\b(sarl|societe\s+a\s+responsabilite\s+limitee)\b': ' SARL ',
    r'\b(sas|societe\s+par\s+actions\s+simplifiee)\b': ' SAS ',
    r'\b(sasu)\b': ' SASU ',
    r'\b(sa|societe\s+anonyme)\b': ' SA ',
    r'\b(sci|societe\s+civile\s+immobiliere)\b': ' SCI ',
    r'\b(eurl)\b': ' EURL ',
}

ABBREV_EXPANSIONS = {
    r'\bintl\.?\b': 'international',
    r'\bmfg\.?\b': 'manufacturing',
    r'\bsvcs?\.?\b': 'services',
    r'\btech\.?\b': 'technologies',
    r'\bdept\.?\b': 'department',
    r'\bassoc\.?\b': 'associates',
    r'\b&\b': ' and ',
    r'\b\+\b': ' and ',
}

ROAD_EXPANSIONS = {
    r'\brd\.?\b': 'road',
    r'\bst\.?\b': 'street',
    r'\bave?\.?\b': 'avenue',
    r'\bblvd\.?\b': 'boulevard',
    r'\bdr\.?\b': 'drive',
    r'\bln\.?\b': 'lane',
    r'\bct\.?\b': 'court',
    r'\bpl\.?\b': 'place',
    r'\bhwy\.?\b': 'highway',
    r'\bp\.?o\.?\s*box\b': 'pobox',
    r'\bbd\b|\bbvd\b': 'boulevard',
    r'\brte\b': 'route',
    r'\ball\b': 'allee',
    r'\bimp\b': 'impasse',
}

RE_POSTAL = re.compile(r'\b[1-9]\d{2}\s?\d{3}\b|\b\d{5}(?:-\d{4})?\b')
RE_URL = re.compile(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
RE_LANDMARKS = re.compile(r'\b(near|opp|opposite|behind|beside|adj|adjacent to|next to)\s+[\w\s]+?(?=,|\\.|$)', re.IGNORECASE)

def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ''
    text = text.replace('œ', 'oe').replace('æ', 'ae').replace('Œ', 'oe').replace('Æ', 'ae')
    nfkd = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()
    m_url = RE_URL.search(text)
    if m_url:
        domain = m_url.group(1)
        base = domain.split('.')[0]
        text = text.replace(domain, base)
    return text

def clean_business_name(name: str) -> tuple[str, str]:
    text = normalize_text(name)
    if not text:
        return '', 'NONE'
    detected_suffix = 'NONE'
    for pattern, canonical in LEGAL_SUFFIXES.items():
        if re.search(pattern, text):
            detected_suffix = canonical.strip()
            text = re.sub(pattern, ' ', text)
            break
    for pattern, replacement in ABBREV_EXPANSIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip(), detected_suffix

def clean_address(addr: str) -> tuple[str, str, str]:
    if not isinstance(addr, str) or not addr:
        return '', '', ''
    m_post = RE_POSTAL.search(addr)
    postal_code = m_post.group(0).replace(' ', '')[:5] if m_post else ''
    text = normalize_text(addr)
    bldg_m = re.search(r'\b(?:no\.?|plot\s*no\.?|#)?\s*(\d+[a-z]?(?:/\d+)?)\b', text)
    bldg_num = bldg_m.group(1) if bldg_m else ''
    text = RE_LANDMARKS.sub(' ', text)
    for pattern, replacement in ROAD_EXPANSIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip(), postal_code, bldg_num

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    names = df['business_name'].tolist()
    addrs = df['business_address'].tolist()
    clean_names, suffixes = [], []
    clean_addrs, postals, bldgs = [], [], []
    for n, a in zip(names, addrs):
        cn, suf = clean_business_name(n)
        ca, p, b = clean_address(a)
        clean_names.append(cn)
        suffixes.append(suf)
        clean_addrs.append(ca)
        postals.append(p)
        bldgs.append(b)
    df['clean_name'] = clean_names
    df['legal_suffix'] = suffixes
    df['clean_addr'] = clean_addrs
    df['postal_code'] = postals
    df['building_num'] = bldgs
    df['country'] = df['country'].fillna('UNKNOWN').astype(str).str.strip()
    return df

# 2. Load Validation Split (size controlled by VAL_SAMPLE_SIZE)
print("="*60)
print(f"PHASE 1: Building {VAL_SAMPLE_SIZE:,}-Record Holdout Validation Split")
print("="*60)
t0 = time.time()
gt_all = pd.read_csv(f'{DATA_DIR}/train/train_ground_truth.tsv', sep='\t')
gt_sample = gt_all.head(VAL_SAMPLE_SIZE).copy()

# Vectorized parse (was a 50k-row .iterrows() loop; this is >50x faster and scales to the full 2.2M-row file)
match_lists = gt_sample['matched_entity_ids'].fillna('').astype(str).apply(
    lambda m: [x.strip() for x in m.split(',') if x.strip()]
)
gt_mapping = dict(zip(gt_sample['source1_entity_id'], match_lists.apply(set)))
all_match_ids = set().union(*match_lists) if len(match_lists) else set()

target_s1_ids = set(gt_sample['source1_entity_id'])
singletons_count = sum(1 for m in gt_mapping.values() if not m)
print(f"{VAL_SAMPLE_SIZE:,} S1 sample contains {len(all_match_ids):,} true match IDs across {len(gt_sample)-singletons_count:,} linked entities and {singletons_count:,} singletons ({singletons_count/len(gt_sample)*100:.2f}%)")

# S1 records
s1_records = []
for chunk in pd.read_csv(f'{DATA_DIR}/train/train_source1.tsv', sep='\t', chunksize=200000):
    sub = chunk[chunk['entity_id'].isin(target_s1_ids)]
    if len(sub):
        s1_records.append(sub)
    if sum(len(x) for x in s1_records) >= len(target_s1_ids):
        break
val_s1 = pd.concat(s1_records, ignore_index=True)
val_s1 = preprocess_dataframe(val_s1)

# S2 and S3 true matches + background negatives
s2_records, s3_records = [], []
for chunk in pd.read_csv(f'{DATA_DIR}/train/train_source2.tsv', sep='\t', chunksize=500000):
    m_sub = chunk[chunk['entity_id'].isin(all_match_ids)]
    if len(m_sub):
        s2_records.append(m_sub)
    if len(s2_records) < 2:
        s2_records.append(chunk.head(30000))

for chunk in pd.read_csv(f'{DATA_DIR}/train/train_source3.tsv', sep='\t', chunksize=500000):
    m_sub = chunk[chunk['entity_id'].isin(all_match_ids)]
    if len(m_sub):
        s3_records.append(m_sub)
    if len(s3_records) < 2:
        s3_records.append(chunk.head(30000))

val_s23 = pd.concat(s2_records + s3_records, ignore_index=True).drop_duplicates('entity_id')
val_s23 = preprocess_dataframe(val_s23)
loaded_matches = len(all_match_ids & set(val_s23['entity_id']))
print(f"Validation pool loaded in {time.time()-t0:.2f}s: {len(val_s1):,} S1, {len(val_s23):,} S2/S3 (contains {loaded_matches:,}/{len(all_match_ids):,} matches, {loaded_matches/len(all_match_ids)*100:.2f}%)")

# 3. Dynamic Country-Partitioned Candidate Blocking
print("\n" + "="*60)
print("PHASE 2: Country-Partitioned Candidate Generation")
print("="*60)
# Tuned to the validated faster/higher-recall config from notes.md (was 45/0.20, 25/0.25, cap 65,
# which generates ~40% more candidate pairs than needed and slows every downstream phase).
def generate_candidates_for_country(s1_sub, s23_sub, top_n_name=35, thresh_name=0.25, top_n_addr=20, thresh_addr=0.30, max_total_cands=50):
    if len(s1_sub) == 0 or len(s23_sub) == 0:
        return pd.DataFrame(columns=['s1_idx', 's23_idx', 'tfidf_name_sim', 'tfidf_addr_sim'])
    vec_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=50000, sublinear_tf=True, dtype=np.float32)
    vec_name.fit(pd.concat([s1_sub['clean_name'], s23_sub['clean_name']]))
    res_name = sp_matmul_topn(vec_name.transform(s1_sub['clean_name']), vec_name.transform(s23_sub['clean_name']).T, top_n=top_n_name, threshold=thresh_name, n_threads=8)
    
    vec_addr = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=50000, sublinear_tf=True, dtype=np.float32)
    vec_addr.fit(pd.concat([s1_sub['clean_addr'], s23_sub['clean_addr']]))
    res_addr = sp_matmul_topn(vec_addr.transform(s1_sub['clean_addr']), vec_addr.transform(s23_sub['clean_addr']).T, top_n=top_n_addr, threshold=thresh_addr, n_threads=8)
    
    coo_n = res_name.tocoo()
    coo_a = res_addr.tocoo()
    df_n = pd.DataFrame({'s1_idx': coo_n.row, 's23_idx': coo_n.col, 'tfidf_name_sim': coo_n.data, 'tfidf_addr_sim': 0.0})
    df_a = pd.DataFrame({'s1_idx': coo_a.row, 's23_idx': coo_a.col, 'tfidf_name_sim': 0.0, 'tfidf_addr_sim': coo_a.data})
    
    pairs = pd.concat([df_n, df_a], ignore_index=True)
    pairs = pairs.groupby(['s1_idx', 's23_idx'], as_index=False)[['tfidf_name_sim', 'tfidf_addr_sim']].max()
    pairs['max_sim'] = np.maximum(pairs['tfidf_name_sim'], pairs['tfidf_addr_sim'])
    del vec_name, vec_addr, res_name, res_addr, coo_n, coo_a, df_n, df_a
    return pairs.sort_values(['s1_idx', 'max_sim'], ascending=[True, False]).groupby('s1_idx').head(max_total_cands).drop(columns=['max_sim']).reset_index(drop=True)

pairs_ckpt = f'{CKPT_DIR}/val_pairs_{VAL_SAMPLE_SIZE}.parquet'
if os.path.exists(pairs_ckpt):
    print(f"Found checkpoint, skipping blocking: {pairs_ckpt}")
    val_pairs = pd.read_parquet(pairs_ckpt)
else:
    candidate_pairs_list = []
    for country_val, s1_c in val_s1.groupby('country', sort=False):
        s1_c = s1_c.reset_index(drop=True)
        s23_c = val_s23[val_s23['country'] == country_val].reset_index(drop=True)
        print(f"Blocking [{country_val}]: {len(s1_c):,} S1 entities against {len(s23_c):,} candidate records...")
        t_blk = time.time()
        cands_c = generate_candidates_for_country(s1_c, s23_c)
        cands_c['source1_entity_id'] = s1_c.iloc[cands_c['s1_idx'].values]['entity_id'].values
        cands_c['cand_entity_id'] = s23_c.iloc[cands_c['s23_idx'].values]['entity_id'].values
        candidate_pairs_list.append(cands_c[['source1_entity_id', 'cand_entity_id', 'tfidf_name_sim', 'tfidf_addr_sim']])
        print(f"  Generated {len(cands_c):,} pairs in {time.time()-t_blk:.2f}s")
        del s1_c, s23_c, cands_c
        gc.collect()

    val_pairs = pd.concat(candidate_pairs_list, ignore_index=True)
    val_pairs.to_parquet(pairs_ckpt)
    print(f"Checkpoint saved: {pairs_ckpt}")
print(f"Total candidate pairs: {len(val_pairs):,}")

# Blocking Recall
cands_dict = val_pairs.groupby('source1_entity_id')['cand_entity_id'].apply(set).to_dict()
eval_matches = sum(len(m) for m in gt_mapping.values())
eval_recalled = sum(len(m & cands_dict.get(s1, set())) for s1, m in gt_mapping.items())
blocking_recall = eval_recalled / max(1, eval_matches)
print(f"Blocking Recall across ALL {eval_matches:,} true matches: {eval_recalled:,}/{eval_matches:,} ({blocking_recall*100:.2f}%)")

# 4. Feature Engineering
print("\n" + "="*60)
print("PHASE 3: Pairwise Feature Engineering (23 Discriminators)")
print("="*60)
s1_cols = {'clean_name': 's1_name', 'clean_addr': 's1_addr', 'postal_code': 's1_postal', 'building_num': 's1_bldg', 'legal_suffix': 's1_suffix'}
s23_cols = {'clean_name': 's23_name', 'clean_addr': 's23_addr', 'postal_code': 's23_postal', 'building_num': 's23_bldg', 'legal_suffix': 's23_suffix'}

t_fe = time.time()
merged_pairs = val_pairs.merge(val_s1[['entity_id', *s1_cols.keys()]].rename(columns=s1_cols), left_on='source1_entity_id', right_on='entity_id').drop(columns=['entity_id'])
merged_pairs = merged_pairs.merge(val_s23[['entity_id', *s23_cols.keys()]].rename(columns=s23_cols), left_on='cand_entity_id', right_on='entity_id').drop(columns=['entity_id'])

s1_names = merged_pairs['s1_name'].fillna('').astype(str).tolist()
s2_names = merged_pairs['s23_name'].fillna('').astype(str).tolist()
s1_addrs = merged_pairs['s1_addr'].fillna('').astype(str).tolist()
s2_addrs = merged_pairs['s23_addr'].fillna('').astype(str).tolist()

name_lev = [distance.Levenshtein.normalized_similarity(a, b) for a, b in zip(s1_names, s2_names)]
name_jw = [distance.JaroWinkler.similarity(a, b) for a, b in zip(s1_names, s2_names)]
name_sort = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_names, s2_names)]
name_set = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_names, s2_names)]
name_partial = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_names, s2_names)]
name_exact = [1.0 if a == b and a else 0.0 for a, b in zip(s1_names, s2_names)]
first_token_match = [1.0 if a and b and a.split()[0] == b.split()[0] else 0.0 for a, b in zip(s1_names, s2_names)]
name_len_diff = [abs(len(a) - len(b)) / max(1, max(len(a), len(b))) for a, b in zip(s1_names, s2_names)]

addr_lev = [distance.Levenshtein.normalized_similarity(a, b) if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
addr_jw = [distance.JaroWinkler.similarity(a, b) if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
addr_sort = [fuzz.token_sort_ratio(a, b) / 100.0 if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
addr_set = [fuzz.token_set_ratio(a, b) / 100.0 if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
addr_partial = [fuzz.partial_ratio(a, b) / 100.0 if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
addr_exact = [1.0 if a == b and a else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
addr_len_diff = [abs(len(a) - len(b)) / max(1, max(len(a), len(b))) for a, b in zip(s1_addrs, s2_addrs)]

p1 = merged_pairs['s1_postal'].fillna('').astype(str)
p2 = merged_pairs['s23_postal'].fillna('').astype(str)
postal_exact = ((p1 != '') & (p2 != '') & (p1 == p2)).astype(float).values
postal_prefix = ((p1 != '') & (p2 != '') & (p1.str[:3] == p2.str[:3])).astype(float).values
postal_conflict = ((p1 != '') & (p2 != '') & (p1.str[:3] != p2.str[:3])).astype(float).values

b1 = merged_pairs['s1_bldg'].fillna('').astype(str)
b2 = merged_pairs['s23_bldg'].fillna('').astype(str)
bldg_match = ((b1 != '') & (b2 != '') & (b1 == b2)).astype(float).values
bldg_conflict = ((b1 != '') & (b2 != '') & (b1 != b2)).astype(float).values

suf1 = merged_pairs['s1_suffix'].fillna('NONE').astype(str)
suf2 = merged_pairs['s23_suffix'].fillna('NONE').astype(str)
suffix_match = ((suf1 != 'NONE') & (suf1 == suf2)).astype(float).values
suffix_conflict = ((suf1 != 'NONE') & (suf2 != 'NONE') & (suf1 != suf2)).astype(float).values

# Word Jaccard similarities
def jaccard(a: str, b: str) -> float:
    wa = set(a.split())
    wb = set(b.split())
    u = wa | wb
    return len(wa & wb) / len(u) if u else 0.0

name_jaccard = [jaccard(a, b) for a, b in zip(s1_names, s2_names)]
addr_jaccard = [jaccard(a, b) if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
geom_sim = [np.sqrt(njw * ajw) for njw, ajw in zip(name_jw, addr_jw)]

X_val = pd.DataFrame({
    'name_lev': name_lev, 'name_jw': name_jw, 'name_sort': name_sort, 'name_set': name_set,
    'name_partial': name_partial, 'name_exact': name_exact, 'first_token_match': first_token_match,
    'name_len_diff': name_len_diff, 'name_jaccard': name_jaccard,
    'addr_lev': addr_lev, 'addr_jw': addr_jw, 'addr_sort': addr_sort, 'addr_set': addr_set,
    'addr_partial': addr_partial, 'addr_exact': addr_exact, 'addr_len_diff': addr_len_diff,
    'addr_jaccard': addr_jaccard, 'geom_sim': geom_sim,
    'postal_exact': postal_exact, 'postal_prefix': postal_prefix, 'postal_conflict': postal_conflict,
    'bldg_match': bldg_match, 'bldg_conflict': bldg_conflict,
    'suffix_match': suffix_match, 'suffix_conflict': suffix_conflict,
    'tfidf_name_sim': merged_pairs['tfidf_name_sim'].values,
    'tfidf_addr_sim': merged_pairs['tfidf_addr_sim'].values,
})

y_val = np.array([1 if c in gt_mapping.get(s1, set()) else 0 for s1, c in zip(merged_pairs['source1_entity_id'], merged_pairs['cand_entity_id'])])
print(f"Extracted {len(X_val.columns)} features for {len(X_val):,} pairs in {time.time()-t_fe:.2f}s")
print(f"Positives: {np.sum(y_val):,} ({np.mean(y_val)*100:.2f}%), Negatives: {len(y_val)-np.sum(y_val):,}")

# 5. Model Training & 1-to-1 Calibration
print("\n" + "="*60)
print("PHASE 4: 5-Fold GroupKFold LightGBM & 1-to-1 Optimization")
print("="*60)
gkf = GroupKFold(n_splits=5)
groups = merged_pairs['source1_entity_id'].values
oof_probs = np.zeros(len(X_val))
models = []

lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'learning_rate': 0.04, 'num_leaves': 45, 'max_depth': 7, 'min_child_samples': 30,
    'subsample': 0.85, 'colsample_bytree': 0.85,
    'scale_pos_weight': 2.5,
    'verbose': -1, 'random_state': 42
}

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_val, y_val, groups=groups)):
    X_trn_f, y_trn_f = X_val.iloc[trn_idx], y_val[trn_idx]
    X_val_f, y_val_f = X_val.iloc[val_idx], y_val[val_idx]
    clf = lgb.LGBMClassifier(**lgb_params, n_estimators=450)
    clf.fit(X_trn_f, y_trn_f, eval_set=[(X_val_f, y_val_f)], callbacks=[lgb.early_stopping(30, verbose=False)])
    oof_probs[val_idx] = clf.predict_proba(X_val_f)[:, 1]
    models.append(clf)
    print(f"  Fold {fold+1} trained (best iter: {clf.best_iteration_})")

# Save trained models for test inference
joblib.dump(models, 'experiments/models/lgbm_5fold_models.pkl')

def compute_macro_f05(gt_mapping: dict[str, set], predictions: dict[str, set]) -> tuple[float, float, float]:
    scores, precs, recs = [], [], []
    for s1_id, true_set in gt_mapping.items():
        pred_set = predictions.get(s1_id, set())
        if not true_set:
            if not pred_set:
                scores.append(1.0); precs.append(1.0); recs.append(1.0)
            else:
                scores.append(0.0); precs.append(0.0); recs.append(1.0)
            continue
        if not pred_set:
            scores.append(0.0); precs.append(0.0); recs.append(0.0)
            continue
        tp = len(true_set & pred_set)
        if tp == 0:
            scores.append(0.0); precs.append(0.0); recs.append(0.0)
        else:
            p = tp / len(pred_set)
            r = tp / len(true_set)
            denom = 0.25 * p + r
            scores.append((1.25 * p * r) / denom if denom > 0 else 0.0)
            precs.append(p)
            recs.append(r)
    return float(np.mean(scores)), float(np.mean(precs)), float(np.mean(recs))

print("\n--- Calibration with 1-to-1 Argmax Matching across 50,000 entities ---")
pairs_df = merged_pairs[['source1_entity_id', 'cand_entity_id']].copy()
pairs_df['prob'] = oof_probs

best_t, best_f05, best_p, best_r = 0.70, 0.0, 0.0, 0.0
for t in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90]:
    filtered = pairs_df[pairs_df['prob'] >= t].sort_values('prob', ascending=False)
    deduped = filtered.drop_duplicates('cand_entity_id')
    preds = {s1: set() for s1 in gt_mapping.keys()}
    for s1, c in zip(deduped['source1_entity_id'], deduped['cand_entity_id']):
        preds[s1].add(c)
    f05, p, r = compute_macro_f05(gt_mapping, preds)
    print(f"Threshold {t:.2f} + 1-to-1 -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")
    if f05 > best_f05:
        best_f05, best_t, best_p, best_r = f05, t, p, r

print("\n" + "="*60)
print(f"FINAL {VAL_SAMPLE_SIZE:,}-RECORD HOLDOUT RESULTS")
print("="*60)
print(f"Optimal Threshold: {best_t:.2f}")
print(f"Macro F0.5 Score:  {best_f05:.4f} ({best_f05*100:.2f}%)")
print(f"Macro Precision:   {best_p:.4f} ({best_p*100:.2f}%)")
print(f"Macro Recall:      {best_r:.4f} ({best_r*100:.2f}%)")
print(f"Blocking Recall:   {blocking_recall:.4f} ({blocking_recall*100:.2f}%)")
print(f"Gate Target (>= 98%): {'PASSED' if best_f05 >= 0.98 else 'FAILED'}")
print(f"Total wall-clock time: {(time.time()-t_run_start)/60:.1f} min")
print("="*60)
