import os, sys, time, re, unicodedata, gc, warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz, distance
import joblib

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore', category=UserWarning)

# Configurable paths via environment variables
DATA_DIR = os.environ.get('DATA_DIR', 'dataset')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
MODEL_PATH = os.environ.get('MODEL_PATH', 'experiments/models/ensemble_models.pkl')
if not os.path.exists(MODEL_PATH) and os.path.exists('experiments/models/lgbm_5fold_models.pkl'):
    MODEL_PATH = 'experiments/models/lgbm_5fold_models.pkl'
CKPT_DIR = os.environ.get('CKPT_DIR', 'experiments/checkpoints')
ENV_THRESHOLD = os.environ.get('THRESHOLD', None)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# 1. Normalization Rules (US, India, France)
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
    r'\b(societe\s+anonyme)\b': ' SA ',
    r'\b(sci|societe\s+civile\s+immobiliere)\b': ' SCI ',
    r'\b(eurl)\b': ' EURL ',
    r'\b(gie)\b': ' GIE ',
    r'\b(snc)\b': ' SNC ',
}

ABBREV_EXPANSIONS = {
    r'\bintl\.?\b': 'international',
    r'\bmfg\.?\b': 'manufacturing',
    r'\bsvcs?\.?\b': 'services',
    r'\btech\.?\b': 'technologies',
    r'\bdept\.?\b': 'department',
    r'\bassoc\.?\b': 'associates',
    r'\bste\.?\b': 'societe',
    r'\bcie\.?\b': 'compagnie',
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
    r'\ball[eé]e?\b': 'allee',
    r'\bimp\b': 'impasse',
    r'\brue\b': 'rue',
    r'\bchemin\b': 'chemin',
    r'\bcours\b': 'cours',
    r'\bquai\b': 'quai',
    r'\bsq\.?\b|\bsquare\b': 'square',
}

RE_POSTAL = re.compile(r'\b[1-9]\d{2}\s?\d{3}\b|\b\d{5}(?:-\d{4})?\b')
RE_URL = re.compile(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
RE_LANDMARKS = re.compile(r'\b(near|opp|opposite|behind|beside|adj|adjacent to|next to|pres de|en face de)\s+[\w\s]+?(?=,|\\.|$)', re.IGNORECASE)
RE_DIGITS = re.compile(r'\b\d+\b')
N_THREADS = min(8, os.cpu_count() or 4)

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

def jaccard(a: str, b: str) -> float:
    wa = set(a.split())
    wb = set(b.split())
    u = wa | wb
    return len(wa & wb) / len(u) if u else 0.0

def compute_pairwise_features(merged_pairs_df: pd.DataFrame) -> pd.DataFrame:
    s1_names = merged_pairs_df['s1_name'].fillna('').astype(str).tolist()
    s2_names = merged_pairs_df['s23_name'].fillna('').astype(str).tolist()
    s1_addrs = merged_pairs_df['s1_addr'].fillna('').astype(str).tolist()
    s2_addrs = merged_pairs_df['s23_addr'].fillna('').astype(str).tolist()
    cand_ids = merged_pairs_df['cand_entity_id'].fillna('').astype(str).tolist()

    name_lev = [distance.Levenshtein.normalized_similarity(a, b) for a, b in zip(s1_names, s2_names)]
    name_jw = [distance.JaroWinkler.similarity(a, b) for a, b in zip(s1_names, s2_names)]
    name_sort = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_names, s2_names)]
    name_set = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_names, s2_names)]
    name_partial = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_names, s2_names)]
    name_exact = [1.0 if a == b and a else 0.0 for a, b in zip(s1_names, s2_names)]
    first_token_match = [1.0 if a and b and a.split()[0] == b.split()[0] else 0.0 for a, b in zip(s1_names, s2_names)]
    name_len_diff = [abs(len(a) - len(b)) / max(1, max(len(a), len(b))) for a, b in zip(s1_names, s2_names)]
    name_jaccard = [jaccard(a, b) for a, b in zip(s1_names, s2_names)]

    addr_lev = [distance.Levenshtein.normalized_similarity(a, b) if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
    addr_jw = [distance.JaroWinkler.similarity(a, b) if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
    addr_sort = [fuzz.token_sort_ratio(a, b) / 100.0 if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
    addr_set = [fuzz.token_set_ratio(a, b) / 100.0 if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
    addr_partial = [fuzz.partial_ratio(a, b) / 100.0 if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
    addr_exact = [1.0 if a == b and a else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
    addr_len_diff = [abs(len(a) - len(b)) / max(1, max(len(a), len(b))) for a, b in zip(s1_addrs, s2_addrs)]
    addr_jaccard = [jaccard(a, b) if a and b else 0.0 for a, b in zip(s1_addrs, s2_addrs)]
    geom_sim = [np.sqrt(njw * ajw) for njw, ajw in zip(name_jw, addr_jw)]

    p1 = merged_pairs_df['s1_postal'].fillna('').astype(str)
    p2 = merged_pairs_df['s23_postal'].fillna('').astype(str)
    postal_exact = ((p1 != '') & (p2 != '') & (p1 == p2)).astype(float).values
    postal_prefix = ((p1 != '') & (p2 != '') & (p1.str[:3] == p2.str[:3])).astype(float).values
    postal_conflict = ((p1 != '') & (p2 != '') & (p1.str[:3] != p2.str[:3])).astype(float).values

    b1 = merged_pairs_df['s1_bldg'].fillna('').astype(str)
    b2 = merged_pairs_df['s23_bldg'].fillna('').astype(str)
    bldg_match = ((b1 != '') & (b2 != '') & (b1 == b2)).astype(float).values
    bldg_conflict = ((b1 != '') & (b2 != '') & (b1 != b2)).astype(float).values

    suf1 = merged_pairs_df['s1_suffix'].fillna('NONE').astype(str)
    suf2 = merged_pairs_df['s23_suffix'].fillna('NONE').astype(str)
    suffix_match = ((suf1 != 'NONE') & (suf1 == suf2)).astype(float).values
    suffix_conflict = ((suf1 != 'NONE') & (suf2 != 'NONE') & (suf1 != suf2)).astype(float).values

    # Source discriminator (S2 vs S3)
    is_s2 = [1.0 if c.startswith('S2-') else 0.0 for c in cand_ids]

    # Numerical sequence overlap and branch conflict
    num_overlap, num_conflict, cross_overlap = [], [], []
    for n1, a1, n2, a2 in zip(s1_names, s1_addrs, s2_names, s2_addrs):
        nums1 = set(RE_DIGITS.findall(n1 + ' ' + a1))
        nums2 = set(RE_DIGITS.findall(n2 + ' ' + a2))
        if nums1 and nums2:
            inter = len(nums1 & nums2)
            num_overlap.append(inter / len(nums1 | nums2))
            num_conflict.append(1.0 if inter == 0 else 0.0)
        else:
            num_overlap.append(0.0)
            num_conflict.append(0.0)

        # Cross-attribute overlap
        w_n1, w_a1 = set(n1.split()), set(a1.split())
        w_n2, w_a2 = set(n2.split()), set(a2.split())
        raw_cross = len(w_n1 & w_a2) + len(w_n2 & w_a1)
        total_words = len(w_n1) + len(w_a2) + len(w_n2) + len(w_a1)
        cross_overlap.append(raw_cross / max(1, total_words))

    # Group-relative ranking features within each S1 cluster
    grp = merged_pairs_df.groupby('source1_entity_id')
    max_n_sim = grp['tfidf_name_sim'].transform('max').values
    max_a_sim = grp['tfidf_addr_sim'].transform('max').values
    cand_cnt = grp['cand_entity_id'].transform('count').values

    temp_jw = pd.Series(name_jw, index=merged_pairs_df.index)
    grp_jw = temp_jw.groupby(merged_pairs_df['source1_entity_id'])
    max_jw_val = grp_jw.transform('max').values
    cand_rank_jw = grp_jw.rank(ascending=False, method='min').values - 1.0

    sim_diff_name = merged_pairs_df['tfidf_name_sim'].values - max_n_sim
    sim_diff_addr = merged_pairs_df['tfidf_addr_sim'].values - max_a_sim
    jw_diff_name = np.array(name_jw) - max_jw_val

    return pd.DataFrame({
        'name_lev': name_lev, 'name_jw': name_jw, 'name_sort': name_sort, 'name_set': name_set,
        'name_partial': name_partial, 'name_exact': name_exact, 'first_token_match': first_token_match,
        'name_len_diff': name_len_diff, 'name_jaccard': name_jaccard,
        'addr_lev': addr_lev, 'addr_jw': addr_jw, 'addr_sort': addr_sort, 'addr_set': addr_set,
        'addr_partial': addr_partial, 'addr_exact': addr_exact, 'addr_len_diff': addr_len_diff,
        'addr_jaccard': addr_jaccard, 'geom_sim': geom_sim,
        'postal_exact': postal_exact, 'postal_prefix': postal_prefix, 'postal_conflict': postal_conflict,
        'bldg_match': bldg_match, 'bldg_conflict': bldg_conflict,
        'suffix_match': suffix_match, 'suffix_conflict': suffix_conflict,
        'tfidf_name_sim': merged_pairs_df['tfidf_name_sim'].values,
        'tfidf_addr_sim': merged_pairs_df['tfidf_addr_sim'].values,
        'is_s2': is_s2,
        'num_overlap': num_overlap,
        'num_conflict': num_conflict,
        'cross_overlap': cross_overlap,
        'sim_diff_name': sim_diff_name,
        'sim_diff_addr': sim_diff_addr,
        'jw_diff_name': jw_diff_name,
        'cand_rank_jw': cand_rank_jw,
        'cand_cnt': cand_cnt,
    })

def main():
    print("="*60)
    print("STARTING FULL TEST SET INFERENCE PIPELINE (V3 ENSEMBLE)")
    print("="*60)
    
    t_start = time.time()
    
    # Load Models (Support Ensemble dictionary artifact or legacy list)
    model_obj = joblib.load(MODEL_PATH)
    if isinstance(model_obj, dict):
        lgb_models = model_obj.get('lgb_models', [])
        cb_models = model_obj.get('cb_models', [])
        has_cb = model_obj.get('has_catboost', False) and len(cb_models) > 0
        calibrated_t = model_obj.get('optimal_threshold', 0.50)
        print(f"Loaded ensemble artifact from {MODEL_PATH}: {len(lgb_models)} LightGBM, {len(cb_models)} CatBoost models.")
    else:
        lgb_models = model_obj
        cb_models = []
        has_cb = False
        calibrated_t = 0.55
        print(f"Loaded legacy model list from {MODEL_PATH}: {len(lgb_models)} models.")
        
    threshold = float(ENV_THRESHOLD) if ENV_THRESHOLD is not None else calibrated_t
    print(f"Active Decision Threshold: {threshold:.2f} | 1-to-1 Argmax Matching")

    # Read S1 test entities to guarantee 100% presence in output
    print("Reading reference test entities (test_source1.tsv)...")
    df_s1_test = pd.read_csv(f'{DATA_DIR}/test/test_source1.tsv', sep='\t')
    all_s1_ids = df_s1_test['entity_id'].tolist()
    print(f"Total required test S1 entities: {len(all_s1_ids):,}")

    ckpt_path = f'{CKPT_DIR}/test_inference_progress_v3.pkl'
    done_countries = set()
    if os.path.exists(ckpt_path):
        ckpt = joblib.load(ckpt_path)
        final_matches, final_candidates, done_countries = ckpt['final_matches'], ckpt['final_candidates'], ckpt['done_countries']
        print(f"Resuming from checkpoint: {len(done_countries)} countries already completed ({sorted(done_countries)})")
    else:
        final_matches = {s1_id: [] for s1_id in all_s1_ids}
        final_candidates = {s1_id: [] for s1_id in all_s1_ids}

    countries = df_s1_test['country'].unique()
    print(f"Test countries discovered: {list(countries)}")

    for country in countries:
        if country in done_countries:
            print(f"\n>>> Skipping Country [{country}] (already completed in checkpoint) <<<")
            continue
        print(f"\n>>> Processing Country [{country}] <<<")
        s1_c = df_s1_test[df_s1_test['country'] == country].copy().reset_index(drop=True)
        print(f"  Reference S1 entities in {country}: {len(s1_c):,}")

        # Load S2 and S3 for this country
        print(f"  Loading test candidates for {country}...")
        s2_chunks = []
        for c in pd.read_csv(f'{DATA_DIR}/test/test_source2.tsv', sep='\t', chunksize=500000):
            sub = c[c['country'] == country]
            if len(sub): s2_chunks.append(sub)
        s2_c = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()

        s3_chunks = []
        for c in pd.read_csv(f'{DATA_DIR}/test/test_source3.tsv', sep='\t', chunksize=500000):
            sub = c[c['country'] == country]
            if len(sub): s3_chunks.append(sub)
        s3_c = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()

        s23_c = pd.concat([s2_c, s3_c], ignore_index=True).drop_duplicates('entity_id').reset_index(drop=True)
        print(f"  Total candidate records in {country}: {len(s23_c):,}")

        del s2_chunks, s3_chunks, s2_c, s3_c
        gc.collect()

        # Preprocess
        s1_c = preprocess_dataframe(s1_c)
        s23_c = preprocess_dataframe(s23_c)

        # Build candidate vectorizers with expanded 150k vocabulary
        print("  Building sparse vectorizers (150,000 max features)...")
        t_vec = time.time()
        vec_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=150000, sublinear_tf=True, dtype=np.float32)
        X_cand_name = vec_name.fit_transform(s23_c['clean_name'])

        vec_addr = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=150000, sublinear_tf=True, dtype=np.float32)
        X_cand_addr = vec_addr.fit_transform(s23_c['clean_addr'])
        print(f"  Vectorizers ready in {time.time()-t_vec:.2f}s")

        # Build secondary index on (postal_code, prefix)
        s23_post = s23_c['postal_code'].values
        s23_name_arr = s23_c['clean_name'].values
        s23_idx_dict = {}
        for idx, (p, name) in enumerate(zip(s23_post, s23_name_arr)):
            if p and len(name) >= 3:
                k = (p, name[:4])
                lst = s23_idx_dict.get(k)
                if lst is None:
                    s23_idx_dict[k] = [idx]
                elif len(lst) < 15:
                    lst.append(idx)

        # Process S1 in batches
        BATCH_SIZE = 30000
        n_batches = (len(s1_c) + BATCH_SIZE - 1) // BATCH_SIZE
        s1_cols = {'clean_name': 's1_name', 'clean_addr': 's1_addr', 'postal_code': 's1_postal', 'building_num': 's1_bldg', 'legal_suffix': 's1_suffix'}
        s23_cols = {'clean_name': 's23_name', 'clean_addr': 's23_addr', 'postal_code': 's23_postal', 'building_num': 's23_bldg', 'legal_suffix': 's23_suffix'}
        s23_sub_cols = s23_c[['entity_id', *s23_cols.keys()]].rename(columns=s23_cols)

        country_scored_pairs = []

        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(len(s1_c), b_start + BATCH_SIZE)
            batch_s1 = s1_c.iloc[b_start:b_end].copy().reset_index(drop=True)
            print(f"  [{country}] Batch {b+1}/{n_batches} ({len(batch_s1):,} S1 entities)...")

            # Blocking Pass 1 & 2
            X_s1_name = vec_name.transform(batch_s1['clean_name'])
            res_name = sp_matmul_topn(X_s1_name, X_cand_name.T, top_n=40, threshold=0.22, n_threads=N_THREADS)

            X_s1_addr = vec_addr.transform(batch_s1['clean_addr'])
            res_addr = sp_matmul_topn(X_s1_addr, X_cand_addr.T, top_n=25, threshold=0.28, n_threads=N_THREADS)

            coo_n = res_name.tocoo()
            coo_a = res_addr.tocoo()
            df_n = pd.DataFrame({'s1_idx': coo_n.row, 's23_idx': coo_n.col, 'tfidf_name_sim': coo_n.data, 'tfidf_addr_sim': 0.0})
            df_a = pd.DataFrame({'s1_idx': coo_a.row, 's23_idx': coo_a.col, 'tfidf_name_sim': 0.0, 'tfidf_addr_sim': coo_a.data})

            b_dfs = [df_n, df_a]

            # Blocking Pass 3: Postal + Name Prefix
            extra_s1_b, extra_s23_b = [], []
            for s1_i, (p, name) in enumerate(zip(batch_s1['postal_code'].values, batch_s1['clean_name'].values)):
                if p and len(name) >= 3:
                    k = (p, name[:4])
                    m_indices = s23_idx_dict.get(k)
                    if m_indices:
                        for m_idx in m_indices:
                            extra_s1_b.append(s1_i)
                            extra_s23_b.append(m_idx)
            if extra_s1_b:
                b_dfs.append(pd.DataFrame({'s1_idx': extra_s1_b, 's23_idx': extra_s23_b, 'tfidf_name_sim': 0.20, 'tfidf_addr_sim': 0.20}))

            b_pairs = pd.concat(b_dfs, ignore_index=True)
            b_pairs = b_pairs.groupby(['s1_idx', 's23_idx'], as_index=False)[['tfidf_name_sim', 'tfidf_addr_sim']].max()
            b_pairs['max_sim'] = np.maximum(b_pairs['tfidf_name_sim'], b_pairs['tfidf_addr_sim'])
            b_pairs = b_pairs.sort_values(['s1_idx', 'max_sim'], ascending=[True, False]).groupby('s1_idx').head(60).drop(columns=['max_sim']).reset_index(drop=True)

            b_pairs['source1_entity_id'] = batch_s1.iloc[b_pairs['s1_idx'].values]['entity_id'].values
            b_pairs['cand_entity_id'] = s23_c.iloc[b_pairs['s23_idx'].values]['entity_id'].values

            # Record candidate pairs
            for s1_id, c_id in zip(b_pairs['source1_entity_id'], b_pairs['cand_entity_id']):
                final_candidates[s1_id].append(c_id)

            # Feature extraction
            merged = b_pairs.merge(batch_s1[['entity_id', *s1_cols.keys()]].rename(columns=s1_cols), left_on='source1_entity_id', right_on='entity_id').drop(columns=['entity_id'])
            merged = merged.merge(s23_sub_cols, left_on='cand_entity_id', right_on='entity_id').drop(columns=['entity_id'])

            X_b = compute_pairwise_features(merged)
            
            # Predict ensemble probability
            lgb_p = np.mean([clf.predict_proba(X_b)[:, 1] for clf in lgb_models], axis=0)
            if has_cb and cb_models:
                cb_p = np.mean([clf.predict_proba(X_b)[:, 1] for clf in cb_models], axis=0)
                probs = 0.55 * lgb_p + 0.45 * cb_p
            else:
                probs = lgb_p
            
            scored_df = b_pairs[['source1_entity_id', 'cand_entity_id']].copy()
            scored_df['prob'] = probs
            country_scored_pairs.append(scored_df[scored_df['prob'] >= threshold])

        # Apply 1-to-1 argmax assignment across all country predictions
        if country_scored_pairs:
            all_c_scored = pd.concat(country_scored_pairs, ignore_index=True).sort_values('prob', ascending=False)
            deduped = all_c_scored.drop_duplicates('cand_entity_id')
            for s1_id, c_id in zip(deduped['source1_entity_id'], deduped['cand_entity_id']):
                final_matches[s1_id].append(c_id)
            print(f"  Matches assigned for {country}: {len(deduped):,}")

        done_countries.add(country)
        joblib.dump({'final_matches': final_matches, 'final_candidates': final_candidates, 'done_countries': done_countries}, ckpt_path)
        print(f"  Checkpoint saved ({len(done_countries)}/{len(countries)} countries done)")

        del s1_c, s23_c, X_cand_name, X_cand_addr, vec_name, vec_addr, country_scored_pairs
        gc.collect()

    # 4. Write Final Official TSVs
    print("\nWriting output files...")
    df_matching = pd.DataFrame({
        'source1_entity_id': all_s1_ids,
        'matched_entity_ids': [','.join(final_matches[s1_id]) for s1_id in all_s1_ids]
    })
    df_matching.to_csv(f'{OUTPUT_DIR}/matching_results.tsv', sep='\t', index=False)

    df_candidates = pd.DataFrame({
        'source1_entity_id': all_s1_ids,
        'candidate_entity_ids': [','.join(dict.fromkeys(final_candidates[s1_id])) for s1_id in all_s1_ids]
    })
    df_candidates.to_csv(f'{OUTPUT_DIR}/candidate_pairs.tsv', sep='\t', index=False)

    non_empty_matches = sum(1 for m in df_matching['matched_entity_ids'] if m)
    total_links = sum(len(m.split(',')) for m in df_matching['matched_entity_ids'] if m)
    print(f"Output files successfully written in {time.time()-t_start:.2f}s:")
    print(f"  - {OUTPUT_DIR}/matching_results.tsv: {len(df_matching):,} rows ({non_empty_matches:,} linked, {total_links:,} total links)")
    print(f"  - {OUTPUT_DIR}/candidate_pairs.tsv: {len(df_candidates):,} rows")

    # 5. Run Official Validator
    print("\nValidating submission against official checker...")
    import subprocess
    cmd = [
        sys.executable, 'student_resource/utils/validate_submission.py',
        '--matching', f'{OUTPUT_DIR}/matching_results.tsv',
        '--candidate', f'{OUTPUT_DIR}/candidate_pairs.tsv',
        '--test-dir', f'{DATA_DIR}/test',
        '--check-id'
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator stderr:", res.stderr)

    # 6. Package Submission Zip
    print("\nPackaging final submission zip...")
    pkg_cmd = [sys.executable, 'utils/create_submission_zip.py']
    subprocess.run(pkg_cmd)

if __name__ == '__main__':
    main()
