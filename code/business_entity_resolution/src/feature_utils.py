import re
import numpy as np
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.fuzz as fuzz

WORD_RE = re.compile(r'\b\w+\b')
DIGIT_RE = re.compile(r'\b\d+\b')

GEO_STOPS = {
    'mumbai', 'delhi', 'kolkata', 'bangalore', 'bengaluru', 'hyderabad', 'chennai',
    'pune', 'jaipur', 'ahmedabad', 'surat', 'indore', 'bhopal', 'lucknow', 'patna',
    'kanpur', 'nagpur', 'thane', 'charlotte', 'indianapolis', 'dallas', 'houston',
    'austin', 'seattle', 'portland', 'denver', 'phoenix', 'atlanta', 'boston',
    'chicago', 'miami', 'orlando', 'detroit', 'minneapolis', 'cleveland', 'columbus',
    'india', 'usa', 'america', 'american', 'california', 'texas', 'florida', 'york',
    'washington', 'ohio', 'illinois', 'georgia', 'north', 'south', 'east', 'west'
}

CORP_STOPS = {
    'the', 'a', 'an', 'and', '&', 'of', 'in', 'on', 'at', 'for', 'to',
    'llc', 'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'pvt', 'private', 'group', 'services', 'enterprises',
    'holdings', 'technologies', 'solutions', 'consulting', 'management',
    'international', 'products', 'trading', 'industries', 'associates'
}

CORE_NAME_STOPS = CORP_STOPS | GEO_STOPS

def extract_tokens(text):
    if not text:
        return set()
    return set(WORD_RE.findall(text))

def extract_numbers(text):
    if not text:
        return set()
    return set(DIGIT_RE.findall(text))

def jaccard_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0

def _acronym(text):
    toks = text.split()
    if len(toks) < 2:
        return ''
    return ''.join(t[0] for t in toks if t)

def compute_pair_features(s1_name, s1_addr, s1_pc, s2_name, s2_addr, s2_pc,
                          embed_score=0.0,
                          s1_has_state=False, s2_has_state=False,
                          s1_needs_trans=False, s2_needs_trans=False):
    """
    Computes 27 fine-grained lexical, Jaro-Winkler, phonetic, and topological features for a pair.
    Inputs are cleaned strings.
    """
    s1_n = s1_name or ''
    s2_n = s2_name or ''
    s1_a = s1_addr or ''
    s2_a = s2_addr or ''
    
    # 1-4. Name Fuzzy Ratios
    name_ratio = fuzz.ratio(s1_n, s2_n) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_n, s2_n) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_n, s2_n) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_n, s2_n) / 100.0
    
    # 5-6. Jaro-Winkler Similarities (Name & Address)
    name_jw = float(jw.similarity(s1_n, s2_n))
    addr_jw = float(jw.similarity(s1_a, s2_a))
    
    # 7-10. Address Fuzzy Ratios
    addr_ratio = fuzz.ratio(s1_a, s2_a) / 100.0
    addr_partial_ratio = fuzz.partial_ratio(s1_a, s2_a) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_a, s2_a) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_a, s2_a) / 100.0
    
    # 11-12. Jaccard Token Overlaps
    s1_n_toks = extract_tokens(s1_n)
    s2_n_toks = extract_tokens(s2_n)
    name_jaccard = jaccard_similarity(s1_n_toks, s2_n_toks)
    
    s1_a_toks = extract_tokens(s1_a)
    s2_a_toks = extract_tokens(s2_a)
    addr_jaccard = jaccard_similarity(s1_a_toks, s2_a_toks)
    
    # 13. Postal Code Match (1.0=match, 0.0=mismatch, 0.5=missing)
    pc1 = s1_pc or ''
    pc2 = s2_pc or ''
    if pc1 and pc2:
        postal_match = 1.0 if pc1 == pc2 else 0.0
    else:
        postal_match = 0.5
        
    # 14. Street / Building Number Overlap
    s1_nums = extract_numbers(s1_a)
    s2_nums = extract_numbers(s2_a)
    if s1_nums and s2_nums:
        num_overlap = 1.0 if (s1_nums & s2_nums) else 0.0
    elif not s1_nums and not s2_nums:
        num_overlap = 0.5
    else:
        num_overlap = 0.0
        
    # 15-16. Length differences
    len_diff_name = abs(len(s1_n) - len(s2_n)) / max(len(s1_n), len(s2_n), 1)
    len_diff_addr = abs(len(s1_a) - len(s2_a)) / max(len(s1_a), len(s2_a), 1)
    
    # 17-21. Stage 2 Blocking Signals (adapted for v2 single embed_score)
    sig_both = 1.0 if embed_score >= 0.70 else 0.0
    sig_name = 0.0
    name_sim = float(embed_score)
    addr_sim = float(embed_score)
    score_gap = 0.0
    
    # 22. postal_and_num_match interaction
    postal_and_num_match = 1.0 if (postal_match == 1.0 and num_overlap == 1.0) else 0.0
    
    # 23. name_containment asymmetry
    if s1_n_toks and s2_n_toks:
        smaller, larger = (s1_n_toks, s2_n_toks) if len(s1_n_toks) <= len(s2_n_toks) else (s2_n_toks, s1_n_toks)
        name_containment = len(smaller & larger) / len(smaller)
    else:
        name_containment = 0.0
        
    # 24. name_acronym_match
    s1_acronym = _acronym(s1_n.lower())
    s2_acronym = _acronym(s2_n.lower())
    name_acronym_match = 0.0
    if s1_acronym and s2_n.lower().replace(' ', '') == s1_acronym:
        name_acronym_match = 1.0
    elif s2_acronym and s1_n.lower().replace(' ', '') == s2_acronym:
        name_acronym_match = 1.0
        
    # 25. Both sides have a detected state/region in their address
    both_has_state = 1.0 if (s1_has_state and s2_has_state) else 0.0

    # 26. State-presence mismatch (one side has it, other doesn't) — signals
    # incomplete address data on one side rather than a true mismatch
    state_presence_mismatch = 1.0 if (bool(s1_has_state) != bool(s2_has_state)) else 0.0

    # 27. Either side required transliteration — proxy for non-Latin-script
    # source data, which correlates with noisier address/name normalization
    either_needs_translit = 1.0 if (s1_needs_trans or s2_needs_trans) else 0.0

    # 28. Address number conflict (both specify numbers but have 0 overlap)
    num_conflict = 1.0 if (s1_nums and s2_nums and not (s1_nums & s2_nums)) else 0.0

    # 29. Candidate address is completely empty
    cand_addr_empty = 1.0 if not s2_a else 0.0

    # 30-31. Core name token overlap (excluding corporate and geo stopwords)
    s1_core = {t.lower() for t in s1_n_toks if t.lower() not in CORE_NAME_STOPS and len(t) > 2}
    s2_core = {t.lower() for t in s2_n_toks if t.lower() not in CORE_NAME_STOPS and len(t) > 2}
    core_name_jaccard = jaccard_similarity(s1_core, s2_core)
    core_name_exact = 1.0 if (s1_core and s1_core == s2_core) else 0.0

    # 32. Space-collapsed exact name match
    name_nospace_match = 1.0 if (s1_n and s1_n.replace(' ', '') == s2_n.replace(' ', '')) else 0.0

    # 33. Harmonic mean of Name JW and Address Token Set Ratio (strictly penalizes single-attribute false positives)
    name_addr_harmonic = (2.0 * name_jw * addr_token_set) / (name_jw + addr_token_set + 1e-6)

    return [
        name_ratio,
        name_partial_ratio,
        name_token_sort,
        name_token_set,
        name_jw,
        addr_jw,
        addr_ratio,
        addr_partial_ratio,
        addr_token_sort,
        addr_token_set,
        name_jaccard,
        addr_jaccard,
        postal_match,
        num_overlap,
        len_diff_name,
        len_diff_addr,
        sig_both,
        sig_name,
        name_sim,
        addr_sim,
        score_gap,
        postal_and_num_match,
        name_containment,
        name_acronym_match,
        both_has_state,
        state_presence_mismatch,
        either_needs_translit,
        num_conflict,
        cand_addr_empty,
        core_name_jaccard,
        core_name_exact,
        name_nospace_match,
        name_addr_harmonic
    ]

FEATURE_NAMES = [
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort',
    'name_token_set',
    'name_jw',
    'addr_jw',
    'addr_ratio',
    'addr_partial_ratio',
    'addr_token_sort',
    'addr_token_set',
    'name_jaccard',
    'addr_jaccard',
    'postal_match',
    'num_overlap',
    'len_diff_name',
    'len_diff_addr',
    'sig_both',
    'sig_name',
    'name_sim',
    'addr_sim',
    'score_gap',
    'postal_and_num_match',
    'name_containment',
    'name_acronym_match',
    'both_has_state',
    'state_presence_mismatch',
    'either_needs_translit',
    'num_conflict',
    'cand_addr_empty',
    'core_name_jaccard',
    'core_name_exact',
    'name_nospace_match',
    'name_addr_harmonic'
]

def compute_embedding_features(emb1, emb2):
    """
    emb1, emb2: L2-normalized 384-dim float32 vectors.
    Returns 3 features.
    """
    cos_sim = float(np.dot(emb1, emb2))  # L2-normalized -> dot = cosine
    l2_dist = float(np.linalg.norm(emb1 - emb2))
    high_conf_semantic = 1.0 if cos_sim >= 0.75 else 0.0
    return [cos_sim, l2_dist, high_conf_semantic]

EMBEDDING_FEATURE_NAMES = ['embed_cosine', 'embed_l2_dist', 'embed_high_conf']

ALL_FEATURE_NAMES = FEATURE_NAMES + EMBEDDING_FEATURE_NAMES
# Total: 27 lexical + 3 embedding = 30 features

def extract_pair_features(s1_rec, cand_rec, embed_score=0.0):
    """
    Computes all 30 features for an (s1_rec, cand_rec) record pair.
    s1_rec, cand_rec: 7-tuples from load_full_record_map:
      (clean_name, clean_addr, postal, has_state, needs_trans_name, needs_trans_addr, embedding)
    """
    lexical = compute_pair_features(
        s1_rec[0], s1_rec[1], s1_rec[2],
        cand_rec[0], cand_rec[1], cand_rec[2],
        embed_score=float(embed_score),
        s1_has_state=s1_rec[3], s2_has_state=cand_rec[3],
        s1_needs_trans=s1_rec[4], s2_needs_trans=cand_rec[4]
    )
    emb_feats = compute_embedding_features(s1_rec[6], cand_rec[6])
    return lexical + emb_feats

