# Feature Engineering & String Similarity

## Overview

For each candidate pair (S1 record, S2/S3 record), extract ~30 features for classification.

---

## 1. String Similarity Metrics

### Core Metrics (via `rapidfuzz`)

```python
from rapidfuzz import fuzz, distance

def compute_string_metrics(s1: str, s2: str) -> dict:
    if not s1 or not s2:
        return {k: 0.0 for k in [
            "levenshtein", "jaro_winkler", "token_sort", 
            "token_set", "partial_ratio", "token_jaccard", "monge_elkan"
        ]}
    
    t1, t2 = set(s1.split()), set(s2.split())
    jaccard = len(t1 & t2) / max(1, len(t1 | t2))
    
    # Symmetrized Monge-Elkan (JW inner)
    def _me(a_tokens, b_tokens):
        if not a_tokens or not b_tokens:
            return 0.0
        return sum(
            max(distance.JaroWinkler.similarity(a, b) for b in b_tokens)
            for a in a_tokens
        ) / len(a_tokens)
    
    me = 0.5 * (_me(list(t1), list(t2)) + _me(list(t2), list(t1)))
    
    return {
        "levenshtein": distance.Levenshtein.normalized_similarity(s1, s2),
        "jaro_winkler": distance.JaroWinkler.similarity(s1, s2),
        "token_sort": fuzz.token_sort_ratio(s1, s2) / 100.0,
        "token_set": fuzz.token_set_ratio(s1, s2) / 100.0,
        "partial_ratio": fuzz.partial_ratio(s1, s2) / 100.0,
        "token_jaccard": jaccard,
        "monge_elkan": me
    }
```

### Metric Properties

| Metric | Captures | Robust To | Weakness |
|---|---|---|---|
| Levenshtein | Edit distance (typos, char swaps) | Single-char errors | Token reordering |
| Jaro-Winkler | Prefix similarity | Common prefix branding | Long suffixes |
| Token Sort Ratio | Sorted token comparison | Word order permutation | Substring matches |
| Token Set Ratio | Set intersection/union | Word order + extra tokens | Very loose |
| Jaccard (token) | Set overlap | Token permutation | Ignores typos within tokens |
| Monge-Elkan | Best token-to-token matching | Abbreviations, partial tokens | Expensive O(n²) |

### Soft TF-IDF

Combines TF-IDF weighting with fuzzy token matching:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SoftTFIDF:
    def __init__(self, corpus, threshold=0.82):
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(corpus)
        self.vocab = self.vectorizer.vocabulary_
        self.idf = self.vectorizer.idf_
        self.threshold = threshold

    def score(self, s1: str, s2: str) -> float:
        t1 = [w for w in s1.lower().split() if w in self.vocab]
        t2 = [w for w in s2.lower().split() if w in self.vocab]
        if not t1 or not t2:
            return 0.0
        
        w1 = {t: self.idf[self.vocab[t]] for t in t1}
        w2 = {t: self.idf[self.vocab[t]] for t in t2}
        n1 = np.sqrt(sum(v**2 for v in w1.values()))
        n2 = np.sqrt(sum(v**2 for v in w2.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        
        score = 0.0
        for tok1, wt1 in w1.items():
            best_sim, best_tok = 0.0, None
            for tok2 in w2:
                sim = distance.JaroWinkler.similarity(tok1, tok2)
                if sim > best_sim:
                    best_sim, best_tok = sim, tok2
            if best_sim >= self.threshold and best_tok:
                score += (wt1 / n1) * (w2[best_tok] / n2) * best_sim
        return min(1.0, score)
```

---

## 2. Business Name Normalization

### Multi-Jurisdiction Legal Suffix Extraction

```python
import re, unicodedata

SUFFIX_MAP = {
    # India & Commonwealth
    r"\b(pvt\.?\s*ltd\.?|private\s+limited)\b": "PVT_LTD",
    r"\b(ltd\.?|limited)\b": "LTD",
    r"\b(llp|limited\s+liability\s+partnership)\b": "LLP",
    # US
    r"\b(inc\.?|incorporated)\b": "INC",
    r"\b(corp\.?|corporation)\b": "CORP",
    r"\b(llc|l\.l\.c\.)\b": "LLC",
    r"\b(co\.?|company)\b": "CO",
    # France
    r"\b(sarl|societe\s+a\s+responsabilite\s+limitee)\b": "SARL",
    r"\b(sas|societe\s+par\s+actions\s+simplifiee)\b": "SAS",
    r"\b(sasu)\b": "SASU",
    r"\b(sa|societe\s+anonyme)\b": "SA",
    r"\b(sci|societe\s+civile\s+immobiliere)\b": "SCI",
    r"\b(eurl)\b": "EURL",
}

ABBREV_MAP = {
    r"\bintl\.?\b": "international",
    r"\bmfg\.?\b": "manufacturing",
    r"\bsvcs?\.?\b": "services",
    r"\btech\.?\b": "technologies",
    r"\b&\b": " and ",
}

DBA_PATTERN = re.compile(
    r"\s+(?:d/?b/?a|t/?a|trading\s+as|operating\s+as|aka)\s+", re.IGNORECASE
)

def normalize_name(raw: str) -> dict:
    if not raw:
        return {"clean_name": "", "suffix": "NONE", "dba": ""}
    
    # Unicode decomposition (French diacritics)
    text = unicodedata.normalize("NFKD", raw)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower().strip()
    
    # Extract DBA
    parts = DBA_PATTERN.split(text, maxsplit=1)
    legal_part = parts[0].strip()
    dba = parts[1].strip() if len(parts) > 1 else ""
    
    # Extract legal suffix
    suffix = "NONE"
    for pattern, canonical in SUFFIX_MAP.items():
        if re.search(pattern, legal_part):
            suffix = canonical
            legal_part = re.sub(pattern, "", legal_part)
            break
    
    # Expand abbreviations
    for pat, repl in ABBREV_MAP.items():
        legal_part = re.sub(pat, repl, legal_part)
    
    # Clean
    clean = re.sub(r"[^\w\s]", " ", legal_part)
    clean = re.sub(r"\s+", " ", clean).strip()
    
    return {"clean_name": clean, "suffix": suffix, "dba": dba}
```

---

## 3. Address Parsing & Normalization

### Multi-Country Postal Code Extraction

```python
import re

US_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
IN_PIN = re.compile(r"\b[1-9]\d{2}\s?\d{3}\b")      # 6-digit
FR_CODE = re.compile(r"\b\d{5}\b")                     # 5-digit

def extract_postal(address: str, country: str) -> str:
    if not address:
        return ""
    c = country.lower().strip() if country else ""
    
    if "india" in c or c == "in":
        m = IN_PIN.search(address)
        return m.group(0).replace(" ", "") if m else ""
    elif "us" in c or "united states" in c:
        m = US_ZIP.search(address)
        return m.group(0)[:5] if m else ""
    elif "france" in c or c == "fr":
        m = FR_CODE.search(address)
        return m.group(0) if m else ""
    else:
        m = re.search(r"\b\d{5,6}\b", address)
        return m.group(0) if m else ""

ROAD_ABBREVS = {
    r"\brd\.?\b": "road", r"\bst\.?\b": "street",
    r"\bave?\.?\b": "avenue", r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive", r"\bln\.?\b": "lane",
    r"\bct\.?\b": "court", r"\bpl\.?\b": "place",
}

IN_LANDMARKS = re.compile(
    r"\b(near|opp|opposite|behind|beside|adjacent to|next to)\s+[\w\s]+?(?=,|\.|\s*$)",
    re.IGNORECASE
)

def normalize_address(address: str, country: str) -> dict:
    if not address:
        return {"clean_addr": "", "postal": "", "building": "", "landmark": ""}
    
    text = address.lower()
    postal = extract_postal(address, country)
    
    # Building/plot number
    num_m = re.search(r"\b(?:no\.?|plot\s*no\.?|#)?\s*(\d+[a-z]?(?:/\d+)?)\b", text)
    building = num_m.group(1) if num_m else ""
    
    # India: extract and remove landmarks
    landmark = ""
    c = country.lower().strip() if country else ""
    if "india" in c:
        lm = IN_LANDMARKS.search(text)
        if lm:
            landmark = lm.group(0)
            text = text.replace(landmark, " ")
    
    # Expand road abbreviations
    for pat, repl in ROAD_ABBREVS.items():
        text = re.sub(pat, repl, text)
    
    clean = re.sub(r"[^\w\s]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    
    return {"clean_addr": clean, "postal": postal, "building": building, "landmark": landmark}
```

### Postal Code Features
- **Exact match**: Strong positive signal
- **3-digit prefix match** (India PIN / US ZIP): Same district/region
- **2-digit prefix match** (France): Same département
- **No postal code**: Weaker but address text similarity still useful

---

## 4. Phonetic Encoding

```python
import jellyfish

def phonetic_features(name1: str, name2: str) -> dict:
    if not name1 or not name2:
        return {"soundex": 0, "nysiis": 0, "metaphone": 0, "mra": 0}
    
    w1 = name1.split()[0] if name1.split() else ""
    w2 = name2.split()[0] if name2.split() else ""
    
    return {
        "soundex": 1.0 if jellyfish.soundex(w1) == jellyfish.soundex(w2) else 0.0,
        "nysiis": 1.0 if jellyfish.nysiis(w1) == jellyfish.nysiis(w2) else 0.0,
        "metaphone": 1.0 if jellyfish.metaphone(w1) == jellyfish.metaphone(w2) else 0.0,
        "mra": 1.0 if jellyfish.match_rating_comparison(w1, w2) else 0.0,
    }
```

| Algorithm | Strengths | Best Use |
|---|---|---|
| Soundex | Simple, 4-char code | Rough blocking key |
| Double Metaphone | Handles multi-origin names | Feature engineering |
| NYSIIS | Preserves vowel order | European names |
| Match Rating | Comparison-based | Binary feature |

---

## 5. Character N-Gram Features

```python
def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Padded char n-gram Jaccard. Immune to token reordering and missing spaces."""
    s1, s2 = f"^{s1.lower().strip()}$", f"^{s2.lower().strip()}$"
    if len(s1) < n or len(s2) < n:
        return 0.0
    ng1 = {s1[i:i+n] for i in range(len(s1)-n+1)}
    ng2 = {s2[i:i+n] for i in range(len(s2)-n+1)}
    return len(ng1 & ng2) / max(1, len(ng1 | ng2))
```

---

## 6. Embedding-Based Features

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embedding_features(emb1: np.ndarray, emb2: np.ndarray) -> dict:
    return {
        "emb_cosine": float(np.dot(emb1, emb2)),
        "emb_euclidean": float(np.linalg.norm(emb1 - emb2)),
        "emb_manhattan": float(np.sum(np.abs(emb1 - emb2))),
    }
```

---

## 7. Multilingual / Transliteration Handling

### India
- Transliteration variants: Choudhury/Chowdhury/Chaudhari, Laxmi/Lakshmi, Shree/Sri/Shri
- PIN code is most reliable geographic anchor
- Landmark-heavy addresses (Near SBI ATM, Opposite Railway Station)

### France (UNSEEN in training)
- Diacritics: é, è, ê, ë, à, â, ç, œ, æ → strip via NFKD
- Legal suffixes: SARL, SAS, SA, SCI, EURL
- Street prefixes: Rue, Boulevard (Bd), Avenue (Av), Allée, Place, Impasse
- Postal code first 2 digits = département (75=Paris, 69=Lyon, 13=Marseille)

```python
import unicodedata, re

def normalize_french(text: str) -> str:
    text = text.replace("œ", "oe").replace("æ", "ae")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_clean = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Remove French articles
    ascii_clean = re.sub(r"\b(de|du|des|le|la|les|d'|l')\b", " ", ascii_clean, flags=re.I)
    return re.sub(r"\s+", " ", ascii_clean).strip()

# Indian transliteration normalization
INDIC_REPLACEMENTS = [
    (r"\bshri\b|\bshree\b|\bsri\b", "shri"),
    (r"ee", "i"), (r"oo", "u"), (r"ou|ow", "au"),
    (r"ph", "f"), (r"dh", "d"), (r"bh", "b"),
]
```

---

## 8. Complete Feature Vector (~30 features)

```python
def build_features(rec1, rec2, emb1, emb2, soft_tfidf) -> dict:
    n1, n2 = rec1["clean_name"], rec2["clean_name"]
    a1, a2 = rec1["clean_addr"], rec2["clean_addr"]
    
    feats = {}
    
    # Name string metrics (7)
    name_m = compute_string_metrics(n1, n2)
    feats.update({f"name_{k}": v for k, v in name_m.items()})
    
    # Name extras (4)
    feats["name_soft_tfidf"] = soft_tfidf.score(n1, n2)
    feats["name_char3_jaccard"] = char_ngram_jaccard(n1, n2, 3)
    feats["name_char4_jaccard"] = char_ngram_jaccard(n1, n2, 4)
    feats["name_len_diff"] = abs(len(n1)-len(n2)) / max(1, max(len(n1), len(n2)))
    
    # Phonetic (4)
    feats.update({f"phon_{k}": v for k, v in phonetic_features(n1, n2).items()})
    
    # Legal suffix (2)
    s1, s2 = rec1["suffix"], rec2["suffix"]
    feats["suffix_match"] = 1.0 if s1 != "NONE" and s1 == s2 else 0.0
    feats["suffix_conflict"] = 1.0 if s1 != "NONE" and s2 != "NONE" and s1 != s2 else 0.0
    
    # Address string metrics (4)
    addr_m = compute_string_metrics(a1, a2)
    feats["addr_levenshtein"] = addr_m["levenshtein"]
    feats["addr_jaro_winkler"] = addr_m["jaro_winkler"]
    feats["addr_token_set"] = addr_m["token_set"]
    feats["addr_jaccard"] = addr_m["token_jaccard"]
    
    # Address component (3)
    p1, p2 = rec1["postal"], rec2["postal"]
    feats["postal_exact"] = 1.0 if p1 and p2 and p1 == p2 else 0.0
    feats["postal_prefix3"] = 1.0 if p1 and p2 and p1[:3] == p2[:3] else 0.0
    feats["building_match"] = 1.0 if rec1["building"] and rec1["building"] == rec2["building"] else 0.0
    
    # Country (1)
    feats["country_match"] = 1.0 if rec1["country"].lower() == rec2["country"].lower() else 0.0
    
    # Embedding (3)
    feats.update(embedding_features(emb1, emb2))
    
    return feats  # ~28 features total
```
