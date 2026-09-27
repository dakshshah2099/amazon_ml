import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

INDIC_SCRIPTS = [
    ('devanagari', re.compile(r'[\u0900-\u097F]')),
    ('bengali',    re.compile(r'[\u0980-\u09FF]')),
    ('gurmukhi',   re.compile(r'[\u0A00-\u0A7F]')),
    ('gujarati',   re.compile(r'[\u0A80-\u0AFF]')),
    ('oriya',      re.compile(r'[\u0B00-\u0B7F]')),
    ('tamil',      re.compile(r'[\u0B80-\u0BFF]')),
    ('telugu',     re.compile(r'[\u0C00-\u0C7F]')),
    ('kannada',    re.compile(r'[\u0C80-\u0CFF]')),
    ('malayalam',  re.compile(r'[\u0D00-\u0D7F]'))
]

NON_LATIN_INDIC_REGEX = re.compile(r'[\u0900-\u0D7F]')
PUNCT_STEP4_REGEX = re.compile(r'[\[\]#*@!><_:+|]')
FINAL_PUNCT_REGEX = re.compile(r'[,\\-]')
POSTAL_CODE_REGEX = re.compile(r'\b\d{5,6}\b')
LANDMARK_REGEX = re.compile(r'(?i)\b(near|opp\.?|opposite|behind)\s+[^,]+')
WHITESPACE_REGEX = re.compile(r'\s+')

# URL and Domain regexes for web entity normalization
URL_PREFIX_REGEX = re.compile(r'(?i)\bhttps?:\/\/(?:www\.)?|\bwww\.')
DOMAIN_TLD_REGEX = re.compile(r'(?i)\.(?:com|org|net|co\.in|gov|edu|mil|in|co|io|biz|info|ai|us|me|org\.in|net\.in)(?:\/.*)?\b')

def strip_domain_and_url(text):
    """
    Strips URL schemes (http://, https://, www.) and top-level domain extensions (.com, .org, .in, etc.)
    from entity names.
    """
    if not text:
        return text
    t = URL_PREFIX_REGEX.sub('', text)
    t = DOMAIN_TLD_REGEX.sub('', t)
    return t

# Standard missing value sentinels (do not treat as valid business names/addresses)
SENTINELS = {'', 'null', 'nan', 'none', 'n/a', 'na', 'undefined'}

NAME_TOKEN_MAP = {
    'ltd': 'limited',
    'pvt': 'private',
    'corp': 'corporation',
    'inc': 'incorporated',
    'co': 'company',
    '&': 'and'
}

ADDR_TOKEN_MAP = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'dr': 'drive',
    'fl': 'floor',
    'apt': 'apartment',
    'ste': 'suite',
    'blvd': 'boulevard'
}

# Lookup set for has_state
US_STATES = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
    'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
    'Wisconsin', 'Wyoming', 'District of Columbia'
]
US_CODES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA',
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
    'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

INDIA_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat',
    'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
    'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
    'Uttarakhand', 'West Bengal',
    'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu',
    'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
]
INDIA_CODES = [
    'MH', 'DL', 'KA', 'TN', 'WB', 'TS', 'AP', 'UP', 'GJ', 'RJ', 'MP', 'KL', 'HR', 'PB',
    'OR', 'BR', 'JH', 'CG', 'CT', 'GA', 'HP', 'JK', 'AS', 'UT', 'UK', 'TR', 'ML', 'MN',
    'NL', 'MZ', 'SK', 'AR', 'AN', 'CH', 'DN', 'DD', 'LA', 'LD', 'PY'
]

FRENCH_REGIONS = [
    'Auvergne-Rhône-Alpes', 'Auvergne-Rhone-Alpes',
    'Bourgogne-Franche-Comté', 'Bourgogne-Franche-Comte',
    'Bretagne', 'Brittany',
    'Centre-Val de Loire',
    'Corse', 'Corsica',
    'Grand Est',
    'Hauts-de-France',
    'Île-de-France', 'Ile-de-France',
    'Normandie', 'Normandy',
    'Nouvelle-Aquitaine',
    'Occitanie',
    'Pays de la Loire',
    "Provence-Alpes-Côte d'Azur", "Provence-Alpes-Cote d'Azur", 'PACA',
    'Guadeloupe', 'Guyane', 'French Guiana', 'Martinique', 'Mayotte', 'La Réunion', 'La Reunion', 'Reunion'
]

# Case-sensitive 2-letter uppercase codes (prevents false matches on words like 'in', 'or', 'me')
TWO_LETTER_STATE_CODES = set(US_CODES + INDIA_CODES)
TWO_LETTER_REGEX = re.compile(r'(?<![a-zA-Z])([A-Z]{2})(?![a-zA-Z])')

# Full state & region names for case-insensitive matching
FULL_STATE_NAMES = set(US_STATES + INDIA_STATES + FRENCH_REGIONS)
_escaped_names = sorted([re.escape(t) for t in FULL_STATE_NAMES], key=len, reverse=True)
FULL_STATE_NAME_REGEX = re.compile(r'\b(?:' + '|'.join(_escaped_names) + r')\b', re.IGNORECASE)

def transliterate_if_needed(text):
    for s_name, pat in INDIC_SCRIPTS:
        if pat.search(text):
            text = transliterate(text, getattr(sanscript, s_name.upper()), sanscript.ITRANS)
    return text

def expand_tokens_with_punctuation(text, mapping):
    """
    Expands tokens while correctly handling attached punctuation like Ltd., St., Rd., etc.
    Preserves commas for subsequent steps.
    """
    tokens = text.split()
    expanded = []
    for tok in tokens:
        # Separate trailing punctuation such as periods and commas
        clean = tok.rstrip('.,')
        punct = tok[len(clean):]
        if clean in mapping:
            # Map abbreviation and re-attach comma if present
            expanded.append(mapping[clean] + (',' if ',' in punct else ''))
        else:
            expanded.append(tok)
    return ' '.join(expanded)

def check_has_state(raw_addr, clean_addr_intermediate):
    """
    High-precision state detection:
    - 2-letter postal codes match ONLY if uppercase in the raw address (eliminating 'in', 'or', 'me', 'as', 'la')
    - Full state/region names match case-insensitively on whole phrase boundaries
    """
    # 1. Check uppercase 2-letter codes in raw address
    found_codes = TWO_LETTER_REGEX.findall(raw_addr)
    for c in found_codes:
        if c in TWO_LETTER_STATE_CODES:
            return True
            
    # 2. Check full names case-insensitively
    if FULL_STATE_NAME_REGEX.search(raw_addr) or FULL_STATE_NAME_REGEX.search(clean_addr_intermediate):
        return True
        
    return False

def clean_record(raw_name, raw_addr):
    """
    Clean an individual record (name, address) following the 10-step spec.
    """
    raw_name_str = '' if raw_name is None else str(raw_name).strip()
    raw_addr_str = '' if raw_addr is None else str(raw_addr).strip()

    # Sentinel check: if text is pure missing sentinel, normalize to empty
    is_name_sentinel = raw_name_str.lower() in SENTINELS
    is_addr_sentinel = raw_addr_str.lower() in SENTINELS

    if is_name_sentinel:
        clean_name = ''
        needs_trans_name = False
    else:
        # Step 1
        needs_trans_name = bool(NON_LATIN_INDIC_REGEX.search(raw_name_str))
        n = raw_name_str
        # Step 2
        if needs_trans_name:
            n = transliterate_if_needed(n)
        # Step 3
        n = n.lower()
        # Domain and URL normalization
        n = strip_domain_and_url(n)
        # Step 4
        n = PUNCT_STEP4_REGEX.sub(' ', n)
        # Step 5: Suffix expansion with attached punctuation support
        n = expand_tokens_with_punctuation(n, NAME_TOKEN_MAP)
        # Final punct strip
        n = FINAL_PUNCT_REGEX.sub(' ', n)
        # Step 10
        clean_name = WHITESPACE_REGEX.sub(' ', n).strip()

    if is_addr_sentinel:
        clean_addr = ''
        postal_code = ''
        has_state = False
        needs_trans_addr = False
    else:
        # Step 1
        needs_trans_addr = bool(NON_LATIN_INDIC_REGEX.search(raw_addr_str))
        a = raw_addr_str
        # Step 2
        if needs_trans_addr:
            a = transliterate_if_needed(a)
        # Step 3
        a = a.lower()
        # Step 4
        a = PUNCT_STEP4_REGEX.sub(' ', a)

        # Step 7: PIN/ZIP extraction BEFORE final punct strip
        m_zip = POSTAL_CODE_REGEX.search(a)
        if m_zip:
            postal_code = m_zip.group()
            a = a[:m_zip.start()] + ' ' + a[m_zip.end():]
        else:
            postal_code = ''

        # Step 8: has_state flag with high-precision filtering
        has_state = check_has_state(raw_addr_str, a)

        # Step 9: landmark removal
        a = LANDMARK_REGEX.sub('', a)

        # Step 6: Address abbreviation expansion with attached punctuation support
        a = expand_tokens_with_punctuation(a, ADDR_TOKEN_MAP)

        # Step 4 final punct strip on address
        a = FINAL_PUNCT_REGEX.sub(' ', a)

        # Step 10: whitespace collapse
        clean_addr = WHITESPACE_REGEX.sub(' ', a).strip()

    return clean_name, clean_addr, postal_code, has_state, needs_trans_name, needs_trans_addr

def clean_for_embedding(raw_name, raw_addr):
    """
    Minimal cleaning for embedding models: transliterate non-Latin scripts
    to Latin, strip only control/junk punctuation, but preserve case,
    word order, and abbreviations.
    """
    raw_name_str = '' if raw_name is None else str(raw_name).strip()
    raw_addr_str = '' if raw_addr is None else str(raw_addr).strip()

    if raw_name_str.lower() in SENTINELS:
        name_out = ''
    else:
        name_out = transliterate_if_needed(raw_name_str) if NON_LATIN_INDIC_REGEX.search(raw_name_str) else raw_name_str
        name_out = strip_domain_and_url(name_out)
        name_out = WHITESPACE_REGEX.sub(' ', name_out).strip()

    if raw_addr_str.lower() in SENTINELS:
        addr_out = ''
    else:
        addr_out = transliterate_if_needed(raw_addr_str) if NON_LATIN_INDIC_REGEX.search(raw_addr_str) else raw_addr_str
        addr_out = WHITESPACE_REGEX.sub(' ', addr_out).strip()

    return name_out, addr_out

def clean_batch(batch_records):
    """
    batch_records: list of tuples (entity_id, country, business_name, business_address)
    returns: list of tuples for output columns
    """
    results = []
    for entity_id, country, raw_name, raw_addr in batch_records:
        clean_name, clean_addr, postal_code, has_state, needs_trans_name, needs_trans_addr = clean_record(raw_name, raw_addr)
        embed_name, embed_addr = clean_for_embedding(raw_name, raw_addr)
        results.append((
            entity_id,
            country,
            '' if raw_name is None else str(raw_name),
            '' if raw_addr is None else str(raw_addr),
            clean_name,
            clean_addr,
            postal_code,
            has_state,
            needs_trans_name,
            needs_trans_addr,
            embed_name,
            embed_addr
        ))
    return results
