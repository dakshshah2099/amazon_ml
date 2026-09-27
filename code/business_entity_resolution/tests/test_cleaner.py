import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from cleaning_utils import clean_record, check_has_state

def test_transliteration():
    # Hindi/Devanagari
    name, addr, _, _, t_name, t_addr = clean_record('राम मार्केटिंग प्राइवेट लिमिटेड', 'मुंबई, महाराष्ट्र')
    assert t_name is True
    assert t_addr is True
    assert 'rama' in name
    assert 'limiteda' in name
    assert 'maharashtra' in addr

    # Telugu
    name2, addr2, _, _, t_name2, t_addr2 = clean_record('కృష్ణా ఇంపెక్స్ లిమిటెడ్', 'హైదరాబాద్ shop')
    assert t_name2 is True
    assert t_addr2 is True
    assert 'limi' in name2
    assert 'haidar' in addr2

def test_landmark_removal():
    # Landmark phrases up to next comma or end
    raw_addr = '123 Main St, Near SBI ATM, Suite 400'
    _, clean_addr, _, _, _, _ = clean_record('Acme', raw_addr)
    assert 'near sbi atm' not in clean_addr
    assert 'suite 400' in clean_addr

    raw_addr2 = 'Shop 5, opp. RTA Office'
    _, clean_addr2, _, _, _, _ = clean_record('Acme', raw_addr2)
    assert 'opp' not in clean_addr2
    assert 'rta office' not in clean_addr2

    raw_addr3 = 'Behind Grand Mall, MG Road'
    _, clean_addr3, _, _, _, _ = clean_record('Acme', raw_addr3)
    assert 'behind grand mall' not in clean_addr3
    assert 'mg road' in clean_addr3

def test_abbreviation_expansion():
    # Punctuation-attached name abbreviations
    name, _, _, _, _, _ = clean_record('Apex Corp., Pvt. Ltd. & Co.', '100 Main St.')
    assert 'corporation' in name
    assert 'private' in name
    assert 'limited' in name
    assert 'company' in name
    assert 'and' in name
    assert 'ltd.' not in name
    assert 'pvt.' not in name
    assert 'corp.' not in name

    # Protected entities (llc, llp, pllc, sarl, etc.)
    name_prot, _, _, _, _, _ = clean_record('Tech LLC, Legal LLP, France SARL', '100 Main St.')
    assert 'llc' in name_prot
    assert 'llp' in name_prot
    assert 'sarl' in name_prot

    # Address abbreviations (attached and unattached)
    _, addr, _, _, _, _ = clean_record('Acme', '100 Main St., 5th Ave. Rd., Fl. 2, Apt. 4B, Ste. 100, North Blvd.')
    assert 'street' in addr
    assert 'avenue' in addr
    assert 'road' in addr
    assert 'floor' in addr
    assert 'apartment' in addr
    assert 'suite' in addr
    assert 'boulevard' in addr

def test_postal_code_extraction():
    # 5 or 6 digit postal code
    _, addr, postal, _, _, _ = clean_record('Acme', '123 Main St, Austin, TX 78701')
    assert postal == '78701'
    assert '78701' not in addr

    _, addr2, postal2, _, _, _ = clean_record('Acme', 'Bangalore, Karnataka 560001')
    assert postal2 == '560001'
    assert '560001' not in addr2

def test_state_detection_precision():
    # True positives: uppercase 2-letter codes or full names
    assert clean_record('A', 'Austin, TX')[3] is True
    assert clean_record('A', 'Portland, OR')[3] is True
    assert clean_record('A', 'Bangor, ME')[3] is True
    assert clean_record('A', 'New Delhi, DL')[3] is True
    assert clean_record('A', 'Pune, Maharashtra')[3] is True
    assert clean_record('A', 'Lille, Hauts-de-France')[3] is True
    assert clean_record('A', 'Bordeaux, Nouvelle-Aquitaine')[3] is True

    # False positive prevention: lowercase common English words ('in', 'or', 'me')
    assert clean_record('A', 'Unit in building')[3] is False
    assert clean_record('A', 'Shop or office')[3] is False
    assert clean_record('A', 'Contact me at 555')[3] is False
    assert clean_record('A', 'Floor 2 in plaza')[3] is False

def test_missing_values_sentinels():
    # Sentinels should be treated as empty
    for s in ['null', 'NAN', 'None', 'n/a', 'NA', '']:
        name, addr, postal, state, t_n, t_a = clean_record(s, s)
        assert name == ''
        assert addr == ''
        assert postal == ''
        assert state is False
        assert t_n is False
        assert t_a is False

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
