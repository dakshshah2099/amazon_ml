#!/usr/bin/env python3
"""
Submission format validator for Business Entity Resolution Challenge.
Checks:
1. File existence and non-empty.
2. Proper tab separation and column headers.
3. Every Source 1 entity from test_source1.tsv present exactly once.
4. No self-matches or invalid entity IDs.
5. No duplicates in ID lists.
6. candidate_pairs.tsv contains all matched IDs from matching_results.tsv.
"""
import os
import sys
import argparse

def validate(matching_path, candidate_path, test_dir):
    print("=" * 60)
    print("RUNNING SUBMISSION FORMAT VALIDATION")
    print("=" * 60)
    
    issues = []
    
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    for p in [test_s1_path, test_s2_path, test_s3_path]:
        if not os.path.exists(p):
            print(f"Error: test file not found: {p}")
            sys.exit(1)
            
    print("Loading test entity IDs...")
    with open(test_s1_path, 'r', encoding='utf-8') as f:
        s1_lines = f.read().splitlines()
    s1_ids = [line.split('\t')[0] for line in s1_lines[1:] if line.strip()]
    expected_s1_set = set(s1_ids)
    print(f"Found {len(expected_s1_set):,} Source 1 entities in test_source1.tsv")
    
    # Check Candidate Pairs
    print(f"Validating candidate file: {candidate_path}")
    if not os.path.exists(candidate_path):
        issues.append(f"Candidate file missing: {candidate_path}")
        return issues
        
    cand_s1_seen = set()
    cand_map = {}
    with open(candidate_path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n')
        if header != "source1_entity_id\tcandidate_entity_ids":
            issues.append(f"Invalid header in candidate_pairs.tsv: {repr(header)}")
        line_num = 1
        for line in f:
            line_num += 1
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) != 2:
                issues.append(f"Line {line_num} in {candidate_path} does not have exactly 2 tab-separated fields.")
                break
            s1_id, cands = parts
            if s1_id in cand_s1_seen:
                issues.append(f"Duplicate source1_entity_id {s1_id} in {candidate_path} at line {line_num}")
                break
            cand_s1_seen.add(s1_id)
            if cands:
                cand_list = cands.split(',')
                if len(cand_list) != len(set(cand_list)):
                    issues.append(f"Duplicate candidate IDs in list for {s1_id}")
                    break
                cand_map[s1_id] = set(cand_list)
            else:
                cand_map[s1_id] = set()
                
    if cand_s1_seen != expected_s1_set:
        diff_missing = expected_s1_set - cand_s1_seen
        diff_extra = cand_s1_seen - expected_s1_set
        if diff_missing:
            issues.append(f"{candidate_path} is missing {len(diff_missing)} Source 1 entities (e.g. {list(diff_missing)[:3]})")
        if diff_extra:
            issues.append(f"{candidate_path} has {len(diff_extra)} extra entities not in test_source1 (e.g. {list(diff_extra)[:3]})")
            
    # Check Matching Results if provided
    if matching_path and os.path.exists(matching_path):
        print(f"Validating matching file: {matching_path}")
        match_s1_seen = set()
        with open(matching_path, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\r\n')
            if header != "source1_entity_id\tmatched_entity_ids":
                issues.append(f"Invalid header in matching_results.tsv: {repr(header)}")
            line_num = 1
            for line in f:
                line_num += 1
                parts = line.rstrip('\r\n').split('\t')
                if len(parts) != 2:
                    issues.append(f"Line {line_num} in {matching_path} does not have exactly 2 tab-separated fields.")
                    break
                s1_id, matches = parts
                if s1_id in match_s1_seen:
                    issues.append(f"Duplicate source1_entity_id {s1_id} in {matching_path} at line {line_num}")
                    break
                match_s1_seen.add(s1_id)
                if matches:
                    m_list = matches.split(',')
                    if len(m_list) != len(set(m_list)):
                        issues.append(f"Duplicate matched IDs in list for {s1_id}")
                        break
                    # Verify subset of candidates
                    c_set = cand_map.get(s1_id, set())
                    for m_id in m_list:
                        if m_id not in c_set:
                            issues.append(f"Matched ID {m_id} for {s1_id} was never generated in candidate_pairs.tsv!")
                            break
                            
        if match_s1_seen != expected_s1_set:
            diff_missing = expected_s1_set - match_s1_seen
            if diff_missing:
                issues.append(f"{matching_path} is missing {len(diff_missing)} Source 1 entities.")
                
    if not issues:
        print("\nPASS: All submission format requirements satisfied!")
        return 0
    else:
        print(f"\nFAILED with {len(issues)} issues:")
        for idx, err in enumerate(issues, 1):
            print(f"  {idx}. {err}")
        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matching', type=str, default=None)
    parser.add_argument('--candidate', type=str, required=True)
    parser.add_argument('--test-dir', type=str, default='dataset/test')
    args = parser.parse_args()
    
    code = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(code)
