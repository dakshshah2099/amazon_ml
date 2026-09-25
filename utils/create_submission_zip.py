#!/usr/bin/env python3
"""
ML Challenge 2026 — Submission Packager & Validator

Packages submission files into the required archive structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── pipeline.ipynb
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Usage:
    python utils/create_submission_zip.py --team-name my_team
"""

import argparse
import os
import subprocess
import sys
import zipfile

REQUIRED_FILES = [
    ("output/matching_results.tsv", "output/matching_results.tsv"),
    ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
    ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
    ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ("Documentation_template.md", "Documentation_template.md"),
]

CODE_SRC_DIR = "code/business_entity_resolution/src"


def check_prerequisites():
    """Verify all mandatory files and folders exist before packaging."""
    missing = []
    for local_path, _ in REQUIRED_FILES:
        if not os.path.isfile(local_path):
            missing.append(local_path)

    if not os.path.isdir(CODE_SRC_DIR):
        missing.append(CODE_SRC_DIR)
    else:
        src_files = [f for f in os.listdir(CODE_SRC_DIR) if not f.startswith(".")]
        if not src_files:
            missing.append(f"{CODE_SRC_DIR} (directory is empty)")

    return missing


def run_validator(matching_path, candidate_path, test_dir):
    """Run official validate_submission.py before zipping."""
    validator_path = "student_resource/utils/validate_submission.py"
    if not os.path.isfile(validator_path):
        print(f"[WARN] Validator script not found at {validator_path}. Skipping check.")
        return True

    cmd = [
        sys.executable,
        validator_path,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]
    print(f"[INFO] Running submission validator: {' '.join(cmd)}")
    p = subprocess.run(cmd)
    return p.returncode == 0


def create_zip(team_name, output_zip_path):
    """Build the submission zip archive with exact relative paths."""
    print(f"\n[INFO] Creating submission package: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add required individual files
        for local_path, arcname in REQUIRED_FILES:
            zf.write(local_path, arcname=arcname)
            size_kb = os.path.getsize(local_path) / 1024
            print(f"  + Added {arcname} ({size_kb:.1f} KB)")

        # Add code/business_entity_resolution/src recursively
        for root, _, files in os.walk(CODE_SRC_DIR):
            for file in files:
                if file.startswith(".") or file.endswith((".pyc", ".pyo")):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=".")
                # normalize path separators to forward slash
                arcname = rel_path.replace(os.sep, "/")
                zf.write(full_path, arcname=arcname)
                size_kb = os.path.getsize(full_path) / 1024
                print(f"  + Added {arcname} ({size_kb:.1f} KB)")

    total_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Package created successfully: {output_zip_path} ({total_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build Amazon ML Challenge submission zip package.")
    parser.add_argument(
        "--team-name", "-t",
        default="team_solution",
        help="Team name used for the zip file (<team_name>_submission.zip). Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory where the zip file will be saved. Default: %(default)s",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip running validate_submission.py before packaging.",
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test" if os.path.isdir("dataset/test") else "student_resource/dataset/test",
        help="Test dataset directory for validation check. Default: %(default)s",
    )
    args = parser.parse_args()

    # Clean team name
    clean_team = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in args.team_name.strip())
    zip_filename = f"{clean_team}_submission.zip"
    zip_filepath = os.path.join(args.output_dir, zip_filename)

    # 1. Check prerequisites
    missing = check_prerequisites()
    if missing:
        print("[ERROR] Cannot build submission zip. Missing required files/directories:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)

    # 2. Run validator
    if not args.skip_validation:
        valid = run_validator(
            matching_path="output/matching_results.tsv",
            candidate_path="output/candidate_pairs.tsv",
            test_dir=args.test_dir,
        )
        if not valid:
            print("\n[ERROR] Submission validation failed! Fix reported errors before packaging.")
            print("        (Pass --skip-validation to bypass if testing).")
            sys.exit(1)

    # 3. Create zip
    os.makedirs(args.output_dir, exist_ok=True)
    create_zip(clean_team, zip_filepath)


if __name__ == "__main__":
    main()
