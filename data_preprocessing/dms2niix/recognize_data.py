#!/usr/bin/env python3
"""
Script to reorganize CT data folders to match input_list.csv structure.
Converts folder names from descriptive format to sampleX format,
and renames DICOMSAVE subdirectories to dicom/.
"""

import os
import re
from pathlib import Path
import shutil
from datetime import datetime


def extract_sample_number(folder_name: str) -> int | None:
    """Extract sample number from folder name."""
    # Match patterns like "sample 7 CT..." or "sample7"
    match = re.match(r'sample\s*(\d+)', folder_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def rename_folders(data_dir: Path, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """
    Rename folders from descriptive names to sampleX format.

    Returns list of (old_path, new_path) tuples.
    """
    renamed = []

    if not data_dir.exists():
        print(f"❌ Error: {data_dir} does not exist")
        return renamed

    folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])

    for folder in folders:
        sample_num = extract_sample_number(folder.name)

        if sample_num is None:
            print(f"⚠️  Skipping {folder.name}: Cannot extract sample number")
            continue

        # Check if folder name needs renaming
        expected_name = f"sample{sample_num}"
        if "." in folder.name and "sample" in folder.name.lower():
            # Handle sample15.2 case
            parts = folder.name.split()
            if parts[0].lower().startswith("sample") and "." in parts[0]:
                expected_name = parts[0].lower()

        if folder.name == expected_name:
            print(f"✓ {folder.name}: Already correct")
            continue

        new_path = data_dir / expected_name

        if new_path.exists():
            print(f"⚠️  Skipping {folder.name}: Target {expected_name} already exists")
            continue

        if dry_run:
            print(f"[DRY RUN] Would rename: {folder.name} → {expected_name}")
        else:
            print(f"📝 Renaming: {folder.name} → {expected_name}")
            folder.rename(new_path)

        renamed.append((folder, new_path))

    return renamed


def rename_dicom_subdirs(data_dir: Path, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """
    Rename DICOMSAVE-* subdirectories to dicom/.

    Returns list of (old_path, new_path) tuples.
    """
    renamed = []

    if not data_dir.exists():
        print(f"❌ Error: {data_dir} does not exist")
        return renamed

    folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])

    for folder in folders:
        # Check subdirectories
        subdirs = [d for d in folder.iterdir() if d.is_dir()]

        dicom_subdir = None
        for subdir in subdirs:
            if subdir.name == "dicom":
                print(f"✓ {folder.name}/dicom: Already correct")
                break
            elif subdir.name.startswith("DICOMSAVE-"):
                dicom_subdir = subdir
                break

        if dicom_subdir is None:
            continue

        new_path = folder / "dicom"

        if new_path.exists():
            print(f"⚠️  Skipping {folder.name}/{dicom_subdir.name}: dicom/ already exists")
            continue

        if dry_run:
            print(f"[DRY RUN] Would rename: {folder.name}/{dicom_subdir.name} → {folder.name}/dicom")
        else:
            print(f"📝 Renaming: {folder.name}/{dicom_subdir.name} → {folder.name}/dicom")
            dicom_subdir.rename(new_path)

        renamed.append((dicom_subdir, new_path))

    return renamed


def verify_structure(data_dir: Path, csv_path: Path) -> None:
    """Verify that all folders match the expected structure from CSV."""
    import csv

    print("\n" + "="*60)
    print("Verification Report")
    print("="*60)

    # Read expected samples from CSV
    expected_samples = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected_samples.add(row['ID'])

    # Check actual folders
    actual_folders = [f for f in data_dir.iterdir() if f.is_dir()]

    print(f"\n✓ Expected samples from CSV: {len(expected_samples)}")
    print(f"✓ Actual folders in data/: {len(actual_folders)}")

    # Check each expected sample
    missing = []
    incorrect_structure = []
    correct = []

    for sample_id in sorted(expected_samples):
        sample_path = data_dir / sample_id
        dicom_path = sample_path / "dicom"

        if not sample_path.exists():
            missing.append(sample_id)
        elif not dicom_path.exists() or not dicom_path.is_dir():
            incorrect_structure.append(sample_id)
        else:
            # Count DICOM files
            dcm_files = list(dicom_path.glob("*.dcm"))
            correct.append((sample_id, len(dcm_files)))

    print(f"\n✅ Correct structure: {len(correct)}")
    for sample_id, file_count in correct[:5]:  # Show first 5
        print(f"   {sample_id}/dicom/ ({file_count} files)")
    if len(correct) > 5:
        print(f"   ... and {len(correct) - 5} more")

    if incorrect_structure:
        print(f"\n⚠️  Incorrect structure: {len(incorrect_structure)}")
        for sample_id in incorrect_structure[:10]:
            print(f"   {sample_id}")

    if missing:
        print(f"\n❌ Missing folders: {len(missing)}")
        for sample_id in missing[:10]:
            print(f"   {sample_id}")


def main():
    """Main execution function."""
    data_dir = Path("/home/yuya/research/dcm2niix/data")
    csv_path = Path("/home/yuya/research/dcm2niix/input_list.csv")

    print("="*60)
    print("CT Data Folder Reorganization Script")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"CSV file: {csv_path}")
    print()

    # Step 1: Rename folders
    print("\n" + "="*60)
    print("Step 1: Renaming folders to sampleX format")
    print("="*60)
    renamed_folders = rename_folders(data_dir, dry_run=False)
    print(f"\n✓ Renamed {len(renamed_folders)} folders")

    # Step 2: Rename DICOM subdirectories
    print("\n" + "="*60)
    print("Step 2: Renaming DICOM subdirectories")
    print("="*60)
    renamed_subdirs = rename_dicom_subdirs(data_dir, dry_run=False)
    print(f"\n✓ Renamed {len(renamed_subdirs)} DICOM subdirectories")

    # Step 3: Verify structure
    verify_structure(data_dir, csv_path)

    print("\n" + "="*60)
    print("Reorganization Complete!")
    print("="*60)
    print(f"Total operations: {len(renamed_folders) + len(renamed_subdirs)}")
    print()


if __name__ == "__main__":
    main()
