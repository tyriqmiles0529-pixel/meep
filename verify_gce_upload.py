#!/usr/bin/env python
"""
GCE Upload Verification Script
Verifies integrity of uploaded models using checksums

Usage on GCE:
    python verify_gce_upload.py --checksum-file model_checksums.json
"""

import argparse
import json
import hashlib
import sys
from pathlib import Path


def calculate_file_checksum(file_path):
    """Calculate SHA256 checksum of a file"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error calculating checksum for {file_path}: {e}")
        return None


def verify_upload(checksum_file):
    """Verify uploaded files against checksums"""
    print("=== GCE UPLOAD INTEGRITY VERIFICATION ===")
    
    # Load checksums
    try:
        with open(checksum_file, 'r') as f:
            checksums = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load checksum file: {e}")
        return False
    
    print(f"Loaded checksums from: {checksum_file}")
    print(f"Generated at: {checksums.get('generated_at', 'Unknown')}")
    
    # Verify window models
    window_errors = []
    window_success = 0
    
    print(f"\nVerifying {len(checksums.get('window_models', {}))} window models...")
    
    for rel_path, expected_hash in checksums.get('window_models', {}).items():
        file_path = Path('player_models') / rel_path
        
        if not file_path.exists():
            window_errors.append(f'Missing: {rel_path}')
            continue
        
        actual_hash = calculate_file_checksum(file_path)
        if actual_hash is None:
            window_errors.append(f'Failed to read: {rel_path}')
            continue
        
        if actual_hash != expected_hash:
            window_errors.append(f'Checksum mismatch: {rel_path}')
        else:
            window_success += 1
            print(f"✅ {rel_path}")
    
    # Verify meta models  
    meta_errors = []
    meta_success = 0
    
    print(f"\nVerifying {len(checksums.get('meta_models', {}))} meta models...")
    
    for rel_path, expected_hash in checksums.get('meta_models', {}).items():
        file_path = Path('meta_models') / rel_path
        
        if not file_path.exists():
            meta_errors.append(f'Missing: {rel_path}')
            continue
        
        actual_hash = calculate_file_checksum(file_path)
        if actual_hash is None:
            meta_errors.append(f'Failed to read: {rel_path}')
            continue
        
        if actual_hash != expected_hash:
            meta_errors.append(f'Checksum mismatch: {rel_path}')
        else:
            meta_success += 1
            print(f"✅ {rel_path}")
    
    # Summary
    print(f"\n=== VERIFICATION SUMMARY ===")
    print(f"Window models: {window_success} verified, {len(window_errors)} errors")
    print(f"Meta models: {meta_success} verified, {len(meta_errors)} errors")
    
    if window_errors or meta_errors:
        print(f"\n❌ VERIFICATION FAILED:")
        for error in window_errors + meta_errors:
            print(f"  - {error}")
        return False
    else:
        print(f"\n✅ ALL FILES VERIFIED SUCCESSFULLY")
        print(f"✅ Upload integrity confirmed")
        return True


def main():
    parser = argparse.ArgumentParser(description="Verify GCE upload integrity")
    parser.add_argument("--checksum-file", required=True, help="Path to checksum JSON file")
    
    args = parser.parse_args()
    
    if not Path(args.checksum_file).exists():
        print(f"❌ Checksum file not found: {args.checksum_file}")
        sys.exit(1)
    
    success = verify_upload(args.checksum_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
