#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reset Project Outputs
=====================
Removes all generated outputs to allow fresh execution from the beginning.

Deletes:
- Generated features (features/*.npz, features/*.json)
- Generated noisy audio (Audio_noisy/)
- Trained models (models/*.keras)
- Result files (results/**/*.txt, results/**/*.png)

Preserves:
- Original audio files (Audio/)
- Noise files (Audio_noise/)
- Source code (tasks/, helpers/)
- Documentation (docs/)

Usage:
    python reset_project.py [--confirm]
    
    --confirm: Skip confirmation prompt and reset immediately
"""

import os
import shutil
from pathlib import Path
import argparse


def reset_project(confirm=False):
    """Reset project by removing all generated outputs."""
    
    items_to_delete = [
        # Features
        ('features/features.npz', 'file'),
        ('features/features.json', 'file'),
        ('features/noisy_snr0.npz', 'file'),
        ('features/noisy_snr0.json', 'file'),
        ('features/noisy_snr10.npz', 'file'),
        ('features/noisy_snr10.json', 'file'),
        ('features/noisy_snr20.npz', 'file'),
        ('features/noisy_snr20.json', 'file'),
        
        # Noisy audio
        ('Audio_noisy', 'dir'),
        
        # Models
        ('models/task3_baseline.keras', 'file'),
        ('models/task4_baseline.keras', 'file'),
        ('models/task4_matched.keras', 'file'),
        
        # Results
        ('results/task3', 'dir'),
        ('results/task4', 'dir'),
    ]
    
    # Print what will be deleted
    print("="*70)
    print("PROJECT RESET - Items to Delete")
    print("="*70)
    
    existing_items = []
    for item_path, item_type in items_to_delete:
        path = Path(item_path)
        if path.exists():
            existing_items.append((path, item_type))
            print(f"  - {item_path}")
    
    if not existing_items:
        print("\nNo items to delete. Project is already clean.")
        return
    
    print(f"\nTotal: {len(existing_items)} items")
    print("="*70)
    
    # Confirm deletion
    if not confirm:
        response = input("\nProceed with deletion? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Reset cancelled.")
            return
    
    # Delete items
    print("\nDeleting items...")
    deleted_count = 0
    error_count = 0
    
    for path, item_type in existing_items:
        try:
            if item_type == 'dir':
                shutil.rmtree(path)
                print(f"  Deleted directory: {path}")
            else:
                path.unlink()
                print(f"  Deleted file: {path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ERROR deleting {path}: {e}")
            error_count += 1
    
    # Recreate empty directories
    print("\nRecreating directory structure...")
    Path('features').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('results/task3').mkdir(parents=True, exist_ok=True)
    Path('results/task4/baseline').mkdir(parents=True, exist_ok=True)
    Path('results/task4/matched').mkdir(parents=True, exist_ok=True)
    print("  Created empty directories")
    
    print("\n" + "="*70)
    print("RESET COMPLETE")
    print("="*70)
    print(f"Deleted: {deleted_count} items")
    if error_count > 0:
        print(f"Errors: {error_count} items")
    print("\nProject is ready for fresh execution.")
    print("Run: python run_project.py --task all")
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Reset project by removing all generated outputs'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )
    args = parser.parse_args()
    
    reset_project(confirm=args.confirm)


if __name__ == '__main__':
    main()
