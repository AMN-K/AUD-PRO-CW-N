#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project Runner - Execute Tasks in Correct Order
==============================================

This script runs all project tasks in the correct sequence:

Task 2: Feature Extraction
- Extract MFCC features from Audio/ directory
- Creates features/features.npz (120-dimensional features)

Task 2 (Noisy): Create Noisy Data and Extract Features  
- Generate noisy audio at SNR 20/10/0 dB
- Extract features from noisy audio

Task 3: Baseline Training
- Train CNN on clean data only
- Creates models/task3_baseline.keras

Task 4: Noise Compensation Training
- Train baseline model on clean data
- Train matched model on clean+noisy data  
- Creates models/task4_baseline.keras and task4_matched.keras

Task 5: Final Evaluation
- Evaluate all models on all conditions
- Generate comparison results

Usage:
    python run_project.py [--task TASK_NUMBER]
    
    --task: Run specific task (2, 3, 4, 5) or 'all' for complete pipeline
"""

import sys
import subprocess
from pathlib import Path
import argparse

# Global flag for forcing re-run
FORCE_RUN = False


def run_task(task_script, description):
    """Run a task script and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {task_script}")
    print('='*60)
    
    # Check if we can skip this task (outputs already exist)
    skip = False
    if not FORCE_RUN:
        if task_script == 'tasks/task2.py':
            if Path('features/features.npz').exists():
                print("SKIPPING: features/features.npz already exists")
                skip = True
        elif task_script == 'tasks/task4_create_noise.py':
            if (Path('Audio_noisy/SNR20').exists() and 
                Path('Audio_noisy/SNR10').exists() and 
                Path('Audio_noisy/SNR0').exists()):
                print("SKIPPING: Noisy audio already exists")
                skip = True
        elif task_script == 'tasks/task2_noisy.py':
            if (Path('features/noisy_snr20.npz').exists() and 
                Path('features/noisy_snr10.npz').exists() and 
                Path('features/noisy_snr0.npz').exists()):
                print("SKIPPING: Noisy features already exist")
                skip = True
        elif task_script == 'tasks/task3.py':
            if Path('models/task3_baseline.keras').exists():
                print("SKIPPING: models/task3_baseline.keras already exists")
                skip = True
        elif task_script == 'tasks/task4_train.py':
            if (Path('models/task4_baseline.keras').exists() and 
                Path('models/task4_matched.keras').exists()):
                print("SKIPPING: Task 4 models already exist")
                skip = True
    
    if skip:
        print(f"[OK] SKIPPED: {description}")
        return True
    
    try:
        # Handle special case for task2_noisy which needs to run multiple times
        if task_script == 'tasks/task2_noisy.py':
            # Run for each SNR level
            snr_levels = [('SNR20', 'Audio_noisy/SNR20'), 
                         ('SNR10', 'Audio_noisy/SNR10'),
                         ('SNR0', 'Audio_noisy/SNR0')]
            for desc, audio_dir in snr_levels:
                output_file = f'features/noisy_{desc.lower()}'
                print(f"\nExtracting features for {desc}...")
                result = subprocess.run(
                    [sys.executable, task_script, 
                     '--audio_dir', audio_dir,
                     '--output', output_file,
                     '--description', desc],
                    check=True,
                    cwd=Path(__file__).parent,
                    capture_output=False
                )
        else:
            result = subprocess.run([sys.executable, task_script], 
                                  check=True, 
                                  cwd=Path(__file__).parent,
                                  capture_output=False)
        print(f"[OK] SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description}")
        print(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: {description}")
        print(f"Exception: {e}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run project tasks in sequence')
    parser.add_argument('--task', default='all', 
                       choices=['2', '3', '4', '5', 'all'],
                       help='Task to run (2, 3, 4, 5, or all)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run even if outputs exist')
    args = parser.parse_args()
    
    # Store force flag globally for run_task function
    global FORCE_RUN
    FORCE_RUN = args.force
    
    # Define task sequence
    tasks = [
        ('tasks/task2.py', 'Task 2: Extract Clean Features'),
        ('tasks/task4_create_noise.py', 'Task 2 (Noisy): Create Noisy Data'),  
        ('tasks/task2_noisy.py', 'Task 2 (Noisy): Extract Noisy Features'),
        ('tasks/task3.py', 'Task 3: Train Baseline CNN'),
        ('tasks/task4_train.py', 'Task 4: Noise Compensation Training'), 
        ('tasks/task5.py', 'Task 5: Final Evaluation')
    ]
    
    # Map task numbers to indices
    task_map = {
        '2': [0],           # Just clean features
        '3': [0, 3],        # Clean features + baseline training  
        '4': [0, 1, 2, 4],  # Features + noisy data + noise compensation
        '5': [0, 1, 2, 3, 4, 5],  # Everything
        'all': list(range(len(tasks)))  # All tasks
    }
    
    if args.task not in task_map:
        print(f"Invalid task: {args.task}")
        return 1
        
    # Get tasks to run
    task_indices = task_map[args.task]
    tasks_to_run = [tasks[i] for i in task_indices]
    
    print("Starting Project Execution")
    print(f"Running tasks: {args.task}")
    print(f"Working directory: {Path.cwd()}")
    
    success_count = 0
    total_tasks = len(tasks_to_run)
    
    for script, description in tasks_to_run:
        if run_task(script, description):
            success_count += 1
        else:
            print(f"\n[ERROR] Task failed. Stopping execution.")
            break
    
    print(f"\n{'='*60}")
    print(f"EXECUTION COMPLETE")
    print(f"[OK] Successful: {success_count}/{total_tasks}")
    if success_count == total_tasks:
        print("All tasks completed successfully!")
        return 0
    else:
        print(f"Failed: {total_tasks - success_count} tasks")
        return 1


if __name__ == '__main__':
    sys.exit(main())