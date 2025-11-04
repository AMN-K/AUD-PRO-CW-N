import numpy as np
from pathlib import Path
from collections import defaultdict

mfcc_dir = Path('mfccs_data')

# Check each speaker
speakers = ['ahmed', 'amber', 'charlie', 'emad', 'ngozi']
print("Speaker Data Analysis:")
print("="*70)

for speaker in speakers:
    files = sorted(mfcc_dir.glob(f'{speaker}*.npy'))
    print(f"\n{speaker.upper()} ({len(files)} files):")
    
    if len(files) == 0:
        print("  NO FILES FOUND!")
        continue
    
    data = [np.load(f) for f in files]
    
    # Check shapes
    shapes = [d.shape for d in data]
    print(f"  Shapes: {set(shapes)}")
    
    # Check statistics
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]
    
    print(f"  Mean range: {min(means):.4f} to {max(means):.4f}")
    print(f"  Std range:  {min(stds):.4f} to {max(stds):.4f}")
    
    # Check for outliers
    mean_avg = np.mean(means)
    std_avg = np.mean(stds)
    
    # Find anomalies (more than 2 std deviations)
    mean_threshold = 2 * np.std(means)
    anomalies = [i for i, m in enumerate(means) if abs(m - mean_avg) > mean_threshold]
    
    if anomalies:
        print(f"  WARNING: {len(anomalies)} anomalous files detected!")
        print(f"    Files: {[files[i].name for i in anomalies[:5]]}")
    
    # Sample comparison
    print(f"  First 3 samples:")
    for i in range(min(3, len(files))):
        print(f"    {files[i].name}: mean={means[i]:.4f}, std={stds[i]:.4f}")

print("\n" + "="*70)
