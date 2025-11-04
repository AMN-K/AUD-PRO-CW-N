import numpy as np
from pathlib import Path
from collections import defaultdict

mfcc_dir = Path('mfccs_data')
files = sorted(mfcc_dir.glob('*.npy'))

print(f"Checking {len(files)} MFCC files for duplicates...")
print("="*70)

# Group by speaker
speakers = defaultdict(list)
for f in files:
    speaker = ''.join([c for c in f.stem if not c.isdigit()]).strip()
    speakers[speaker].append(f)

duplicates_found = 0
speakers_with_dups = []

for speaker, spk_files in sorted(speakers.items()):
    # Load all data for this speaker
    data = [np.load(f) for f in spk_files]
    
    # Find duplicates
    seen = {}
    dups = []
    
    for i, d in enumerate(data):
        # Create a hash of the data
        data_hash = hash(d.tobytes())
        
        if data_hash in seen:
            dups.append((i, seen[data_hash]))
        else:
            seen[data_hash] = i
    
    if dups:
        print(f"\n{speaker.upper()}: {len(dups)} duplicate(s) found")
        speakers_with_dups.append(speaker)
        duplicates_found += len(dups)
        
        # Show first few duplicates
        for idx, (dup_idx, orig_idx) in enumerate(dups[:5]):
            print(f"  {spk_files[dup_idx].name} is duplicate of {spk_files[orig_idx].name}")
        
        if len(dups) > 5:
            print(f"  ... and {len(dups)-5} more duplicates")

print("\n" + "="*70)
print(f"\nSummary:")
print(f"  Total duplicates: {duplicates_found}")
print(f"  Speakers affected: {len(speakers_with_dups)}")
if speakers_with_dups:
    print(f"  Affected speakers: {', '.join(speakers_with_dups)}")
