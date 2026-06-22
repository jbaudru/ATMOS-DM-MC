"""
Inspect the WorldMove 380_US_New_York.npz dataset structure and content.
"""

import numpy as np
from pathlib import Path

npz_path = Path(r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\380_US_New_York.npz')

print("="*70)
print("INSPECTING: 380_US_New_York.npz")
print("="*70)

data = np.load(npz_path, allow_pickle=True)

print("\nAvailable arrays in NPZ:")
for key in data.files:
    print(f"\n  {key}:")
    obj = data[key]
    print(f"    Type: {type(obj)}")
    print(f"    Dtype: {obj.dtype}")
    print(f"    Shape: {obj.shape}")
    
    # For small objects, show content
    if obj.dtype == object:
        if hasattr(obj.item(), 'keys'):
            print(f"    Keys: {list(obj.item().keys())}")
            sample = obj.item()
            for k, v in list(sample.items())[:3]:
                print(f"      {k}: {type(v)} - {v if not isinstance(v, np.ndarray) else f'array shape {v.shape}'}")
        else:
            print(f"    Content type: {type(obj.item())}")
    else:
        if obj.size <= 20:
            print(f"    Content: {obj}")
        else:
            print(f"    First 5 elements: {obj.flat[:5]}")

print("\n" + "="*70)
