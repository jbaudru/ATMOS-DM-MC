"""
Create minimal test dataset with 'date' column for LTM/STM validation.

Instead of re-running full conversion (which is slow), we create a minimal
dataset that preserves the structure but with reduced size for testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

# Load original NPZ data
npz_path = r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\380_US_New_York.npz'
data = np.load(npz_path, allow_pickle=True)
grid = data['grid'].item()
traj = data['traj']

print(f"Original NPZ:")
print(f"  Grid nodes: {len(grid)}")
print(f"  Trajectories: {len(traj)}")

# Load network JSON
network_json_path = r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\worldmove_380_NY_network.json'
with open(network_json_path, 'r') as f:
    network = json.load(f)

grid_to_osm = network['grid_mapping']
osm_nodes = {n['id']: n for n in network['nodes']}

# Create minimal dataset with every Nth trajectory for speed
STEP = 50  # Use every 50th trajectory to create minimal dataset
base_date = datetime(2024, 10, 21, 6, 0, 0)
num_unique_users = 100  # Fewer users for testing
num_days = 14

rows = []
skipped = 0

print(f"\nCreating minimal test dataset (every {STEP}th trajectory)...")

for trip_idx, trajectory in enumerate(traj[::STEP]):
    # Filter zeros
    valid_nodes = trajectory[trajectory > 0]
    if len(valid_nodes) < 2:
        skipped += 1
        continue
    
    # Convert to OSM nodes (simplified - just use first and last)
    try:
        origin_grid = int(valid_nodes[0])
        dest_grid = int(valid_nodes[-1])
        
        if str(origin_grid) not in grid_to_osm or str(dest_grid) not in grid_to_osm:
            skipped += 1
            continue
        
        origin_osm = grid_to_osm[str(origin_grid)]['osm_id']
        dest_osm = grid_to_osm[str(dest_grid)]['osm_id']
        
        # Create path string
        path_str = ','.join(str(n) for n in valid_nodes[:20])  # Limit path length
        
        # Simple distance estimate (km)
        dist_km = len(valid_nodes) * 0.5
        
        # User and date assignment
        actual_trip_idx = trip_idx * STEP
        user_num = (actual_trip_idx % num_unique_users)
        user_id = f"user_{user_num:05d}"
        role = 'passenger' if (user_num % 10) != 0 else 'driver'
        
        day_offset = (actual_trip_idx // (len(traj) // num_days)) % num_days
        hour_offset = (actual_trip_idx * 17) % 18 + 6
        minute_offset = (actual_trip_idx * 23) % 60
        
        trip_date = base_date + timedelta(days=day_offset, hours=hour_offset, minutes=minute_offset)
        arrival_date = trip_date + timedelta(minutes=dist_km * 2)  # Assume 30 km/h
        
        row = {
            'user_id': user_id,
            'date': trip_date.strftime('%Y-%m-%d'),
            'role': role,
            'orig_trip_id': 1000000 + actual_trip_idx,
            'agent_id': f"B{(user_num % 2000):05d}",
            'origin_node': origin_grid,
            'destination_node': dest_grid,
            'q_path': path_str,
            'q_km': round(dist_km, 3),
            'capacity': 3 if role == 'driver' else 0,
            'occupied_seats': 0,
            'detour_tolerance': 1.25,
            'detour_ratio': 1.1 if role == 'driver' else np.nan,
            'tau': np.nan if role == 'driver' else 1.5,
            'earliest_departure': trip_date.strftime('%Y-%m-%d %H:%M:%S'),
            'latest_arrival': arrival_date.strftime('%Y-%m-%d %H:%M:%S'),
            'trip_distance_km': round(dist_km, 3),
            'trip_duration_min': round(dist_km * 2, 2),
            'average_speed_kmh': 30.0
        }
        
        rows.append(row)
        
    except Exception as e:
        skipped += 1
        continue

# Save to CSV
output_csv = r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\worldmove_380_NY_test.csv'
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print(f"\n✓ Minimal test dataset created!")
print(f"  Rows: {len(df)}")
print(f"  File: {output_csv}")
print(f"  Size: {Path(output_csv).stat().st_size / (1024*1024):.1f} MB")

print(f"\nDataset Temporal Structure:")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Unique dates: {df['date'].nunique()}")
print(f"  Unique users: {df['user_id'].nunique()}")
print(f"  Avg trips/user: {len(df) / df['user_id'].nunique():.1f}")

print(f"\nLTM/STM Split (First 7 days):")
first_7_days = df[df['date'] <= sorted(df['date'].unique())[6]]
print(f"  LTM trajectories: {len(first_7_days)}")
print(f"  LTM users: {first_7_days['user_id'].nunique()}")

print(f"\nSample data:")
print(df.head(3).to_string())
