"""
Add synthetic 'date' column to existing worldmove_380_NY.csv for LTM/STM training.
This allows splitting data without re-running the full conversion pipeline.
"""

import pandas as pd
from datetime import datetime, timedelta
import sys

csv_path = r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\worldmove_380_NY.csv'

print(f"Loading CSV from {csv_path}...")
df = pd.read_csv(csv_path)

print(f"Current shape: {df.shape}")
print(f"Current columns: {list(df.columns)}")

if 'date' in df.columns:
    print("✓ 'date' column already exists!")
    sys.exit(0)

print("\nGenerating synthetic dates for LTM/STM training...")

# Generate dates matching the same distribution as convert_worldmove_to_csv.py
base_date = datetime(2024, 10, 21, 6, 0, 0)
num_unique_users = 500
num_days = 14

dates = []
for trip_idx in range(len(df)):
    day_offset = (trip_idx // (len(df) // num_days)) % num_days
    trip_date = base_date + timedelta(days=day_offset)
    dates.append(trip_date.strftime('%Y-%m-%d'))

# Insert date column at position 1 (after user_id)
df.insert(1, 'date', dates)

print(f"\nNew shape: {df.shape}")
print(f"New columns: {list(df.columns)}")

print(f"\nTemporal Statistics:")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Unique dates: {df['date'].nunique()}")
print(f"  Unique users: {df['user_id'].nunique()}")
print(f"  Avg trips/user: {len(df) / df['user_id'].nunique():.1f}")
print(f"  Avg trips/date: {len(df) / df['date'].nunique():.1f}")

print(f"\nLTM/STM Data Split:")
first_7_days = df[df['date'] <= df['date'].nsmallest(7, keep='last').max()]
print(f"  LTM (first 7 days, all users): {len(first_7_days)} trips")
print(f"    - Users in first 7 days: {first_7_days['user_id'].nunique()}")

print(f"\n✓ Saving updated CSV...")
df.to_csv(csv_path, index=False)

print(f"✓ Successfully added 'date' column to {csv_path}")
print(f"\nFirst few rows with dates:")
print(df.head(3).to_string())
