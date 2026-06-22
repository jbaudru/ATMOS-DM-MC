"""
Test script to verify WorldMove data compatibility with tpls_rideshare.py
"""

import pandas as pd
import json
import sys
from pathlib import Path

def test_csv_format():
    """Test CSV file format and compatibility"""
    print("=" * 60)
    print("Testing CSV Format Compatibility")
    print("=" * 60)
    
    csv_file = Path(r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\worldmove_380_NY.csv')
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ CSV loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['user_id', 'role', 'origin_node', 'destination_node', 'q_path', 'q_km']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"✗ Missing columns: {missing}")
            return False
        print(f"✓ All required columns present")
        
        # Check data integrity
        print(f"\n  Data Statistics:")
        print(f"  - Passengers: {len(df[df['role'] == 'passenger'])}")
        print(f"  - Drivers: {len(df[df['role'] == 'driver'])}")
        print(f"  - Average distance: {df['q_km'].mean():.2f} km")
        print(f"  - Distance range: {df['q_km'].min():.2f} - {df['q_km'].max():.2f} km")
        
        # Check path format
        sample_path = df['q_path'].iloc[0]
        if isinstance(sample_path, str):
            nodes = [int(n) for n in sample_path.split(',')]
            print(f"\n✓ Path format valid (sample: {len(nodes)} nodes)")
        else:
            print(f"✗ Path format invalid")
            return False
        
        # Check for missing values in key columns (detour_ratio and tau can be empty by design)
        null_in_keys = df[['user_id', 'role', 'origin_node', 'destination_node', 'q_path']].isnull().sum()
        if null_in_keys.sum() > 0:
            print(f"\n✗ Found null values in key columns:")
            print(null_in_keys[null_in_keys > 0])
            return False
        
        # Show expected null values (drivers don't have tau, passengers don't have detour_ratio)
        print(f"\n✓ No null values in key columns")
        print(f"  Note: Empty strings in detour_ratio (drivers) and tau (passengers) by design")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return False

def test_network_format():
    """Test road network JSON format and compatibility"""
    print("\n" + "=" * 60)
    print("Testing Road Network Format Compatibility")
    print("=" * 60)
    
    network_file = Path(r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\worldmove_380_NY_network.json')
    
    try:
        with open(network_file, 'r') as f:
            network = json.load(f)
        print(f"✓ Network JSON loaded successfully")
        
        # Check structure
        required_keys = ['nodes', 'edges']
        missing = [k for k in required_keys if k not in network]
        if missing:
            print(f"✗ Missing keys: {missing}")
            return False
        print(f"✓ Required keys present")
        
        # Check nodes
        nodes = network['nodes']
        print(f"\n  Nodes:")
        print(f"  - Count: {len(nodes)}")
        
        sample_node = nodes[0]
        required_node_fields = ['id', 'x', 'y']
        missing_fields = [f for f in required_node_fields if f not in sample_node]
        if missing_fields:
            print(f"  ✗ Missing node fields: {missing_fields}")
            return False
        print(f"  ✓ Sample node: {sample_node}")
        
        # Check edges
        edges = network['edges']
        print(f"\n  Edges:")
        print(f"  - Count: {len(edges)}")
        
        sample_edge = edges[0]
        required_edge_fields = ['source', 'target', 'length']
        missing_fields = [f for f in required_edge_fields if f not in sample_edge]
        if missing_fields:
            print(f"  ✗ Missing edge fields: {missing_fields}")
            return False
        print(f"  ✓ Sample edge: source={sample_edge['source']}, target={sample_edge['target']}, length={sample_edge['length']}km")
        
        # Validate edge connectivity
        node_ids = set(n['id'] for n in nodes)
        invalid_edges = 0
        for edge in edges[:100]:  # Check first 100
            if edge['source'] not in node_ids or edge['target'] not in node_ids:
                invalid_edges += 1
        
        if invalid_edges > 0:
            print(f"\n✗ Found {invalid_edges} edges with invalid node references")
            return False
        print(f"\n✓ Edge references valid (checked first 100 edges)")
        
        # Check coordinate validity
        invalid_coords = sum(1 for n in nodes if not (-180 <= n['x'] <= 180 and -90 <= n['y'] <= 90))
        if invalid_coords > 0:
            print(f"✗ Found {invalid_coords} invalid coordinates")
            return False
        print(f"✓ All coordinates valid")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading network: {e}")
        return False

def test_compatibility():
    """Test if CSV and network can be used together"""
    print("\n" + "=" * 60)
    print("Testing CSV-Network Compatibility")
    print("=" * 60)
    
    try:
        csv_file = Path(r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\worldmove_380_NY.csv')
        network_file = Path(r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data\worldmove_380_NY_network.json')
        
        df = pd.read_csv(csv_file)
        with open(network_file, 'r') as f:
            network = json.load(f)
        
        node_ids = set(n['id'] for n in network['nodes'])
        
        # Check if origin/destination nodes exist in network
        all_origins = df['origin_node'].unique()
        all_destinations = df['destination_node'].unique()
        
        missing_origins = [o for o in all_origins if o not in node_ids]
        missing_destinations = [d for d in all_destinations if d not in node_ids]
        
        if missing_origins:
            print(f"✗ {len(missing_origins)} origin nodes not in network")
            return False
        if missing_destinations:
            print(f"✗ {len(missing_destinations)} destination nodes not in network")
            return False
        
        print(f"✓ All trip endpoints exist in network")
        print(f"  - Origin nodes: {len(all_origins)} unique")
        print(f"  - Destination nodes: {len(all_destinations)} unique")
        print(f"  - Network nodes: {len(node_ids)}")
        
        # Verify path nodes
        sample_paths = df['q_path'].head(10)
        invalid_paths = 0
        for path_str in sample_paths:
            path_nodes = [int(n) for n in path_str.split(',')]
            if any(n not in node_ids for n in path_nodes):
                invalid_paths += 1
        
        if invalid_paths > 0:
            print(f"✗ {invalid_paths} paths have invalid nodes")
            return False
        
        print(f"✓ Sample paths valid (checked {len(sample_paths)} paths)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing compatibility: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "█" * 60)
    print("WorldMove 380_US_New_York Data Compatibility Test")
    print("█" * 60)
    
    results = {
        'CSV Format': test_csv_format(),
        'Network Format': test_network_format(),
        'CSV-Network Compatibility': test_compatibility()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED - Data ready for use!")
        print("\nYou can now use:")
        print("  - worldmove_380_NY.csv")
        print("  - worldmove_380_NY_network.json")
        print("  in your tpls_rideshare.py and visualization code")
    else:
        print("✗ Some tests FAILED - Review output above")
        sys.exit(1)
    print("=" * 60)

if __name__ == '__main__':
    main()
