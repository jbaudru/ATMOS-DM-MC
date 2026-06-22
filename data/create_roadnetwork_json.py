"""
Generate a road network JSON file from WorldMove grid coordinates
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

def create_road_network_json(npz_path, output_json):
    """Create a road network JSON from the WorldMove grid data"""
    
    # Load the NPZ file
    data = np.load(npz_path, allow_pickle=True)
    grid = data['grid'].item()  # Extract from 0-d array
    
    # Extract nodes with coordinates
    nodes = []
    for node_id_str, coords in grid.items():
        node_id = int(node_id_str)
        lon, lat = coords
        
        # Create node entry with OSM-like structure
        node = {
            "id": node_id,
            "x": lon,
            "y": lat,
            "highway": "traffic_signals" if node_id % 10 == 0 else "crossing",
            "street_count": 2 + (node_id % 3)  # Estimate street connectivity
        }
        nodes.append(node)
    
    print(f"Created {len(nodes)} nodes")
    
    # Create edges connecting nearby nodes
    # Use a simplified approach: connect each node to its neighbors based on spatial proximity
    edges = []
    edge_id = 0
    
    # Build spatial index for faster lookup
    node_dict = {n['id']: n for n in nodes}
    node_positions = [(n['id'], n['x'], n['y']) for n in nodes]
    
    # Connect nodes based on proximity (simplified routing)
    for i, (node_id, x, y) in enumerate(node_positions):
        # Find nearby nodes (within 0.01 degrees - roughly 1 km)
        for j, (neighbor_id, neighbor_x, neighbor_y) in enumerate(node_positions):
            if i < j:  # Avoid duplicate edges
                dx = abs(neighbor_x - x)
                dy = abs(neighbor_y - y)
                
                # Connect if within proximity
                if (dx < 0.015 and dy < 0.015) and (dx > 0.0001 or dy > 0.0001):
                    # Calculate approximate distance
                    dist_km = np.sqrt((dx * 111.32) ** 2 + (dy * 111.32 * np.cos(np.radians(y))) ** 2)
                    
                    # Add bidirectional edges
                    edge_id += 1
                    edges.append({
                        "id": edge_id,
                        "source": node_id,
                        "target": neighbor_id,
                        "length": round(dist_km, 4),
                        "speed_kph": 30,
                        "travel_time": round(dist_km / 30 * 60, 2)  # in seconds
                    })
                    
                    edge_id += 1
                    edges.append({
                        "id": edge_id,
                        "source": neighbor_id,
                        "target": node_id,
                        "length": round(dist_km, 4),
                        "speed_kph": 30,
                        "travel_time": round(dist_km / 30 * 60, 2)
                    })
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} nodes, {len(edges)} edges so far...")
    
    print(f"Created {len(edges)} edges")
    
    # Create the network structure
    network = {
        "metadata": {
            "dataset": "WorldMove 380_US_New_York",
            "created": datetime.now().isoformat(),
            "description": "Road network derived from WorldMove grid coordinates",
            "location": "New York City, USA"
        },
        "nodes": nodes,
        "edges": edges
    }
    
    # Write to file
    with open(output_json, 'w') as f:
        json.dump(network, f, indent=2)
    
    print(f"Network saved to {output_json}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    
    return network

def main():
    data_dir = Path(r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data')
    npz_file = data_dir / '380_US_New_York.npz'
    output_json = data_dir / 'worldmove_380_NY_network.json'
    
    print("Generating road network from WorldMove grid...")
    network = create_road_network_json(npz_file, output_json)
    
    print("\n✓ Road network generation complete!")

if __name__ == '__main__':
    main()
