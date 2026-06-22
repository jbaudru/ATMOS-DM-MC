"""
Fetch a complete road network for New York City from OpenStreetMap using osmnx
and map WorldMove grid nodes to the actual network
"""

import numpy as np
import json
import math
from pathlib import Path
from datetime import datetime

try:
    import osmnx as ox
except ImportError:
    print("osmnx not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'osmnx'])
    import osmnx as ox

def load_grid_data(npz_path):
    """Load the WorldMove grid coordinates"""
    data = np.load(npz_path, allow_pickle=True)
    grid = data['grid'].item()
    
    # Convert to list of (node_id, lon, lat)
    grid_nodes = []
    for node_id_str, coords in grid.items():
        node_id = int(node_id_str)
        lon, lat = coords
        grid_nodes.append((node_id, lon, lat))
    
    return grid_nodes

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in meters"""
    R = 6371000  # Earth's radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def fetch_osm_network():
    """Fetch the road network from OpenStreetMap for NYC"""
    print("Fetching NYC road network from OpenStreetMap...")
    print("This may take several minutes on first run...")
    
    # NYC bounding box (Manhattan + surrounding areas)
    # North, South, East, West
    north, south, east, west = 40.92, 40.70, -73.90, -74.02
    
    try:
        # Download the network for the specified area
        # Use proper osmnx API - no tags parameter in graph_from_bbox
        G = ox.graph_from_bbox(north, south, east, west, network_type='drive', simplify=True)
        
        print(f"Downloaded network: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G
    
    except Exception as e:
        print(f"Error fetching OSM network: {e}")
        print("Retrying with alternative method...")
        try:
            # Alternative: use place name
            G = ox.graph_from_place('New York City, New York, USA', network_type='drive', simplify=True)
            print(f"Downloaded network: {len(G.nodes)} nodes, {len(G.edges)} edges")
            return G
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("Falling back to simplified local generation...")
            return None

def map_grid_to_osm(grid_nodes, G):
    """Map WorldMove grid nodes to nearest OSM network nodes (optimized with spatial indexing)"""
    print("Mapping grid nodes to OSM network...")
    
    try:
        from scipy.spatial import cKDTree
        use_kdtree = True
    except ImportError:
        print("scipy not available, using slower brute-force mapping...")
        use_kdtree = False
    
    # Build OSM nodes list
    osm_nodes_list = [(node_id, data['x'], data['y']) for node_id, data in G.nodes(data=True)]
    
    if use_kdtree:
        # Extract coordinates for KDTree
        osm_coords = np.array([[lon, lat] for _, lon, lat in osm_nodes_list])
        tree = cKDTree(osm_coords)
        
        # Map each grid node to nearest OSM node using KDTree
        grid_to_osm = {}
        grid_coords = np.array([[lon, lat] for _, lon, lat in grid_nodes])
        
        distances, indices = tree.query(grid_coords)
        
        for i, (grid_id, _, _) in enumerate(grid_nodes):
            osm_id = osm_nodes_list[indices[i]][0]
            grid_to_osm[grid_id] = {
                'osm_id': osm_id,
                'distance_m': distances[i] * 111320  # Convert degrees to meters (approx)
            }
    else:
        # Brute force mapping
        grid_to_osm = {}
        for grid_id, grid_lon, grid_lat in grid_nodes:
            min_dist = float('inf')
            nearest_osm_id = None
            
            for osm_id, osm_lon, osm_lat in osm_nodes_list:
                dist = haversine_distance(grid_lat, grid_lon, osm_lat, osm_lon)
                if dist < min_dist:
                    min_dist = dist
                    nearest_osm_id = osm_id
            
            grid_to_osm[grid_id] = {
                'osm_id': nearest_osm_id,
                'distance_m': min_dist
            }
    
    # Print statistics
    distances = [v['distance_m'] for v in grid_to_osm.values()]
    print(f"Mapping complete:")
    print(f"  Average mapping distance: {np.mean(distances):.2f} m")
    print(f"  Max mapping distance: {np.max(distances):.2f} m")
    print(f"  Min mapping distance: {np.min(distances):.2f} m")
    
    return grid_to_osm

def create_fallback_network(grid_nodes):
    """Create a fallback network from grid nodes when OSM fetch fails"""
    print("Creating fallback network from WorldMove grid nodes...")
    
    # Create nodes with indices
    nodes = []
    grid_to_idx = {}  # Map grid node IDs to indices
    
    for idx, (grid_id, grid_lon, grid_lat) in enumerate(grid_nodes):
        grid_to_idx[grid_id] = idx
        node = {
            "id": idx,
            "osm_id": grid_id,
            "x": grid_lon,
            "y": grid_lat,
            "tags": {}
        }
        nodes.append(node)
    
    print(f"Created {len(nodes)} nodes from grid")
    
    # Create edges connecting nearby nodes
    edges = []
    edge_id = 0
    
    for i, (grid_id1, lon1, lat1) in enumerate(grid_nodes):
        for j, (grid_id2, lon2, lat2) in enumerate(grid_nodes):
            if i < j:
                # Calculate distance
                dx = abs(lon2 - lon1)
                dy = abs(lat2 - lat1)
                dist_km = math.sqrt((dx * 111.32) ** 2 + (dy * 111.32 * math.cos(math.radians(lat1))) ** 2)
                
                # Connect if close (within ~2km)
                if dist_km < 2 and dist_km > 0.01:
                    # Bidirectional edges
                    edge_id += 1
                    edges.append({
                        "id": edge_id,
                        "source": i,
                        "target": j,
                        "length": dist_km * 1000,  # Convert to meters
                        "speed_kph": 30,
                        "tags": {}
                    })
                    
                    edge_id += 1
                    edges.append({
                        "id": edge_id,
                        "source": j,
                        "target": i,
                        "length": dist_km * 1000,
                        "speed_kph": 30,
                        "tags": {}
                    })
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} nodes, {len(edges)} edges so far...")
    
    print(f"Created {len(edges)} edges")
    
    # Create grid mapping (1:1 mapping)
    grid_mapping = {}
    for grid_id, idx in grid_to_idx.items():
        grid_mapping[str(grid_id)] = {
            'osm_id': grid_id,
            'distance_m': 0  # Perfect mapping
        }
    
    # Create network structure
    network = {
        "metadata": {
            "dataset": "WorldMove Grid-based Network",
            "created": datetime.now().isoformat(),
            "description": "Network generated from WorldMove grid coordinates (fallback)",
            "location": "New York City, USA",
            "source": "grid",
            "projection": "EPSG:4326",
            "grid_mapping": "WorldMove 380_US_New_York"
        },
        "nodes": nodes,
        "edges": edges,
        "grid_mapping": grid_mapping
    }
    
    return network
    """Map WorldMove grid nodes to nearest OSM network nodes"""
    print("Mapping grid nodes to OSM network...")
    
    # Build OSM nodes list
    osm_nodes = {node_id: (data['x'], data['y']) for node_id, data in G.nodes(data=True)}
    osm_node_list = list(osm_nodes.items())
    
    # Map each grid node to nearest OSM node
    grid_to_osm = {}
    
    for grid_id, grid_lon, grid_lat in grid_nodes:
        min_dist = float('inf')
        nearest_osm_id = None
        
        for osm_id, (osm_lon, osm_lat) in osm_node_list:
            dist = haversine_distance(grid_lat, grid_lon, osm_lat, osm_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_osm_id = osm_id
        
        grid_to_osm[grid_id] = {
            'osm_id': nearest_osm_id,
            'distance_m': min_dist
        }
    
    # Print statistics
    distances = [v['distance_m'] for v in grid_to_osm.values()]
    print(f"Mapping complete:")
    print(f"  Average mapping distance: {np.mean(distances):.2f} m")
    print(f"  Max mapping distance: {np.max(distances):.2f} m")
    print(f"  Min mapping distance: {np.min(distances):.2f} m")
    
    return grid_to_osm

def create_network_json(G, grid_to_osm, output_json):
    """Create JSON network file from OSM graph and grid mapping"""
    print("Creating network JSON...")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kwargs):
            return it

    # Create nodes list
    nodes = []
    osm_id_to_idx = {}  # Map OSM node IDs to indices
    
    for idx, (osm_id, data) in enumerate(
        tqdm(G.nodes(data=True), total=G.number_of_nodes(),
             desc="  nodes", unit="node")
    ):
        osm_id_to_idx[osm_id] = idx
        node = {
            "id": idx,  # Use index as ID
            "osm_id": osm_id,  # Store original OSM ID
            "x": data.get('x', 0.0),
            "y": data.get('y', 0.0),
            "tags": data.get('tags', {})
        }
        nodes.append(node)
    
    print(f"Created {len(nodes)} nodes")
    
    # Create edges list
    edges = []
    edge_id = 0
    
    for u, v, key, data in tqdm(
        G.edges(keys=True, data=True), total=G.number_of_edges(),
        desc="  edges", unit="edge"
    ):
        if u in osm_id_to_idx and v in osm_id_to_idx:
            u_idx = osm_id_to_idx[u]
            v_idx = osm_id_to_idx[v]
            
            # Calculate distance if available
            length = data.get('length', 0)
            
            edge = {
                "id": edge_id,
                "source": u_idx,
                "target": v_idx,
                "length": float(length),
                "speed_kph": 30,  # Default speed
                "tags": data.get('tags', {})
            }
            geometry = data.get('geometry')
            if geometry is not None and hasattr(geometry, 'coords'):
                edge["geometry"] = [[float(x), float(y)] for x, y in geometry.coords]
            edges.append(edge)
            edge_id += 1
    
    print(f"Created {len(edges)} edges")
    
    # Create network structure with grid mapping
    network = {
        "metadata": {
            "dataset": "OpenStreetMap - New York City",
            "created": datetime.now().isoformat(),
            "description": "Complete road network from OpenStreetMap",
            "location": "New York City, USA",
            "source": "osmnx",
            "projection": "EPSG:4326",
            "grid_mapping": "WorldMove 380_US_New_York"
        },
        "nodes": nodes,
        "edges": edges,
        "grid_mapping": grid_to_osm
    }
    
    # Write to file
    with open(output_json, 'w') as f:
        json.dump(network, f, indent=2)
    
    print(f"Network saved to {output_json}")
    
    return network

def main():
    data_dir = Path(r'c:\Users\julien\OneDrive\Documents\Github\GraphIDyOMo\data')
    npz_file = data_dir / '380_US_New_York.npz'
    output_json = data_dir / 'worldmove_380_NY_network.json'
    
    # Load grid data
    print("Loading WorldMove grid data...")
    grid_nodes = load_grid_data(npz_file)
    print(f"Loaded {len(grid_nodes)} grid nodes")
    
    # Try to fetch OSM network
    G = fetch_osm_network()
    
    if G is None:
        print("\nUsing fallback network from grid nodes...")
        network = create_fallback_network(grid_nodes)
    else:
        # Map grid nodes to OSM network
        print("\nMapping grid nodes to OSM network...")
        grid_to_osm = map_grid_to_osm(grid_nodes, G)
        
        # Create network JSON
        print("Creating network JSON from OSM...")
        network = create_network_json(G, grid_to_osm, output_json)
    
    # Save network to file if not already saved
    if G is None or not output_json.exists():
        with open(output_json, 'w') as f:
            json.dump(network, f, indent=2)
        print(f"Network saved to {output_json}")
    
    print("\n✓ Network creation complete!")
    print(f"  Network file: {output_json}")
    print(f"  Total nodes: {len(network['nodes'])}")
    print(f"  Total edges: {len(network['edges'])}")

if __name__ == '__main__':
    main()
