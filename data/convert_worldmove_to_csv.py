"""
Convert WorldMove 380_US_New_York.npz dataset to CSV format.
Uses the complete OpenStreetMap road network and maps grid nodes to OSM nodes.
"""

import numpy as np
import pandas as pd
import json
import heapq
from datetime import datetime, timedelta
import math
from pathlib import Path

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km"""
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def erase_loops(path):
    """Remove loops / back-and-forth so the path is a simple route start->end.

    WorldMove trajectories oscillate between coarse grid cells (A->B->A, or long
    palindromic detours), which after snapping/routing show up as dead-end spurs
    and doubled-back segments. Loop-erasure walks the node sequence and, whenever
    a node is revisited, discards the entire loop traversed since the previous
    visit. The result is a simple path (no repeated node) from the original first
    node to the original last node, and crucially every consecutive pair is still
    a genuine network neighbour (the cut reconnects at the *same* node, whose
    successor in the original path was already an adjacent node).
    """
    result = []
    position = {}  # node -> index in result
    for node in path:
        seen_at = position.get(node)
        if seen_at is None:
            position[node] = len(result)
            result.append(node)
        else:
            # Drop everything traversed after the first visit to ``node``.
            for dropped in result[seen_at + 1:]:
                position.pop(dropped, None)
            del result[seen_at + 1:]
    return result


# ---------------------------------------------------------------------------
# Road-network routing
# ---------------------------------------------------------------------------
# WorldMove trajectories are sequences of *coarse grid* cells (~1 km apart),
# each snapped to the nearest OSM node.  Consecutive snapped nodes are therefore
# NOT connected by a road edge.  To produce trajectories whose consecutive nodes
# are genuine network neighbours, we route each consecutive pair along the road
# graph (shortest path) and stitch the dense node sequences together.
#
# Speed: the snapped points come from only a few thousand distinct grid cells,
# so there are very few distinct *source* nodes.  Rather than running one A*
# search per (src, dst) pair, we run a single Dijkstra per distinct source that
# settles all of that source's destinations at once (early-terminating once
# they are all reached).  Adjacency is stored as plain lists indexed by node id
# (network node ids are dense 0..N-1), which is far faster than dict lookups.

def build_routing_graph(network):
    """Return (adj, edge_len, xs, ys, N) in network-node-index space.

    adj[u]          -> list of (v, length_m)   (undirected: both directions)
    edge_len[(u,v)] -> length in metres (min for parallel edges)
    xs[u], ys[u]    -> (lon, lat) of node u
    N               -> number of nodes
    """
    N = len(network['nodes'])
    xs = [0.0] * N
    ys = [0.0] * N
    for node in network['nodes']:
        i = node['id']
        xs[i] = float(node.get('x', 0.0))
        ys[i] = float(node.get('y', 0.0))

    adj = [[] for _ in range(N)]
    edge_len = {}

    def _add(u, v, L):
        adj[u].append((v, L))
        cur = edge_len.get((u, v))
        if cur is None or L < cur:
            edge_len[(u, v)] = L

    for edge in network['edges']:
        u = edge['source']
        v = edge['target']
        if u == v or u >= N or v >= N:
            continue
        L = float(edge.get('length', 0.0) or 0.0)
        if L <= 0.0:
            latr = math.radians((ys[u] + ys[v]) / 2.0)
            dx = (xs[u] - xs[v]) * 111320.0 * math.cos(latr)
            dy = (ys[u] - ys[v]) * 111320.0
            L = math.hypot(dx, dy) or 1.0
        # undirected so a one-way dead-end near a snapped endpoint does not make
        # it unreachable; consecutive nodes still always share a real edge.
        _add(u, v, L)
        _add(v, u, L)
    return adj, edge_len, xs, ys, N


def precompute_routes(adj, N, pairs, progress=True):
    """Shortest path for every needed (src, dst) pair via per-source Dijkstra.

    ``pairs`` is an iterable of (src, dst) tuples.  Returns a dict
    ``{(src, dst): [src, ..., dst]}`` (or ``None`` when dst is unreachable).
    One Dijkstra is run per distinct source and stops as soon as all of that
    source's destinations have been settled.
    """
    from collections import defaultdict

    by_src = defaultdict(set)
    for s, d in pairs:
        by_src[s].add(d)

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    src_iter = by_src.items()
    if progress and _tqdm is not None:
        src_iter = _tqdm(src_iter, total=len(by_src),
                         desc="  Routing (per source)", unit="src")

    INF = float('inf')
    cache = {}
    for src, dsts in src_iter:
        if src in dsts:
            cache[(src, src)] = [src]
        remaining = set(dsts)
        remaining.discard(src)

        dist = [INF] * N
        prev = [-1] * N
        dist[src] = 0.0
        heap = [(0.0, src)]
        while heap and remaining:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            remaining.discard(u)
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        for dst in dsts:
            if dst == src:
                continue
            if dist[dst] == INF:
                cache[(src, dst)] = None
            else:
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                path.reverse()
                cache[(src, dst)] = path
    return cache

def load_npz_data(npz_path):
    """Load the NPZ file"""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'grid': data['grid'].item(),  # Node coordinates - extract from 0-d array
        'traj': data['traj'],  # Trajectories
        'poi': data['poi'],
        'pop': data['pop']
    }

def load_osm_network(network_json):
    """Load the OSM network JSON with grid mapping"""
    with open(network_json, 'r') as f:
        network = json.load(f)
    
    return network

def convert_trajectories_to_csv(data, network, output_csv, route_on_network=True,
                                clean_paths=True):
    """Convert NPZ trajectories to CSV format using OSM network

    NOTE: The original WorldMove 380_US_New_York.npz dataset contains only:
      - grid: Node coordinates
      - traj: Node trajectories (node IDs only, no timestamps)
      - poi, pop: Geographic metadata

    Since no temporal data exists in the original NPZ, we generate realistic
    synthetic dates that span 14 days to support LTM/STM training structure:
      - LTM (Long-Term): All users across 7 days (population model)
      - STM (Short-Term): Individual users per day (personal model)

    When ``route_on_network`` is True (default) each consecutive pair of snapped
    OSM nodes is connected via the road-network shortest path, so the resulting
    ``q_path`` is a sequence of genuine network neighbours instead of ~1 km grid
    jumps.

    When ``clean_paths`` is True (default) the stitched path is then loop-erased
    so the stored ``q_path`` is a simple start->end route with no back-and-forth
    oscillations or dead-end spurs (artifacts of the coarse-grid trajectories).
    """

    geo = data['grid']
    traj = data['traj']
    grid_to_osm = network['grid_mapping']

    # Create OSM nodes dict for coordinate lookup
    osm_nodes = {node['id']: node for node in network['nodes']}

    # Precompute osm_id -> node index ONCE. Without this, every node of every
    # trajectory triggers a full linear scan over all network nodes (which is
    # ~128k for a real OSM graph), making the conversion O(trips x path x nodes).
    osm_id_to_idx = {
        node['osm_id']: node['id']
        for node in network['nodes']
        if 'osm_id' in node
    }

    # Routing graph (index space) used to fill in road-adjacent nodes between
    # consecutive snapped trajectory points (and for distance / positions).
    print("  Building routing graph from network edges...")
    adj, edge_len_idx, node_xs, node_ys, n_nodes = build_routing_graph(network)
    route_cache = {}
    n_unroutable_pairs = 0
    n_routed_pairs = 0

    # Edge length lookup keyed by (source_index, target_osm_id) for O(1) access.
    edge_len_by_src_target = {}
    for edge in network['edges']:
        source_idx = edge['source']
        target_node = osm_nodes.get(edge['target'])
        if target_node is None:
            continue
        target_osm = target_node.get('osm_id')
        edge_len_by_src_target[(source_idx, target_osm)] = edge['length']

    # Optional progress bar.
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # Generate base date - use extended date range for 7+ days of data
    base_date = datetime(2024, 10, 21, 6, 0, 0)  # Start 7 days earlier

    rows = []
    skipped_trajectories = 0

    # Number of unique users and days for population model
    num_unique_users = 500  # Create 500 distinct users for LTM population model
    num_days = 14  # Span 14 days (2 weeks) for rich temporal distribution

    def _snapped_idx_for(trajectory):
        """Decode one raw trajectory to a deduped sequence of network indices."""
        valid_nodes = trajectory[trajectory > 0]
        # Collapse consecutive duplicate grid nodes (stationary waits / self-loops).
        deduped = []
        for grid_node in valid_nodes:
            g = int(grid_node)
            if not deduped or deduped[-1] != g:
                deduped.append(g)
        if len(deduped) < 2:
            return None
        # Grid nodes -> OSM ids (skip repeats: a 1:1 grid->OSM map can collide).
        osm_path = []
        for g in deduped:
            s = str(g)
            if s not in grid_to_osm:
                return None
            o = grid_to_osm[s]['osm_id']
            if osm_path and osm_path[-1] == o:
                continue
            osm_path.append(o)
        if len(osm_path) < 2:
            return None
        snapped = [osm_id_to_idx[o] for o in osm_path if o in osm_id_to_idx]
        return snapped if len(snapped) >= 2 else None

    # --- Pass 1: decode every trajectory + collect the unique node pairs to route
    decode_iter = enumerate(traj)
    if tqdm is not None:
        decode_iter = tqdm(decode_iter, total=len(traj),
                           desc="Decoding trajectories", unit="traj")
    snapped_paths = []          # list of (trip_idx, snapped_idx)
    pair_set = set()
    for trip_idx, trajectory in decode_iter:
        snapped_idx = _snapped_idx_for(trajectory)
        if snapped_idx is None:
            skipped_trajectories += 1
            continue
        snapped_paths.append((trip_idx, snapped_idx))
        if route_on_network:
            for a, b in zip(snapped_idx, snapped_idx[1:]):
                if a != b:
                    pair_set.add((a, b))

    if route_on_network:
        print(f"  Precomputing routes for {len(pair_set)} unique node pairs "
              f"(per-source Dijkstra)...")
        route_cache = precompute_routes(adj, n_nodes, pair_set)

    # --- Pass 2: stitch routed paths and build trip records
    build_iter = snapped_paths
    if tqdm is not None:
        build_iter = tqdm(snapped_paths, total=len(snapped_paths),
                          desc="Building trip records", unit="traj")

    for trip_idx, snapped_idx in build_iter:
        # Densify: replace each consecutive snapped pair with its road-network
        # shortest path so the final path is made of genuine network neighbours.
        if route_on_network:
            dense = [snapped_idx[0]]
            for a, b in zip(snapped_idx, snapped_idx[1:]):
                seg = [a] if a == b else route_cache.get((a, b))
                if seg is None or len(seg) < 2:
                    dense.append(b)          # unreachable: keep the direct jump
                    n_unroutable_pairs += 1
                else:
                    dense.extend(seg[1:])
                    n_routed_pairs += 1
            # collapse any accidental consecutive duplicates
            idx_path = [dense[0]]
            for v in dense[1:]:
                if v != idx_path[-1]:
                    idx_path.append(v)
        else:
            idx_path = snapped_idx

        # Remove oscillations / dead-ends so the stored q_path is a clean simple
        # route from origin to destination (no back-and-forth artifacts).
        if clean_paths and len(idx_path) > 2:
            cleaned = erase_loops(idx_path)
            if len(cleaned) >= 2:
                idx_path = cleaned
            else:
                # Degenerate round-trip that collapses to a single point.
                skipped_trajectories += 1
                continue

        if len(idx_path) < 2:
            skipped_trajectories += 1
            continue

        origin_idx = idx_path[0]
        dest_idx = idx_path[-1]

        # Build path string (network node indices).
        path_str = ','.join(str(i) for i in idx_path)

        # Total distance along the (now road-adjacent) path.
        total_distance = 0.0
        if route_on_network:
            for u, v in zip(idx_path, idx_path[1:]):
                total_distance += edge_len_idx.get((u, v), 0.0)
        else:
            # No routing: approximate with straight-line metres between nodes.
            for u, v in zip(idx_path, idx_path[1:]):
                latr = math.radians((node_ys[u] + node_ys[v]) / 2.0)
                dx = (node_xs[u] - node_xs[v]) * 111320.0 * math.cos(latr)
                dy = (node_ys[u] - node_ys[v]) * 111320.0
                total_distance += math.hypot(dx, dy)
        
        # Convert to km
        total_distance_km = total_distance / 1000 if total_distance > 0 else 0
        
        # Estimate trip duration based on distance (assuming ~30 km/h average)
        avg_speed = 30
        duration_minutes = (total_distance_km / avg_speed) * 60 if total_distance_km > 0 else 10
        
        # Assign MEANINGFUL user_id: create consistent user distribution across days
        # Use modulo to create ~500 users with repeated trajectories across different days
        user_num = (trip_idx % num_unique_users)
        role = 'passenger' if (user_num % 10) != 0 else 'driver'
        user_id = f"user_{user_num:05d}"
        
        # Generate timestamps: distribute across 14 days with temporal structure
        # Each user has multiple trips on different days
        day_offset = (trip_idx // (len(traj) // num_days)) % num_days  # Spread across days
        hour_offset = (trip_idx * 17) % 18 + 6  # 6am - 11pm trips
        minute_offset = (trip_idx * 23) % 60
        
        trip_date = base_date + timedelta(days=day_offset, hours=hour_offset, minutes=minute_offset)
        arrival_date = trip_date + timedelta(minutes=duration_minutes)
        
        row = {
            'user_id': user_id,
            'date': trip_date.strftime('%Y-%m-%d'),  # NEW: Explicit date column for LTM/STM split
            'role': role,
            'orig_trip_id': 1000000 + trip_idx,
            'agent_id': f"B{(user_num % 2000):05d}",
            'origin_node': origin_idx,
            'destination_node': dest_idx,
            'q_path': path_str,
            'q_km': round(total_distance_km, 3),
            'capacity': 3 if role == 'driver' else 0,
            'occupied_seats': 0,
            'detour_tolerance': 1.25,
            'detour_ratio': '' if role == 'passenger' else f"{round(1.1 + (user_num % 100) / 1000, 4)}",
            'tau': 1.5 if role == 'passenger' else '',
            'earliest_departure': trip_date.strftime('%Y-%m-%d %H:%M:%S'),
            'latest_arrival': arrival_date.strftime('%Y-%m-%d %H:%M:%S'),
            'trip_distance_km': round(total_distance_km, 3),
            'trip_duration_min': round(duration_minutes, 2),
            'average_speed_kmh': round((total_distance_km / (duration_minutes / 60)) if duration_minutes > 0 else 0, 2)
        }
        
        rows.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} valid rows to {output_csv}")
    print(f"Skipped {skipped_trajectories} invalid trajectories")
    if route_on_network:
        total_pairs = n_routed_pairs + n_unroutable_pairs
        pct = (100.0 * n_unroutable_pairs / total_pairs) if total_pairs else 0.0
        print(f"Routing: {n_routed_pairs} pairs routed, "
              f"{n_unroutable_pairs} unroutable ({pct:.2f}%), "
              f"{len(route_cache)} unique pairs cached")
    return df

def main():
    data_dir = Path(__file__).parent.resolve()
    npz_file = data_dir / '380_US_New_York.npz'
    network_json = data_dir / 'worldmove_380_US_network.json'
    output_csv = data_dir / 'worldmove_380_US.csv'
    
    print("Loading NPZ data...")
    data = load_npz_data(npz_file)
    
    print(f"Loaded {len(data['grid'])} nodes")
    print(f"Loaded {len(data['traj'])} trajectories")
    
    # Load OSM network
    print("\nLoading OSM network...")
    network = load_osm_network(network_json)
    print(f"Network has {len(network['nodes'])} nodes and {len(network['edges'])} edges")
    print(f"Grid mapping: {len(network['grid_mapping'])} nodes mapped")
    
    # Convert to CSV
    print(f"\nConverting to CSV format...")
    df = convert_trajectories_to_csv(data, network, output_csv)
    
    print(f"\nConversion complete!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nStatistics:")
    print(f"  Passengers: {len(df[df['role'] == 'passenger'])}")
    print(f"  Drivers: {len(df[df['role'] == 'driver'])}")
    print(f"  Average distance: {df['q_km'].mean():.2f} km")
    print(f"  Average speed: {df['average_speed_kmh'].mean():.2f} km/h")
    print(f"  Min/Max distance: {df['q_km'].min():.2f} / {df['q_km'].max():.2f} km")
    
    print(f"\n" + "="*70)
    print(f"📊 DATASET TEMPORAL INFORMATION")
    print(f"="*70)
    print(f"Original NPZ Dataset: WorldMove 380_US_New_York")
    print(f"  ❌ NO timestamp/date information in original NPZ")
    print(f"  ℹ️  Original contains: grid coords, trajectories (node IDs), POI/population")
    print(f"\n🔧 Synthetic Temporal Distribution (for LTM/STM training):")
    print(f"  📅 Time Range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ⏰ Time Distribution: 6am-11pm (realistic ride-sharing hours)")
    print(f"  👥 Unique Users: {df['user_id'].nunique()} (synthetic, for modeling diversity)")
    print(f"\n🧠 LTM/STM Structure Enabled:")
    print(f"  LTM (Population Model): All {df['user_id'].nunique()} users × 7 days")
    print(f"  STM (Individual Model): Single user × 1 day adaptation")
    print(f"="*70)

if __name__ == '__main__':
    main()
