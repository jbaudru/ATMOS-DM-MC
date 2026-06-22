"""
Master setup script to prepare WorldMove datasets for GraphIDyOM training.

Automatically discovers every .npz file in the data directory and, for each one:
  1. Derives the output prefix from the filename  (e.g. 380_US_New_York → worldmove_380_US)
  2. Fetches the road network from OpenStreetMap using the grid's bounding box
  3. Maps WorldMove grid nodes to the OSM network  →  <prefix>_network.json
  4. Converts trajectory data to CSV              →  <prefix>.csv

Already-generated files are skipped unless --force is passed.

Usage
-----
  # All NPZ files
  python prepare_dataset.py

  # Only specific city/ID
  python prepare_dataset.py --filter New_York
  python prepare_dataset.py --filter 431

  # Regenerate even if output already exists
  python prepare_dataset.py --force

Requirements: osmnx (auto-installed if missing)
"""

import sys
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_osmnx():
    try:
        import osmnx as ox
        return ox
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'osmnx'])
        import osmnx as ox
        return ox


def npz_to_prefix(npz_path: Path) -> str:
    """Derive output file prefix from NPZ filename.

    Pattern: {ID}_{CC}_{CityName…}.npz  →  worldmove_{ID}_{CC}

    Examples
    --------
    380_US_New_York.npz  →  worldmove_380_US
    431_DE_Frankfurt.npz →  worldmove_431_DE
    539_SG_Singapore.npz →  worldmove_539_SG
    609_AE_Abu_Dhabi.npz →  worldmove_609_AE
    """
    parts = npz_path.stem.split('_', 2)   # at most 3 parts
    if len(parts) >= 2:
        return f"worldmove_{parts[0]}_{parts[1]}"
    return f"worldmove_{npz_path.stem}"


def _load_grid_nodes(npz_path: Path):
    """Return list of (node_id, lon, lat) from an NPZ grid array."""
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    grid = data['grid'].item()   # {str_id: [lon, lat]}
    return [(int(k), float(v[0]), float(v[1])) for k, v in grid.items()]


def _compute_bbox(grid_nodes, pad=0.05):
    """Compute (north, south, east, west) bounding box with padding."""
    lons = [lon for _, lon, _ in grid_nodes]
    lats = [lat for _, _, lat in grid_nodes]
    lon_pad = max((max(lons) - min(lons)) * pad, 0.05)
    lat_pad = max((max(lats) - min(lats)) * pad, 0.05)
    return (max(lats) + lat_pad,   # north
            min(lats) - lat_pad,   # south
            max(lons) + lon_pad,   # east
            min(lons) - lon_pad)   # west


def _fetch_osm_graph(north, south, east, west):
    """Download OSM drive graph for a bounding box; return None on failure.

    Supports both the legacy osmnx < 2.0 signature
    (graph_from_bbox(north, south, east, west, ...)) and the osmnx >= 2.0
    signature (graph_from_bbox(bbox=(left, bottom, right, top), ...)).
    """
    ox = _ensure_osmnx()
    print(f"    bbox: N={north:.4f} S={south:.4f} E={east:.4f} W={west:.4f}")
    try:
        version = getattr(ox, "__version__", "0")
        major = int(str(version).split(".", 1)[0])
        if major >= 2:
            # osmnx >= 2.0: single bbox tuple (left, bottom, right, top)
            G = ox.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type='drive', simplify=True)
        else:
            # osmnx < 2.0: positional north, south, east, west
            G = ox.graph_from_bbox(north, south, east, west,
                                   network_type='drive', simplify=True)
        print(f"    OSM: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G
    except Exception as e:
        print(f"    OSM fetch failed: {e}")
        return None


def _network_is_osm(network_json: Path) -> bool:
    """Return True if an existing network JSON was built from real OSM data.

    A grid fallback network stores ``"source": "grid"`` in its metadata, whereas
    a genuine OpenStreetMap network stores ``"source": "osm"`` / ``"osmnx"``.
    """
    try:
        with open(network_json, 'r', encoding='utf-8') as fh:
            meta = json.load(fh).get('metadata', {})
    except (OSError, ValueError):
        return False
    return str(meta.get('source', '')).lower() in {'osm', 'osmnx'}


# ---------------------------------------------------------------------------
# Per-NPZ pipeline
# ---------------------------------------------------------------------------

def process_npz(npz_path: Path, data_dir: Path, force: bool = False) -> bool:
    """Build network JSON + CSV for one NPZ file.  Returns True on success."""

    prefix       = npz_to_prefix(npz_path)
    network_json = data_dir / f"{prefix}_network.json"
    output_csv   = data_dir / f"{prefix}.csv"

    print(f"\n{'='*70}")
    print(f"  {npz_path.name}  →  {prefix}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 1. Load grid
    # ------------------------------------------------------------------
    print("[1/3] Loading grid…")
    grid_nodes = _load_grid_nodes(npz_path)
    print(f"      {len(grid_nodes)} grid nodes")

    # ------------------------------------------------------------------
    # 2. Network JSON
    # ------------------------------------------------------------------
    # By default we always want a genuine OSM-format road network alongside the
    # CSV. Regenerate when the file is missing, when --force is given, or when an
    # existing file is only a grid fallback (source != osm).
    network_is_osm = network_json.exists() and _network_is_osm(network_json)
    network_written = False

    if network_json.exists() and network_is_osm and not force:
        print(f"[2/3] OSM network JSON exists – skipping  ({network_json.name})")
    else:
        if network_json.exists() and not network_is_osm and not force:
            print("[2/3] Existing network is a grid fallback – regenerating in OSM format…")
        else:
            print(f"[2/3] Fetching OSM road network…")

        # Add data_dir to path so we can import the helper module
        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))
        from fetch_osm_network import (map_grid_to_osm,
                                       create_fallback_network,
                                       create_network_json)

        north, south, east, west = _compute_bbox(grid_nodes)
        G = _fetch_osm_graph(north, south, east, west)

        if G is None:
            print("      OSM unavailable – building fallback network from grid nodes")
            network = create_fallback_network(grid_nodes)
            with open(network_json, 'w') as fh:
                json.dump(network, fh, indent=2)
            print(f"      Fallback network saved: {network_json.name}")
        else:
            grid_to_osm = map_grid_to_osm(grid_nodes, G)
            create_network_json(G, grid_to_osm, str(network_json))
        network_written = True

    if not network_json.exists():
        print(f"      ERROR: {network_json.name} was not created")
        return False

    # ------------------------------------------------------------------
    # 3. CSV
    # ------------------------------------------------------------------
    # The CSV node IDs are indices into the network's node list, so whenever the
    # network is (re)generated the CSV must be rebuilt to stay in sync.
    if output_csv.exists() and not force and not network_written:
        print(f"[3/3] CSV exists – skipping  ({output_csv.name})")
    else:
        if network_written and output_csv.exists():
            print("[3/3] Network changed – rebuilding CSV to match new node IDs…")
        else:
            print(f"[3/3] Converting trajectories to CSV…")

        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))
        from convert_worldmove_to_csv import (load_npz_data,
                                               load_osm_network,
                                               convert_trajectories_to_csv)

        npz_data = load_npz_data(str(npz_path))
        network  = load_osm_network(str(network_json))
        df = convert_trajectories_to_csv(npz_data, network, str(output_csv))

        if df is None or len(df) == 0:
            print("      ERROR: no rows written")
            return False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    size_json = network_json.stat().st_size / 1024**2
    size_csv  = output_csv.stat().st_size / 1024**2
    print(f"\n  ✓ {network_json.name}  ({size_json:.2f} MB)")
    print(f"  ✓ {output_csv.name}  ({size_csv:.2f} MB)")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    data_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(
        description="Prepare all WorldMove NPZ datasets for GraphIDyOM")
    parser.add_argument('--data-dir', type=Path, default=data_dir,
                        help="Directory containing .npz files (default: script directory)")
    parser.add_argument('--filter', type=str, default=None,
                        help="Only process NPZ files whose name contains this string "
                             "(e.g. 'New_York', '431', 'SG')")
    parser.add_argument('--force', action='store_true',
                        help="Re-generate output files even if they already exist")
    args = parser.parse_args()

    npz_files = sorted(args.data_dir.glob('*.npz'))
    if not npz_files:
        print(f"No .npz files found in {args.data_dir}")
        sys.exit(1)

    if args.filter:
        npz_files = [f for f in npz_files if args.filter in f.name]
        if not npz_files:
            print(f"No .npz files match filter '{args.filter}'")
            sys.exit(1)

    print("=" * 70)
    print("WorldMove Dataset Preparation for GraphIDyOM")
    print("=" * 70)
    print(f"\nFound {len(npz_files)} NPZ file(s):")
    for f in npz_files:
        print(f"  {f.name}  →  {npz_to_prefix(f)}")

    results = {}
    for npz_path in npz_files:
        results[npz_path.name] = process_npz(npz_path, args.data_dir,
                                              force=args.force)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    all_ok = True
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll datasets prepared.  Example training commands:")
        for npz_path in npz_files:
            prefix = npz_to_prefix(npz_path)
            print(f"  python train_and_evaluate.py --csv-path data/{prefix}.csv")
    else:
        print("\nSome datasets failed – see output above for details.")

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
