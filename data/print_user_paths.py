#!/usr/bin/env python3
"""Plot per-user node visits and first-order transition counts on the associated road network."""

import argparse
import heapq
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with: pip install matplotlib"
    ) from exc

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all paths belonging to one user from a trajectory CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("worldmove_380_US.csv"),
        help="CSV file to inspect.",
    )
    parser.add_argument(
        "--network-json",
        type=Path,
        default=None,
        help="Road-network JSON linked to the dataset. Defaults to <csv_stem>_network.json.",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="User identifier to inspect, for example user_00001.",
    )
    parser.add_argument(
        "--show-dates",
        action="store_true",
        help="Print the date column when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the plot PNG. Defaults to data/<user_id>_paths.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )
    parser.add_argument(
        "--add-basemap",
        action="store_true",
        help="Overlay an OpenStreetMap basemap when coordinates are lon/lat (requires contextily).",
    )
    parser.add_argument(
        "--show-network",
        dest="show_network",
        action="store_true",
        help="Draw the full road-network edges as a faint backdrop (default).",
    )
    parser.add_argument(
        "--hide-network",
        dest="show_network",
        action="store_false",
        help="Hide the full road-network backdrop.",
    )
    parser.set_defaults(show_network=True)
    parser.add_argument(
        "--no-route-transitions",
        dest="route_transitions",
        action="store_false",
        help="Draw transitions as straight node-to-node lines instead of routing them along the road network.",
    )
    parser.add_argument(
        "--top-arrows",
        type=int,
        default=12,
        help="How many of the strongest node-to-node transitions to annotate with arrows and counts.",
    )
    parser.add_argument(
        "--top-repeated",
        type=int,
        default=15,
        help="How many most-visited nodes to print.",
    )
    parser.add_argument(
        "--top-links",
        type=int,
        default=15,
        help="How many most-frequent first-order links to print.",
    )
    parser.add_argument(
        "--random-path",
        action="store_true",
        help="Select and plot a single random path instead of the aggregate distribution.",
    )
    parser.add_argument(
        "--path-index",
        type=int,
        default=None,
        help="Index of a specific path to plot (0-based row index among the user's trips). "
             "Takes precedence over --random-path when both are given.",
    )
    parser.add_argument(
        "--full-trajectory",
        action="store_true",
        help="In single-path mode, draw the raw trajectory node sequence as-is "
             "(may contain back-and-forth/dead-end artifacts). By default a clean "
             "start->end road-network route is drawn instead.",
    )
    return parser.parse_args()


def parse_path(path_value) -> list[int]:
    if pd.isna(path_value):
        return []
    return [int(node.strip()) for node in str(path_value).split(",") if node.strip()]


def resolve_path(path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    if path_value.exists():
        return path_value.resolve()
    if len(path_value.parts) > 1 or str(path_value).startswith("."):
        return (Path.cwd() / path_value).resolve()
    return Path(__file__).resolve().parent / path_value


def default_network_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_network.json")


def default_output_path(csv_path: Path, user_id: str) -> Path:
    safe_user_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(user_id)
    )
    return csv_path.with_name("path_plots") / f"{csv_path.stem}_{safe_user_id}_paths.png"


def load_network(network_path: Path) -> tuple[dict[int, tuple[float, float]], list[dict], dict]:
    with network_path.open("r", encoding="utf-8") as file_obj:
        network = json.load(file_obj)

    node_positions = {
        int(node["id"]): (float(node["x"]), float(node["y"]))
        for node in network.get("nodes", [])
        if {"id", "x", "y"}.issubset(node)
    }
    edges = []
    for edge in network.get("edges", []):
        try:
            src = int(edge["source"])
            dst = int(edge["target"])
        except (KeyError, TypeError, ValueError):
            continue
        if src in node_positions and dst in node_positions:
            geometry = edge.get("geometry")
            if geometry:
                segment = [(float(x), float(y)) for x, y in geometry]
            else:
                segment = [node_positions[src], node_positions[dst]]
            edges.append({
                "source": src,
                "target": dst,
                "length": float(edge.get("length", 1.0) or 1.0),
                "segment": segment,
            })
    return node_positions, edges, network.get("metadata", {})


def looks_like_lon_lat(node_positions: dict[int, tuple[float, float]]) -> bool:
    if not node_positions:
        return False
    xs = [xy[0] for xy in node_positions.values()]
    ys = [xy[1] for xy in node_positions.values()]
    return (min(xs) >= -180 and max(xs) <= 180 and min(ys) >= -90 and max(ys) <= 90)


def build_segments(paths: list[list[int]], node_positions: dict[int, tuple[float, float]]) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    for path_nodes in paths:
        for src, dst in zip(path_nodes, path_nodes[1:]):
            if src in node_positions and dst in node_positions:
                segments.append([node_positions[src], node_positions[dst]])
    return segments


def build_edge_indexes(network_edges: list[dict]) -> tuple[dict[int, list[tuple[int, float]]], dict[tuple[int, int], list[tuple[float, float]]]]:
    adjacency: dict[int, list[tuple[int, float]]] = {}
    edge_segments: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for edge in network_edges:
        src = edge["source"]
        dst = edge["target"]
        adjacency.setdefault(src, []).append((dst, edge["length"]))
        edge_segments[(src, dst)] = edge["segment"]
    return adjacency, edge_segments


def shortest_path_nodes(adjacency: dict[int, list[tuple[int, float]]], src: int, dst: int) -> list[int] | None:
    if src == dst:
        return [src]
    heap = [(0.0, src)]
    distances = {src: 0.0}
    previous: dict[int, int] = {}
    while heap:
        distance, node = heapq.heappop(heap)
        if node == dst:
            path = [dst]
            while path[-1] != src:
                path.append(previous[path[-1]])
            path.reverse()
            return path
        if distance > distances.get(node, float("inf")):
            continue
        for neighbor, length in adjacency.get(node, []):
            next_distance = distance + length
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                previous[neighbor] = node
                heapq.heappush(heap, (next_distance, neighbor))
    return None


def route_transition_counts(
    transition_items: list[tuple[tuple[int, int], int]],
    network_edges: list[dict],
    node_positions: dict[int, tuple[float, float]],
) -> list[tuple[list[tuple[float, float]], int]]:
    adjacency, edge_segments = build_edge_indexes(network_edges)
    routed_counts: Counter = Counter()
    direct_segments: list[tuple[list[tuple[float, float]], int]] = []

    for (src, dst), count in transition_items:
        path = shortest_path_nodes(adjacency, src, dst)
        if path is None or len(path) < 2:
            direct_segments.append(([node_positions[src], node_positions[dst]], count))
            continue
        for start, end in zip(path, path[1:]):
            routed_counts[(start, end)] += count

    routed_segments = [
        (edge_segments[(src, dst)], count)
        for (src, dst), count in routed_counts.items()
        if (src, dst) in edge_segments
    ]
    return routed_segments + direct_segments


def plot_user_paths(
    user_id: str,
    node_positions: dict[int, tuple[float, float]],
    network_edges: list[tuple[int, int]],
    node_counts: Counter,
    link_counts: Counter,
    output_path: Path,
    show_plot: bool,
    add_basemap: bool,
    show_network: bool,
    top_arrows: int,
    route_transitions: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    if show_network and network_edges:
        network_segments = [edge["segment"] for edge in network_edges]
        if network_segments:
            network_collection = LineCollection(
                network_segments,
                colors="#b8b8b8",
                linewidths=0.18,
                alpha=0.45,
                zorder=1,
            )
            ax.add_collection(network_collection)

    # Genuine node-to-node transitions only (exclude self-loops, which are
    # zero-length segments that would otherwise dominate the color/width scale
    # and hide the actual movements between distinct nodes).
    transition_items = [
        ((src, dst), count)
        for (src, dst), count in link_counts.items()
        if src != dst and src in node_positions and dst in node_positions
    ]
    if transition_items:
        transition_items.sort(key=lambda item: item[1])
        if route_transitions and network_edges:
            routed_items = route_transition_counts(transition_items, network_edges, node_positions)
        else:
            routed_items = [
                ([node_positions[src], node_positions[dst]], count)
                for (src, dst), count in transition_items
            ]
        link_segments = [segment for segment, _ in routed_items]
        link_values = [count for _, count in routed_items]
        max_link = max(link_values)
        widths = [0.8 + 5.0 * (count / max_link) for count in link_values]
        edge_collection = LineCollection(
            link_segments,
            cmap="OrRd",
            linewidths=widths,
            alpha=0.9,
            zorder=2,
        )
        edge_collection.set_array(np.asarray(link_values, dtype=float))
        ax.add_collection(edge_collection)
        edge_colorbar = fig.colorbar(edge_collection, ax=ax, shrink=0.8, pad=0.02)
        edge_colorbar.set_label("Node-to-node transition count")

        # Annotate the strongest transitions with directional arrows + counts so
        # the movement from one node to the next is explicitly visible.
        top_transitions = sorted(
            transition_items, key=lambda item: item[1], reverse=True
        )[:max(top_arrows, 0)]
        for (src, dst), count in top_transitions:
            x0, y0 = node_positions[src]
            x1, y1 = node_positions[dst]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#08306b",
                    lw=1.2,
                    alpha=0.9,
                    shrinkA=2.5,
                    shrinkB=2.5,
                ),
                zorder=4,
            )
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                str(count),
                fontsize=7,
                color="#08306b",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=5,
            )

    node_items = [
        (node_id, count)
        for node_id, count in node_counts.items()
        if node_id in node_positions
    ]
    if node_items:
        node_xy = [node_positions[node_id] for node_id, _ in node_items]
        node_values = [count for _, count in node_items]
        max_node = max(node_values)
        node_sizes = [10 + 90 * (count / max_node) for count in node_values]
        scatter = ax.scatter(
            [xy[0] for xy in node_xy],
            [xy[1] for xy in node_xy],
            s=node_sizes,
            c=node_values,
            cmap="Blues",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )
        node_colorbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.08)
        node_colorbar.set_label("Node visit count")

    ax.set_title(f"User {user_id}: node visits and node-to-node transitions")
    ax.set_xlabel("Longitude / x")
    ax.set_ylabel("Latitude / y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)

    if add_basemap:
        try:
            import contextily as ctx  # type: ignore[import-not-found]
        except ImportError:
            ctx = None

        if ctx is None:
            print("Warning: --add-basemap requested but contextily is not installed.")
            print("Install with: pip install contextily")
        elif looks_like_lon_lat(node_positions):
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    crs="EPSG:4326",
                    attribution_size=6,
                )
            except Exception as exc:
                print(f"Warning: could not load basemap tiles: {exc}")
        else:
            print("Warning: coordinates do not look like lon/lat; basemap overlay skipped.")

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def build_undirected_route_index(
    network_edges: list[dict],
) -> tuple[dict[int, list[tuple[int, float]]], dict[tuple[int, int], list[tuple[float, float]]]]:
    """Undirected adjacency + segment geometry lookup for clean routing.

    Returns (adjacency, segment_map) where segment_map[(u, v)] is the road
    polyline from u to v (reversed automatically for the opposite direction).
    """
    adjacency: dict[int, list[tuple[int, float]]] = {}
    segment_map: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for edge in network_edges:
        u = edge["source"]
        v = edge["target"]
        if u == v:
            continue
        length = float(edge.get("length", 1.0) or 1.0)
        seg = edge["segment"]
        adjacency.setdefault(u, []).append((v, length))
        adjacency.setdefault(v, []).append((u, length))
        segment_map.setdefault((u, v), seg)
        segment_map.setdefault((v, u), list(reversed(seg)))
    return adjacency, segment_map


def compute_clean_route(
    path_nodes: list[int],
    network_edges: list[dict],
    node_positions: dict[int, tuple[float, float]],
) -> tuple[list[int], list[list[tuple[float, float]]]] | None:
    """Shortest road-network route from the path's origin to its destination.

    This collapses the raw trajectory (which may oscillate between coarse grid
    cells, creating back-and-forth / dead-end artifacts) into a single clean
    path that runs from start to end with no detours. Returns (route_nodes,
    per-step road polylines) or ``None`` when no route exists / no network.
    """
    if not network_edges or len(path_nodes) < 2:
        return None
    origin = path_nodes[0]
    destination = path_nodes[-1]
    if origin == destination or origin not in node_positions or destination not in node_positions:
        return None

    adjacency, segment_map = build_undirected_route_index(network_edges)
    route_nodes = shortest_path_nodes(adjacency, origin, destination)
    if not route_nodes or len(route_nodes) < 2:
        return None

    segments = [
        segment_map.get((a, b), [node_positions[a], node_positions[b]])
        for a, b in zip(route_nodes, route_nodes[1:])
        if a in node_positions and b in node_positions
    ]
    return route_nodes, segments


def plot_single_path(
    user_id: str,
    path_nodes: list[int],
    path_index: int,
    node_positions: dict[int, tuple[float, float]],
    network_edges: list[dict],
    output_path: Path,
    show_plot: bool,
    add_basemap: bool,
    show_network: bool,
    route_transitions: bool,
    full_trajectory: bool = False,
) -> None:
    """Plot one individual path as a single sequentially-coloured line.

    By default a clean shortest-path route between the trajectory's origin and
    destination is drawn (no back-and-forth / dead-end artifacts). Pass
    ``full_trajectory=True`` to render the raw node sequence instead.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background road network.
    if show_network and network_edges:
        net_segs = [e["segment"] for e in network_edges]
        ax.add_collection(LineCollection(net_segs, colors="#b8b8b8", linewidths=0.18, alpha=0.45, zorder=1))

    # ---------------------------------------------------------------
    # Decide which node sequence + geometry to draw.
    # ---------------------------------------------------------------
    clean = None if full_trajectory else compute_clean_route(path_nodes, network_edges, node_positions)

    if clean is not None:
        route_nodes, step_segments = clean
        drawn_nodes = route_nodes
        mode_label = "clean start\u2192end route"
    else:
        if not full_trajectory:
            print("Note: clean route unavailable (no network / unreachable / loop) – "
                  "drawing raw trajectory instead.")
        # Raw trajectory: route each consecutive pair along the network when asked.
        route_nodes = [n for n in path_nodes if n in node_positions]
        step_segments = []
        if route_transitions and network_edges:
            adjacency, edge_segments_map = build_edge_indexes(network_edges)
        for src, dst in zip(path_nodes, path_nodes[1:]):
            if src == dst or src not in node_positions or dst not in node_positions:
                continue
            if route_transitions and network_edges:
                road_nodes = shortest_path_nodes(adjacency, src, dst)
                if road_nodes and len(road_nodes) >= 2:
                    for s, d in zip(road_nodes, road_nodes[1:]):
                        step_segments.append(
                            edge_segments_map.get((s, d), [node_positions[s], node_positions[d]])
                        )
                    continue
            step_segments.append([node_positions[src], node_positions[dst]])
        drawn_nodes = route_nodes
        mode_label = "raw trajectory"

    n_segments = len(step_segments)

    if n_segments == 0:
        print("Warning: selected path has no drawable segments.")
    else:
        cmap = plt.get_cmap("plasma")
        # Continuous gradient along the route: colour each segment by its order.
        seg_fracs = np.linspace(0.0, 1.0, n_segments)
        route_collection = LineCollection(
            step_segments,
            cmap=cmap,
            linewidths=3.0,
            alpha=0.95,
            zorder=2,
        )
        route_collection.set_array(seg_fracs)
        ax.add_collection(route_collection)

        # A handful of evenly-spaced directional arrows along the route.
        n_arrows = min(10, n_segments)
        for k in range(n_arrows):
            seg = step_segments[round(k * (n_segments - 1) / max(n_arrows - 1, 1))]
            (x0, y0), (x1, y1) = seg[0], seg[-1]
            if (x0, y0) == (x1, y1):
                continue
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.3, alpha=0.85,
                                shrinkA=2, shrinkB=2),
                zorder=4,
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("Route progression (start \u2192 end)")

    # Start node (green).
    if path_nodes and path_nodes[0] in node_positions:
        sx, sy = node_positions[path_nodes[0]]
        ax.scatter([sx], [sy], s=170, c="#2ca02c", edgecolors="black", linewidths=0.9, zorder=5)
        ax.text(sx, sy, "  Start", fontsize=9, color="#2ca02c", fontweight="bold", zorder=6)

    # End node (red), only if different from start.
    if len(path_nodes) > 1 and path_nodes[-1] in node_positions and path_nodes[-1] != path_nodes[0]:
        ex, ey = node_positions[path_nodes[-1]]
        ax.scatter([ex], [ey], s=170, c="#d62728", edgecolors="black", linewidths=0.9, zorder=5)
        ax.text(ex, ey, "  End", fontsize=9, color="#d62728", fontweight="bold", zorder=6)

    ax.set_title(
        f"User {user_id} \u2014 path #{path_index}  ({mode_label}, {len(drawn_nodes)} nodes)"
    )
    ax.set_xlabel("Longitude / x")
    ax.set_ylabel("Latitude / y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)

    if add_basemap:
        try:
            import contextily as ctx  # type: ignore[import-not-found]
        except ImportError:
            ctx = None
        if ctx is None:
            print("Warning: --add-basemap requested but contextily is not installed.")
            print("Install with: pip install contextily")
        elif looks_like_lon_lat(node_positions):
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    crs="EPSG:4326",
                    attribution_size=6,
                )
            except Exception as exc:
                print(f"Warning: could not load basemap tiles: {exc}")
        else:
            print("Warning: coordinates do not look like lon/lat; basemap overlay skipped.")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()

    csv_path = resolve_path(args.csv)
    network_path = (resolve_path(args.network_json)
                    if args.network_json is not None
                    else default_network_path(csv_path))

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return 1
    if not network_path.exists():
        print(f"Network JSON file not found: {network_path}")
        return 1

    df = pd.read_csv(csv_path)
    required_columns = {"user_id", "q_path"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        print(f"Missing required columns: {', '.join(missing_columns)}")
        return 1

    user_rows = df[df["user_id"].astype(str) == str(args.user_id)].copy()
    if user_rows.empty:
        print(f"No trajectories found for user_id={args.user_id}")
        return 0

    user_rows["parsed_path"] = user_rows["q_path"].apply(parse_path)
    node_positions, network_edges, metadata = load_network(network_path)
    all_paths = user_rows["parsed_path"].tolist()
    node_counts = Counter(node for path_nodes in all_paths for node in path_nodes)
    link_counts = Counter(
        (src, dst)
        for path_nodes in all_paths
        for src, dst in zip(path_nodes, path_nodes[1:])
    )
    ranked_nodes = node_counts.most_common()
    transition_counts = Counter(
        {(src, dst): count for (src, dst), count in link_counts.items() if src != dst}
    )
    selfloop_counts = Counter(
        {src: count for (src, dst), count in link_counts.items() if src == dst}
    )
    ranked_links = transition_counts.most_common()

    # ── Single-path mode ─────────────────────────────────────────────────────
    single_path_mode = args.random_path or args.path_index is not None
    if single_path_mode:
        valid_indexed = [(i, p) for i, p in enumerate(all_paths) if p]
        if not valid_indexed:
            print(f"No valid (non-empty) paths found for user_id={args.user_id}")
            return 1
        if args.path_index is not None:
            if args.path_index < 0 or args.path_index >= len(all_paths):
                print(f"Path index {args.path_index} is out of range (0\u2013{len(all_paths) - 1}).")
                return 1
            selected_index = args.path_index
            selected_nodes = all_paths[args.path_index]
            if not selected_nodes:
                print(f"Path at index {args.path_index} is empty.")
                return 1
        else:
            selected_index, selected_nodes = random.choice(valid_indexed)

        if args.output is None:
            base = default_output_path(csv_path, str(args.user_id))
            output_path = base.with_name(f"{base.stem}_path{selected_index}.png")
        else:
            output_path = resolve_path(args.output)

        print(f"CSV: {csv_path}")
        print(f"Network: {network_path}")
        print(f"Network nodes: {len(node_positions)}")
        print(f"Network edges: {len(network_edges)}")
        print(f"User: {args.user_id}")
        print(f"Trips: {len(user_rows)}")
        print(f"Selected path index: {selected_index}")
        print(f"Path nodes ({len(selected_nodes)}): {selected_nodes}")
        distinct_steps = [(s, d) for s, d in zip(selected_nodes, selected_nodes[1:]) if s != d]
        print(f"Distinct steps: {len(distinct_steps)}")
        print()

        plot_single_path(
            user_id=str(args.user_id),
            path_nodes=selected_nodes,
            path_index=selected_index,
            node_positions=node_positions,
            network_edges=network_edges,
            output_path=output_path,
            show_plot=args.show,
            add_basemap=args.add_basemap,
            show_network=args.show_network,
            route_transitions=args.route_transitions,
            full_trajectory=args.full_trajectory,
        )
        return 0

    # ── Aggregate distribution mode (default) ────────────────────────────────
    if args.output is None:
        output_path = default_output_path(csv_path, str(args.user_id))
    else:
        output_path = resolve_path(args.output)

    print(f"CSV: {csv_path}")
    print(f"Network: {network_path}")
    if metadata:
        print(f"Network source: {metadata.get('source', 'unknown')}")
        print(f"Network description: {metadata.get('description', 'N/A')}")
        print(f"Network projection: {metadata.get('projection', 'N/A')}")
    print(f"Network nodes: {len(node_positions)}")
    print(f"Network edges: {len(network_edges)}")
    print(f"User: {args.user_id}")
    print(f"Trips: {len(user_rows)}")
    print(f"Unique dates: {user_rows['date'].nunique() if 'date' in user_rows.columns else 'N/A'}")
    print(f"Total node visits: {sum(node_counts.values())}")
    print(f"Unique nodes visited: {len(node_counts)}")
    print(f"Unique first-order links: {len(link_counts)}")
    print(f"Node-to-node transitions (distinct nodes): {len(transition_counts)}")
    print(f"Stationary self-loops (node -> same node): {len(selfloop_counts)}")
    if metadata.get("source") == "grid":
        print("Warning: this network is a fallback grid network, not a true OSM road graph.")
        print("         Geolocation will be approximate and may not match street-level OSM geometry.")
    print()
    print(f"Top {min(args.top_repeated, len(ranked_nodes))} visited nodes:")
    for node_id, count in ranked_nodes[:args.top_repeated]:
        print(f"  node {node_id}: {count} visits")
    print()

    if ranked_links:
        print(f"Top {min(args.top_links, len(ranked_links))} node-to-node transitions (distinct nodes):")
        for (src, dst), count in ranked_links[:args.top_links]:
            print(f"  {src} -> {dst}: {count} transitions")
        print()

    if selfloop_counts:
        ranked_selfloops = selfloop_counts.most_common()
        print(f"Top {min(args.top_links, len(ranked_selfloops))} stationary self-loops (node -> same node):")
        for node_id, count in ranked_selfloops[:args.top_links]:
            print(f"  {node_id} -> {node_id}: {count} stays")
        print()

    plot_user_paths(
        user_id=str(args.user_id),
        node_positions=node_positions,
        network_edges=network_edges,
        node_counts=node_counts,
        link_counts=link_counts,
        output_path=output_path,
        show_plot=args.show,
        add_basemap=args.add_basemap,
        show_network=args.show_network,
        top_arrows=args.top_arrows,
        route_transitions=args.route_transitions,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())