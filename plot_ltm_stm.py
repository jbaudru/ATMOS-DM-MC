#!/usr/bin/env python3
"""
Visualise the GraphIDyOM LTM and STM next-node distributions on the road
network, for a given network and a given user.

For one query context taken from a trip of the chosen user, this script:

  * builds the *population* LTM from every other user's trajectories
    (the chosen user's own trips are held out, exactly as in the
    user-aware evaluation protocol),
  * builds the *user-specific* STM from the chosen user's other trips
    (concatenated, truncated to the last ``--history-nodes`` nodes),
  * runs ``GraphIDyOMPredictor.forward_with_explanations`` to obtain
    ``P_LTM(next | context)`` and ``P_STM(next | context)``,
  * draws the road network twice (LTM panel + STM panel), colouring and
    sizing each node by its predicted probability, and marking the context
    path, the current node and the true next node.

Usage
-----
    python plot_ltm_stm.py --network SG --user 1234
    python plot_ltm_stm.py --network US             # auto-pick a busy user
    python plot_ltm_stm.py --network SG --user 1234 --order 10 --top-k 15

The network is resolved from the standard WorldMove data/ files, matching
train_models.py (worldmove_{380_US,431_DE,539_SG,609_AE}).
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from train_and_evaluate import (
    load_trajectories,
    load_node_coordinates,
    encode_paths_categorical,
    make_samples,
)
from models import GraphIDyOMPredictor
from models.graph_idyom import build_graph_structure_from_sequences


# ---------------------------------------------------------------------------
# Network resolution (mirror of train_models.py defaults)
# ---------------------------------------------------------------------------
NETWORK_FILES = {
    "US": ("data/worldmove_380_US.csv", "data/worldmove_380_US_network.json"),
    "DE": ("data/worldmove_431_DE.csv", "data/worldmove_431_DE_network.json"),
    "SG": ("data/worldmove_539_SG.csv", "data/worldmove_539_SG_network.json"),
    "AE": ("data/worldmove_609_AE.csv", "data/worldmove_609_AE_network.json"),
}


def resolve_network(network: str) -> Tuple[str, str]:
    key = network.upper()
    if key not in NETWORK_FILES:
        raise ValueError(f"Unknown network '{network}'. "
                         f"Choose one of {list(NETWORK_FILES)}.")
    data_path, graph_path = NETWORK_FILES[key]
    data_path = os.path.join(_REPO, data_path)
    graph_path = os.path.join(_REPO, graph_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Trajectory CSV not found: {data_path}")
    return data_path, graph_path


# ---------------------------------------------------------------------------
# Road-network edges (for drawing)
# ---------------------------------------------------------------------------

def load_edges(graph_path: str, node_to_idx: Dict[int, int]) -> List[Tuple[int, int]]:
    """Return directed (u, v) edges in dense cat-id space for plotting."""
    import json
    try:
        with open(graph_path) as f:
            g = json.load(f)
    except Exception as exc:
        print(f"  Warning: could not read edges from {graph_path}: {exc}")
        return []
    edge_key = "links" if "links" in g else "edges"
    edges = []
    for e in g.get(edge_key, []):
        s, t = e.get("source"), e.get("target")
        if s in node_to_idx and t in node_to_idx:
            edges.append((node_to_idx[s], node_to_idx[t]))
    return edges


# ---------------------------------------------------------------------------
# STM support helpers (find a context the user's history actually covers)
# ---------------------------------------------------------------------------

def build_hist_kgrams(hist: List[int], max_order: int) -> List[set]:
    """Pre-compute the set of contiguous k-grams (k = 1..max_order) that occur
    in ``hist`` *and are followed by another node* (so a successor exists).

    ``sets[k]`` holds every length-k tuple seen at a position ``i`` with
    ``i + k < len(hist)``.  A context whose k-suffix is in ``sets[k]`` therefore
    has genuine order-k STM support.
    """
    H = len(hist)
    sets: List[set] = [set() for _ in range(max_order + 1)]
    for k in range(1, max_order + 1):
        for i in range(H - k):
            sets[k].add(tuple(hist[i:i + k]))
    return sets


def match_order(context: List[int], hist_kgrams: List[set],
                max_order: int) -> int:
    """Highest Markov order at which ``context``'s suffix is covered by the
    user's history (0 = no support)."""
    best = 0
    for k in range(1, min(max_order, len(context)) + 1):
        if tuple(context[-k:]) in hist_kgrams[k]:
            best = k
    return best


# ---------------------------------------------------------------------------
# Plot one distribution panel
# ---------------------------------------------------------------------------

def _draw_panel(ax, title, probs, coords, edges,
                context_nodes, current_node, true_next, top_k,
                zoom_nodes=None, pred_node=None):
    """Draw the road network with nodes coloured/sized by ``probs``."""
    finite = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])

    # --- faint road edges -------------------------------------------------
    seg = []
    for u, v in edges:
        if finite[u] and finite[v]:
            seg.append([(coords[u, 0], coords[u, 1]),
                        (coords[v, 0], coords[v, 1])])
    if seg:
        ax.add_collection(LineCollection(seg, colors="0.85", linewidths=0.4,
                                         zorder=1))

    # --- all reachable nodes, faint --------------------------------------
    ax.scatter(coords[finite, 0], coords[finite, 1], s=2, c="0.8",
               zorder=2, linewidths=0)

    # --- probability heat over nodes -------------------------------------
    pmask = finite & (probs > 0)
    # Treat a (near-)uniform distribution specially: there is no informative
    # peak, so don't blow every node up into one big blob.
    nz = probs[pmask]
    near_uniform = nz.size > 0 and (nz.max() <= 5.0 * nz.min() + 1e-12)
    if pmask.any():
        order = np.argsort(probs[pmask])
        xs = coords[pmask, 0][order]
        ys = coords[pmask, 1][order]
        ps = probs[pmask][order]
        if near_uniform:
            sizes = np.full(ps.shape, 8.0)
        else:
            # sqrt scaling gives readable contrast between the dominant nodes
            # and the long tail (a peaked STM no longer leaves the rest at a
            # single floor size).
            rel = ps / (ps.max() + 1e-12)
            sizes = 12.0 + 480.0 * np.sqrt(rel)
        sc = ax.scatter(xs, ys, c=ps, s=sizes, cmap="viridis",
                        norm=matplotlib.colors.LogNorm(
                            vmin=max(ps.min(), ps.max() * 1e-4),
                            vmax=ps.max()),
                        alpha=0.9, zorder=3, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="P(next node)")

    # --- context path -----------------------------------------------------
    cxy = [(coords[c, 0], coords[c, 1]) for c in context_nodes if finite[c]]
    if len(cxy) >= 2:
        cx, cy = zip(*cxy)
        ax.plot(cx, cy, "-", color="tab:blue", lw=1.5, alpha=0.7, zorder=4,
                label="context path")
    if cxy:
        cx, cy = zip(*cxy)
        ax.scatter(cx, cy, s=30, facecolors="none", edgecolors="tab:blue",
                   linewidths=1.2, zorder=5)

    # --- current node + true next ----------------------------------------
    if current_node is not None and finite[current_node]:
        ax.scatter([coords[current_node, 0]], [coords[current_node, 1]],
                   s=160, marker="o", facecolors="none",
                   edgecolors="tab:orange", linewidths=2.5, zorder=6,
                   label="current node")
    # This panel's own top-1 prediction (argmax of its distribution).
    if pred_node is not None and finite[pred_node]:
        ax.scatter([coords[pred_node, 0]], [coords[pred_node, 1]],
                   s=200, marker="D", facecolors="none",
                   edgecolors="magenta", linewidths=2.2, zorder=6,
                   label="predicted (argmax)")
    if true_next is not None and finite[true_next]:
        ax.scatter([coords[true_next, 0]], [coords[true_next, 1]],
                   s=240, marker="*", color="red", edgecolors="black",
                   linewidths=0.6, zorder=7, label="true next node")

    # --- annotate top-k probability nodes --------------------------------
    if top_k > 0 and pmask.any():
        top_idx = np.argsort(probs)[::-1][:top_k]
        for r, ni in enumerate(top_idx):
            if probs[ni] <= 0 or not finite[ni]:
                continue
            ax.annotate(f"{probs[ni]:.2f}",
                        (coords[ni, 0], coords[ni, 1]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="black", zorder=8)

    # --- zoom to region of interest --------------------------------------
    if zoom_nodes:
        zn = [n for n in zoom_nodes if finite[n]]
        if zn:
            zx = coords[zn, 0]
            zy = coords[zn, 1]
            padx = max((zx.max() - zx.min()) * 0.15, 1e-3)
            pady = max((zy.max() - zy.min()) * 0.15, 1e-3)
            ax.set_xlim(zx.min() - padx, zx.max() + padx)
            ax.set_ylim(zy.min() - pady, zy.max() + pady)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot GraphIDyOM LTM & STM next-node probabilities on the "
                    "road network for a given network and user.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--network", default="SG", choices=list(NETWORK_FILES),
                    help="Which road network / dataset to use.")
    ap.add_argument("--user", type=int, default=None,
                    help="User id to visualise (default: auto-pick the user "
                         "with the most trips).")
    ap.add_argument("--order", type=int, default=10,
                    help="Max Markov order K (also the context window length).")
    ap.add_argument("--history-nodes", type=int, default=3000,
                    help="STM buffer length: last N nodes of the user's other "
                         "trips.")
    ap.add_argument("--trip-index", type=int, default=0,
                    help="Which of the user's trips supplies the query context "
                         "(index into that user's trip list).")
    ap.add_argument("--target-pos", type=int, default=None,
                    help="Position in the chosen trip to predict (default: the "
                         "last node; context = the preceding --order nodes).")
    ap.add_argument("--ltm-max-trips", type=int, default=None,
                    help="Optionally cap the number of population trips used to "
                         "build the LTM (speed/memory; default: use all).")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Annotate this many highest-probability nodes per panel.")
    ap.add_argument("--no-zoom", dest="zoom", action="store_false",
                    help="Plot the whole network instead of zooming to the "
                         "context / probability region.")
    ap.set_defaults(zoom=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="Output image path (default: "
                         "ltm_stm_<network>_user<id>.png).")
    args = ap.parse_args()

    data_path, graph_path = resolve_network(args.network)
    seq_len = args.order
    rng = np.random.default_rng(args.seed)

    print(f"[1] Loading trajectories from {os.path.basename(data_path)} …")
    full_paths, node_vocab, user_ids = load_trajectories(
        data_path, seq_len, return_users=True)
    node_to_idx = {n: i for i, n in enumerate(node_vocab)}
    vocab_size = len(node_vocab)
    print(f"    vocab={vocab_size}  trips={len(full_paths)}  "
          f"users={len(set(user_ids))}")

    coords = load_node_coordinates(graph_path, node_to_idx, vocab_size)
    if coords is None:
        print("ERROR: no node coordinates in the network JSON — cannot plot.")
        return 1
    edges = load_edges(graph_path, node_to_idx)
    print(f"    edges={len(edges)}  nodes-with-coords="
          f"{int(np.isfinite(coords[:, 0]).sum())}")

    # ---- group trips by user --------------------------------------------
    trips_by_user: Dict[int, List[int]] = defaultdict(list)
    for i, u in enumerate(user_ids):
        trips_by_user[u].append(i)

    # ---- choose the user -------------------------------------------------
    if args.user is None:
        user = max(trips_by_user, key=lambda u: len(trips_by_user[u]))
        print(f"[2] Auto-picked user {user} "
              f"({len(trips_by_user[user])} trips).")
    else:
        user = args.user
        if user not in trips_by_user:
            print(f"ERROR: user {user} not found. Example users: "
                  f"{sorted(trips_by_user)[:10]}")
            return 1
        print(f"[2] User {user} has {len(trips_by_user[user])} trips.")

    user_trip_idx = trips_by_user[user]
    if len(user_trip_idx) < 2:
        print("ERROR: the chosen user needs >= 2 trips (one for the query "
              "context, the rest for the STM history).")
        return 1

    # ---- query trip + STM history ---------------------------------------
    ti = args.trip_index % len(user_trip_idx)
    query_global_idx = user_trip_idx[ti]
    query_trip = [node_to_idx[n] for n in full_paths[query_global_idx]
                  if n in node_to_idx]
    if len(query_trip) <= seq_len:
        print(f"ERROR: query trip too short ({len(query_trip)} <= "
              f"seq_len={seq_len}). Try a different --trip-index or smaller "
              f"--order.")
        return 1

    # STM history: the user's OTHER trips (the query trip is held out so the
    # context is never trivially matched against itself).
    stm_nodes: List[int] = []
    for j in user_trip_idx:
        if j == query_global_idx:
            continue
        stm_nodes.extend(node_to_idx[n] for n in full_paths[j]
                         if n in node_to_idx)
    stm_nodes = stm_nodes[-args.history_nodes:]
    print(f"[4] STM history: {len(stm_nodes)} nodes from "
          f"{len(user_trip_idx) - 1} other trips of user {user}.")

    # ---- pick the target position ---------------------------------------
    # By default, auto-select the position whose context the user's STM history
    # actually covers (longest suffix match), so the STM is informative instead
    # of falling back to a uniform distribution (which gives alpha_STM ~ 0).
    hist_kgrams = build_hist_kgrams(stm_nodes, seq_len) if stm_nodes else \
        [set() for _ in range(seq_len + 1)]
    if args.target_pos is not None:
        tpos = max(seq_len, min(args.target_pos, len(query_trip) - 1))
        best_order = match_order(query_trip[tpos - seq_len:tpos],
                                 hist_kgrams, seq_len)
    else:
        best_pos, best_order = seq_len, -1
        for pos in range(seq_len, len(query_trip)):
            mo = match_order(query_trip[pos - seq_len:pos], hist_kgrams, seq_len)
            if mo > best_order:
                best_order, best_pos = mo, pos
        tpos = best_pos
        if best_order <= 0:
            print("    [auto-context] WARNING: no context in this trip is "
                  "covered by the user's STM history (STM will be uniform, "
                  "alpha_STM ~ 0). Try another --trip-index.")
        else:
            print(f"    [auto-context] selected position {tpos} with STM "
                  f"support order {best_order}.")

    context_nodes = query_trip[tpos - seq_len:tpos]
    true_next = query_trip[tpos]
    current_node = context_nodes[-1]
    print(f"[3] Query context (len {len(context_nodes)}): {context_nodes}")
    print(f"    current node = {current_node}   true next = {true_next}   "
          f"STM match order = {best_order}")

    # ---- LTM corpus: every trip NOT belonging to the chosen user ---------
    user_trip_set = set(user_trip_idx)
    ltm_global_idx = [i for i in range(len(full_paths)) if i not in user_trip_set]
    if args.ltm_max_trips is not None and len(ltm_global_idx) > args.ltm_max_trips:
        keep = rng.choice(len(ltm_global_idx), size=args.ltm_max_trips,
                          replace=False)
        ltm_global_idx = [ltm_global_idx[k] for k in sorted(keep)]
    ltm_paths = [full_paths[i] for i in ltm_global_idx]
    print(f"[5] Building LTM from {len(ltm_paths)} population trips "
          f"(user {user} held out) …")
    cat_ltm = encode_paths_categorical(ltm_paths, node_to_idx)
    X_ltm, _ = make_samples(cat_ltm, seq_len)
    if len(X_ltm) == 0:
        print("ERROR: no LTM training windows could be built.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    device={device}  ltm_windows={len(X_ltm)}")
    seqs_t = torch.tensor(X_ltm, dtype=torch.long, device=device)
    graph_structure = build_graph_structure_from_sequences(
        seqs_t, vocab_size, max_order=args.order)

    # ---- run the model: LTM & STM distributions for the query context ----
    print("[6] Computing LTM / STM distributions …")
    model = GraphIDyOMPredictor(vocab_size=vocab_size,
                                max_order=args.order).to(device).eval()
    ctx_t = torch.tensor([context_nodes], dtype=torch.long, device=device)
    if stm_nodes:
        uh_t = torch.tensor([stm_nodes], dtype=torch.long, device=device)
        uhl_t = torch.tensor([len(stm_nodes)], dtype=torch.long, device=device)
    else:
        uh_t = uhl_t = None
    with torch.no_grad():
        out = model.forward_with_explanations(
            ctx_t, graph_structure=graph_structure,
            user_history=uh_t, user_history_lengths=uhl_t)
    p_ltm = out["ltm_probs"][0].cpu().numpy()
    p_stm = out["stm_probs"][0].cpu().numpy()
    a_ltm = float(out["alpha_ltm"][0])
    a_stm = float(out["alpha_stm"][0])
    # Combined log-probs -> probabilities (the model's actual prediction).
    p_comb = torch.softmax(out["logits"][0], dim=0).cpu().numpy()
    pred_ltm  = int(p_ltm.argmax())
    pred_stm  = int(p_stm.argmax())
    pred_comb = int(p_comb.argmax())
    print(f"    LTM top-1      = node {pred_ltm} (p={p_ltm[pred_ltm]:.3f})")
    print(f"    STM top-1      = node {pred_stm} (p={p_stm[pred_stm]:.3f})")
    print(f"    Combined top-1 = node {pred_comb} (p={p_comb[pred_comb]:.3f})")
    print(f"    true next node = {true_next}")
    print(f"    combination weights: alpha_LTM={a_ltm:.3f}  "
          f"alpha_STM={a_stm:.3f}")

    # ---- zoom region: context + top probability mass --------------------
    zoom_nodes = None
    if args.zoom:
        z = set(context_nodes) | {true_next}
        for arr in (p_ltm, p_stm):
            z.update(int(i) for i in np.argsort(arr)[::-1][:max(args.top_k, 20)])
        # include nearby STM nodes so the personal footprint is visible
        z.update(int(n) for n in stm_nodes[-args.order * 5:])
        zoom_nodes = sorted(z)

    # ---- plot ------------------------------------------------------------
    print("[7] Rendering …")
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    _draw_panel(axes[0],
                f"LTM (population)  -  weight alpha={a_ltm:.2f}",
                p_ltm, coords, edges, context_nodes, current_node, true_next,
                args.top_k, zoom_nodes, pred_node=pred_ltm)
    _draw_panel(axes[1],
                f"STM (user {user})  -  weight alpha={a_stm:.2f}",
                p_stm, coords, edges, context_nodes, current_node, true_next,
                args.top_k, zoom_nodes, pred_node=pred_stm)
    _hit = lambda p: "  (hit)" if p == true_next else ""
    fig.suptitle(
        f"{args.network}  -  user {user}  -  order {args.order}  -  "
        f"STM history {len(stm_nodes)} nodes  -  STM match order {best_order}\n"
        f"context {context_nodes}  ->  true next {true_next}\n"
        f"LTM->{pred_ltm} (p={p_ltm[pred_ltm]:.2f}){_hit(pred_ltm)}    "
        f"STM->{pred_stm} (p={p_stm[pred_stm]:.2f}){_hit(pred_stm)}    "
        f"Combined->{pred_comb} (p={p_comb[pred_comb]:.2f}){_hit(pred_comb)}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = args.out or os.path.join(
        _REPO, f"ltm_stm_{args.network}_user{user}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n✓  Saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
