#!/usr/bin/env python3
"""
Plot an example predicted trajectory for each model on the road network.

For one test trajectory, every model is asked to predict the continuation of
the path — either one step at a time from the *true* context ("next-node",
teacher-forced) or in closed loop where each prediction is fed back into the
context ("autoregressive").  Predictions are always made / scored in raw
NODE-ID space (using the same NodeSpaceAdapter decoding as evaluate_models.py),
so de-Bruijn and graph-embedding models are shown as concrete node paths too.

Each model gets its own panel: the road network is drawn as a faint backdrop
(inspired by data/print_user_paths.py), the ground-truth continuation in green,
and the model's prediction in red.

Usage
-----
    # autoregressive example for the default model set, dataset 0 / split 0
    python plot_predicted_paths.py --models-dir saved_models

    # one-step (teacher-forced) predictions, pick a specific test path + order
    python plot_predicted_paths.py --models-dir saved_models \\
        --mode next-node --order 5 --path-index 42 \\
        --dataset worldmove_431_DE --split 1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc

import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluate_models import (  # noqa: E402
    load_graphidyom,
    make_graphidyom_proba_fn,
    load_baseline,
    make_baseline_proba_fn,
    load_cat_neighbors,
)
from models.node_decode import (  # noqa: E402
    NodeSpaceAdapter,
    load_kgram_to_id,
    load_node_to_cluster,
)


# ---------------------------------------------------------------------------
# Network / artifact loading
# ---------------------------------------------------------------------------

def load_node_positions(graph_path: str,
                        node_to_idx: Dict) -> Tuple[Dict[int, Tuple[float, float]],
                                                    List[list],
                                                    Optional[Tuple[float, float, float, float]]]:
    """Load network geometry.

    Returns
    -------
    pos : dict
        dense_idx -> (x, y) for nodes present in ``node_to_idx`` (used to draw
        the predicted/true *paths*, which live in dense-index space).
    net_segments : list
        Full road-network segments in raw coordinate space (every edge of the
        graph, following edge ``geometry`` when available) — drawn as backdrop.
    extent : (xmin, xmax, ymin, ymax) or None
        Bounding box of the *whole* network so every panel shows it all.
    """
    with open(graph_path, encoding="utf-8") as f:
        gd = json.load(f)

    def _dense(raw) -> Optional[int]:
        for k in (str(raw), raw):
            if k in node_to_idx:
                return int(node_to_idx[k])
        return None

    # Raw positions for EVERY network node (full backdrop, like print_user_paths).
    raw_pos: Dict = {}
    pos: Dict[int, Tuple[float, float]] = {}
    for node in gd.get("nodes", []):
        if not {"id", "x", "y"}.issubset(node):
            continue
        xy = (float(node["x"]), float(node["y"]))
        raw_pos[node["id"]] = xy
        raw_pos[str(node["id"])] = xy
        d = _dense(node["id"])
        if d is not None:
            pos[d] = xy

    edges_key = "links" if "links" in gd else "edges"
    net_segments: List[list] = []
    for e in gd.get(edges_key, []):
        geom = e.get("geometry")
        if geom:
            seg = [(float(x), float(y)) for x, y in geom]
            if len(seg) >= 2:
                net_segments.append(seg)
                continue
        a, b = e.get("source"), e.get("target")
        pa = raw_pos.get(a, raw_pos.get(str(a)))
        pb = raw_pos.get(b, raw_pos.get(str(b)))
        if pa is not None and pb is not None:
            net_segments.append([pa, pb])

    extent = None
    if raw_pos:
        xs = [xy[0] for xy in raw_pos.values()]
        ys = [xy[1] for xy in raw_pos.values()]
        extent = (min(xs), max(xs), min(ys), max(ys))
    return pos, net_segments, extent


def build_adapters(split_dir: str, vocab_cat: int, seq_len: int,
                   debruijn_order: int,
                   neighbors: Optional[Dict[int, set]],
                   node_prior, node_transitions,
                   node_transitions2=None) -> Dict[str, NodeSpaceAdapter]:
    adapters: Dict[str, NodeSpaceAdapter] = {
        "categorical_id": NodeSpaceAdapter("categorical_id", vocab_cat, seq_len),
    }
    k2i = os.path.join(split_dir, "kgram_to_id.json")
    if os.path.isfile(k2i):
        adapters["debruijn"] = NodeSpaceAdapter(
            "debruijn", vocab_cat, seq_len,
            kgram_to_id=load_kgram_to_id(k2i), debruijn_order=debruijn_order)
    n2c = os.path.join(split_dir, "node_to_cluster.json")
    if os.path.isfile(n2c):
        adapters["graph_embedding"] = NodeSpaceAdapter(
            "graph_embedding", vocab_cat, seq_len,
            node_to_cluster=load_node_to_cluster(n2c, vocab_cat),
            neighbors=neighbors, node_prior=node_prior,
            transitions=node_transitions, transitions2=node_transitions2)
    return adapters


# ---------------------------------------------------------------------------
# Prediction (in node-ID space, via the encoding adapter)
# ---------------------------------------------------------------------------

def predict_nodes(proba_enc_fn, adapter: NodeSpaceAdapter,
                  node_path: List[int], seq_len: int, max_steps: int,
                  autoregressive: bool) -> Tuple[List[int], List[int]]:
    """Return (true_continuation, predicted_continuation) node lists.

    autoregressive=True feeds each top-1 node prediction back into the context;
    otherwise each prediction uses the true (teacher-forced) context window.
    """
    n_steps = min(len(node_path) - seq_len, max_steps)
    true_cont = [int(node_path[seq_len + s]) for s in range(n_steps)]
    pred_cont: List[int] = []
    node_ctx = [int(x) for x in node_path[:seq_len]]
    for step in range(n_steps):
        if not autoregressive:
            i = seq_len + step
            node_ctx = [int(x) for x in node_path[i - seq_len:i]]
        cur = node_ctx[-1]
        enc_ctx = adapter.encode_context(node_ctx)
        if not enc_ctx:
            pred = cur
        else:
            enc_proba = proba_enc_fn(enc_ctx)
            cand, probs = adapter.decode(enc_proba, current_node=cur)
            pred = int(cand[int(np.argmax(probs))]) if cand.size else cur
        pred_cont.append(pred)
        if autoregressive:
            node_ctx = node_ctx + [pred]
    return true_cont, pred_cont


def gather_model_predictors(split_dir: str, args, adapters, device):
    """Yield (display_name, proba_enc_fn, adapter) for each requested model."""
    out = []
    # GraphIDyOM, categorical encoding only (skip debruijn / graph_embedding).
    for enc in ("categorical_id",):
        if enc not in adapters:
            continue
        pt = os.path.join(split_dir, f"graphidyom_{enc}_ord{args.order}.pt")
        if not os.path.isfile(pt):
            continue
        model, gs, _, _ = load_graphidyom(pt, device)
        fn = make_graphidyom_proba_fn(model, gs, device)
        out.append((f"GraphIDyOM[{enc}] ord{args.order}", fn, adapters[enc]))
    # Baselines.
    for bname in args.baselines:
        pkl = os.path.join(split_dir, f"baseline_{bname}.pkl")
        if not os.path.isfile(pkl):
            continue
        model, vs, _ = load_baseline(pkl)
        fn = make_baseline_proba_fn(model, vs)
        adapter = adapters["graph_embedding"] if bname == "mc2_ge" else adapters["categorical_id"]
        if bname == "mc2_ge" and "graph_embedding" not in adapters:
            continue
        out.append((bname.upper(), fn, adapter))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panels(panels, pos, net_segments, extent, seq_len, mode, out_path,
                full_network: bool = False):
    """panels: list of (name, true_cont, pred_cont, seed_nodes)."""
    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.4 * nrows),
                             squeeze=False)

    if extent is not None:
        net_xlim = (extent[0], extent[1])
        net_ylim = (extent[2], extent[3])
    else:
        net_xlim = net_ylim = None

    for idx, (name, true_cont, pred_cont, seed) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]

        if net_segments:
            lc = LineCollection(net_segments, colors="#b8b8b8",
                                linewidths=0.18, alpha=0.45, zorder=1,
                                label="road network")
            ax.add_collection(lc)

        seed_xy   = [pos[v] for v in seed if v in pos]
        true_full = [v for v in (seed + true_cont) if v in pos]
        # AR predicted path continues from the seed; next-node predictions are
        # one-step hops from the corresponding true node (drawn separately).
        true_xy = [pos[v] for v in true_full]

        # ground-truth continuation (green)
        if len(true_xy) >= 2:
            tx, ty = zip(*true_xy)
            ax.plot(tx, ty, "-", color="#2ca02c", lw=1.5, alpha=0.9,
                    zorder=3, label="true")
            # Hollow markers keep the line readable on dense segments and make
            # overlap with predicted nodes easier to see.
            ax.scatter(tx, ty, s=10, marker="o", facecolors="none",
                   edgecolors="#2ca02c", linewidths=0.9, zorder=4)

        # predicted continuation (red)
        n_hit = 0
        if mode == "autoregressive":
            pred_full = [pos[v] for v in (seed + pred_cont) if v in pos]
            if len(pred_full) >= 2:
                px, py = zip(*pred_full)
                ax.plot(px, py, "--", color="#d62728", lw=1.5, alpha=0.9,
                        zorder=5, label="predicted")
                # Use x-markers so predicted nodes remain visible even when
                # they coincide with true nodes.
                ax.scatter(px[len(seed_xy):], py[len(seed_xy):], s=14,
                           marker="x", color="#d62728", linewidths=0.9,
                           zorder=6)
        else:  # next-node: arrow from each true current node to predicted node
            for step, pred in enumerate(pred_cont):
                cur = seed[-1] if step == 0 else true_cont[step - 1]
                if cur in pos and pred in pos:
                    x0, y0 = pos[cur]
                    x1, y1 = pos[pred]
                    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                                lw=1.3, alpha=0.85), zorder=5)
            pred_xy = [pos[v] for v in pred_cont if v in pos]
            if pred_xy:
                px, py = zip(*pred_xy)
                ax.scatter(px, py, s=14, marker="x", color="#d62728",
                           linewidths=0.9, zorder=6,
                           label="predicted next")
        n_hit = sum(1 for t, p in zip(true_cont, pred_cont) if t == p)

        # start marker
        if seed_xy:
            ax.scatter([seed_xy[0][0]], [seed_xy[0][1]], s=44, marker="*",
                       color="#1f77b4", zorder=7, label="start")

        acc = n_hit / len(true_cont) if true_cont else 0.0
        ax.set_title(f"{name}\nstep-acc {acc:.0%} ({n_hit}/{len(true_cont)})",
                     fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
        # By default, zoom to the trajectory region so true/pred differences
        # are readable; optionally keep full-network extent for context.
        if full_network and net_xlim and net_ylim:
            mx = (net_xlim[1] - net_xlim[0]) * 0.02 + 1e-6
            my = (net_ylim[1] - net_ylim[0]) * 0.02 + 1e-6
            ax.set_xlim(net_xlim[0] - mx, net_xlim[1] + mx)
            ax.set_ylim(net_ylim[0] - my, net_ylim[1] + my)
        else:
            local_nodes = seed + true_cont + pred_cont
            local_xy = [pos[v] for v in local_nodes if v in pos]
            if local_xy:
                xs = [p[0] for p in local_xy]
                ys = [p[1] for p in local_xy]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                dx = max((xmax - xmin) * 0.20, 1e-5)
                dy = max((ymax - ymin) * 0.20, 1e-5)
                ax.set_xlim(xmin - dx, xmax + dx)
                ax.set_ylim(ymin - dy, ymax + dy)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    # hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Predicted vs true continuation — mode={mode}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default="saved_models")
    p.add_argument("--dataset", default=None,
                   help="Dataset label or index (default: first dataset).")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--order", type=int, default=None,
                   help="GraphIDyOM Markov order to plot (default: largest trained).")
    p.add_argument("--mode", choices=["autoregressive", "next-node"],
                   default="autoregressive")
    p.add_argument("--path-index", type=int, default=None,
                   help="Index into the test path pool (default: longest path).")
    p.add_argument("--max-steps", type=int, default=120,
                   help="Max continuation steps to predict/plot.")
    p.add_argument("--full-network", action="store_true",
                   help="Show the full network extent in each panel instead of "
                        "zooming to the trajectory region.")
    p.add_argument("--baselines", nargs="+", default=None,
                   help="Baseline models to include as panels "
                        "(default: all trained baselines).")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = os.path.join(args.models_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        print(f"ERROR: metadata.json not found in '{args.models_dir}'.")
        return 1
    with open(meta_path) as f:
        meta = json.load(f)

    seq_len = int(meta.get("sequence_length", 10))
    debruijn_order = int(meta.get("debruijn_order", 2))
    orders = meta.get("orders", [2, 5, 10])
    if args.order is None:
        args.order = max(orders)
    if args.baselines is None:
        args.baselines = list(meta.get("baselines",
                                       ["mc2", "mc5p", "mc10p", "mc2_ge", "mc5", "mc10"]))

    # ---- select dataset --------------------------------------------------
    datasets = meta.get("datasets", [])
    if not datasets:
        print("ERROR: no datasets in metadata.")
        return 1
    # default: the US dataset if present, else the first one
    ds = next((d for d in datasets if "US" in str(d.get("label", ""))), datasets[0])
    if args.dataset is not None:
        for d in datasets:
            if str(d.get("label")) == args.dataset or str(d.get("ds_idx")) == str(args.dataset):
                ds = d
                break
    split_dir = os.path.join(ds["ds_dir"], f"split_{args.split}")
    if not os.path.isdir(split_dir):
        # ds_dir may be an absolute path from another machine; rebuild locally.
        split_dir = os.path.join(args.models_dir,
                                 f"ds{ds['ds_idx']}_{ds['label']}", f"split_{args.split}")
    if not os.path.isdir(split_dir):
        print(f"ERROR: split dir not found: {split_dir}")
        return 1
    print(f"Dataset: {ds['label']}  split: {args.split}  order: {args.order}  "
          f"mode: {args.mode}")

    # ---- load artifacts --------------------------------------------------
    vocab_cat = int(ds["vocab_size_cat"])
    with open(os.path.join(split_dir, "node_to_idx.json")) as f:
        node_to_idx = json.load(f)
    with open(os.path.join(split_dir, "test_paths.pkl"), "rb") as f:
        tp = pickle.load(f)
    cat_test = tp["cat_test"]
    node_prior = tp.get("node_prior")
    node_transitions = tp.get("node_transitions")
    node_transitions2 = tp.get("node_transitions2")

    graph_path = ds.get("graph_path")
    pos, net_segments, extent = ({}, [], None)
    neighbors = None
    if graph_path and os.path.isfile(graph_path):
        pos, net_segments, extent = load_node_positions(graph_path, node_to_idx)
        neighbors = load_cat_neighbors(graph_path, node_to_idx)
        print(f"Network: {len(net_segments)} edges drawn as backdrop")
    else:
        print(f"WARNING: graph not found ({graph_path}); network backdrop disabled.")

    adapters = build_adapters(split_dir, vocab_cat, seq_len, debruijn_order,
                              neighbors, node_prior, node_transitions,
                              node_transitions2)

    # ---- pick an example path -------------------------------------------
    if args.path_index is not None:
        if not (0 <= args.path_index < len(cat_test)):
            print(f"ERROR: --path-index out of range [0, {len(cat_test)}).")
            return 1
        path = cat_test[args.path_index]
        chosen = args.path_index
    else:
        # longest path so there is a meaningful continuation to show
        lengths = [len(p) for p in cat_test]
        chosen = int(np.argmax(lengths))
        path = cat_test[chosen]
    if len(path) <= seq_len + 1:
        print(f"ERROR: chosen path (len {len(path)}) too short for seq_len {seq_len}.")
        return 1
    print(f"Example path index {chosen}: length {len(path)} nodes "
          f"(seed {seq_len} + up to {min(len(path)-seq_len, args.max_steps)} steps)")

    # ---- run each model --------------------------------------------------
    predictors = gather_model_predictors(split_dir, args, adapters, device)
    if not predictors:
        print("ERROR: no models found to plot.")
        return 1

    seed = [int(x) for x in path[:seq_len]]
    panels = []
    ar = args.mode == "autoregressive"
    for name, fn, adapter in predictors:
        try:
            true_cont, pred_cont = predict_nodes(fn, adapter, path, seq_len,
                                                 args.max_steps, autoregressive=ar)
        except Exception as exc:  # a broken baseline shouldn't kill the plot
            print(f"  {name:<32} SKIPPED ({type(exc).__name__}: {exc})")
            continue
        panels.append((name, true_cont, pred_cont, seed))
        hit = sum(1 for t, p in zip(true_cont, pred_cont) if t == p)
        print(f"  {name:<32} step-acc {hit}/{len(true_cont)}")

    if not panels:
        print("ERROR: every model failed to produce a prediction.")
        return 1

    out_path = args.output or os.path.join(
        "plot_prediction",
        f"{ds['label']}_split{args.split}_{args.mode}_ord{args.order}.png")
    plot_panels(panels, pos, net_segments, extent, seq_len, args.mode, out_path,
                full_network=args.full_network)
    return 0


if __name__ == "__main__":
    sys.exit(main())
