#!/usr/bin/env python3
"""
Diagnose the GraphIDyOM dual-memory blend: how much weight does the model give
to the population LTM vs the per-user STM across predictions, and which memory
is actually more accurate?

For every teacher-forced evaluation window of a trained split this script runs
``GraphIDyOMPredictor.forward_with_explanations`` and records, per window:

  * ``alpha_ltm`` / ``alpha_stm`` -- the entropy-/evidence-gated weights the
    model assigns to each memory,
  * whether the LTM-only argmax, the STM-only argmax and the combined argmax
    each hit the true next node,
  * whether the STM had genuine per-user support for the context.

It then produces four figures and a per-user CSV that answer the question
"is the STM helping, and if so when / for which users?":

  1. ``*_weight_hist.png``    -- distribution of the STM weight (alpha_STM).
  2. ``*_acc_bars.png``       -- overall Acc@1 of LTM-only / STM-only / combined.
  3. ``*_acc_vs_weight.png``  -- Acc@1 of each memory stratified by alpha_STM bin
                                 (does the combined model gain where it leans on
                                 the STM?), with the window count per bin.
  4. ``*_per_user.png``       -- per-user mean alpha_STM vs combined-minus-LTM
                                 Acc@1 gain (which users benefit from the STM).

Usage
-----
    python analyze_ltm_stm_weights.py
    python analyze_ltm_stm_weights.py --split-dir saved_models/ds2_worldmove_539_SG/split_0 --order 10
    python analyze_ltm_stm_weights.py --encoding categorical_id --order 5 --max-windows 50000

The default split is the Singapore (SG) dataset, the categorical-id encoding and
order 10 -- i.e. the headline DM-MC model.
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluate_models import load_graphidyom, _build_history_array
from train_and_evaluate import make_samples_with_users

try:
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    pass


# ---------------------------------------------------------------------------
# Encoding -> stored arrays in test_paths.pkl
# ---------------------------------------------------------------------------
ENCODINGS = {
    "categorical_id": ("cat_test", "cat_test_users", "history_cat"),
    "debruijn":       ("db_test",  "db_test_users",  "history_db"),
    "graph_embedding":("ge_test",  "ge_test_users",  "history_ge"),
}


def _checkpoint_name(encoding: str, order: int) -> str:
    return f"graphidyom_{encoding}_ord{order}.pt"


# ---------------------------------------------------------------------------
# Core sweep over evaluation windows
# ---------------------------------------------------------------------------

def collect_window_records(
    model,
    graph_structure: dict,
    X: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    history_by_user: Dict[int, List[int]],
    hist_nodes: int,
    vocab_size: int,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Run the model over all windows and return per-window diagnostic arrays."""
    n = len(y)
    a_ltm   = np.empty(n, dtype=np.float64)
    a_stm   = np.empty(n, dtype=np.float64)
    ltm_hit = np.empty(n, dtype=bool)
    stm_hit = np.empty(n, dtype=bool)
    cmb_hit = np.empty(n, dtype=bool)
    support = np.empty(n, dtype=bool)        # STM has genuine context match

    # A context with no STM match falls back to a (near-)uniform distribution,
    # whose peak sits at ~1/V.  A clearly higher peak means real per-user
    # evidence, so we flag support when the STM peak exceeds twice uniform.
    support_thr = 2.0 / max(1, vocab_size)

    has_hist = bool(history_by_user)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        bseq = torch.tensor(X[start:end], dtype=torch.long, device=device)
        btgt = y[start:end]

        uh_t = uhl_t = None
        if has_hist:
            uh_arr, uh_len = _build_history_array(U[start:end], history_by_user,
                                                  hist_nodes)
            uh_t  = torch.tensor(uh_arr, dtype=torch.long, device=device)
            uhl_t = torch.tensor(uh_len, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.forward_with_explanations(
                bseq, graph_structure=graph_structure,
                user_history=uh_t, user_history_lengths=uhl_t)

        p_ltm = out["ltm_probs"]
        p_stm = out["stm_probs"]
        logits = out["logits"]

        ltm_arg = p_ltm.argmax(dim=1).cpu().numpy()
        stm_arg = p_stm.argmax(dim=1).cpu().numpy()
        cmb_arg = logits.argmax(dim=1).cpu().numpy()
        stm_peak = p_stm.max(dim=1).values.cpu().numpy()

        sl = slice(start, end)
        a_ltm[sl]   = out["alpha_ltm"].cpu().numpy()
        a_stm[sl]   = out["alpha_stm"].cpu().numpy()
        ltm_hit[sl] = ltm_arg == btgt
        stm_hit[sl] = stm_arg == btgt
        cmb_hit[sl] = cmb_arg == btgt
        support[sl] = stm_peak > support_thr

    return {
        "alpha_ltm": a_ltm,
        "alpha_stm": a_stm,
        "ltm_hit":   ltm_hit,
        "stm_hit":   stm_hit,
        "cmb_hit":   cmb_hit,
        "support":   support,
        "user":      U[:n].astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_weight_hist(rec: Dict[str, np.ndarray], out_path: str, title: str):
    a_stm = rec["alpha_stm"]
    sup = rec["support"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(a_stm, bins=bins, color="0.75", edgecolor="0.4", linewidth=0.4,
            label=f"all windows (n={len(a_stm)})")
    if sup.any():
        ax.hist(a_stm[sup], bins=bins, color="tab:red", alpha=0.65,
                label=f"STM-supported (n={int(sup.sum())})")
    mean_stm = float(np.mean(a_stm))
    ax.axvline(mean_stm, color="k", ls="--", lw=1.2,
               label=f"mean alpha_STM = {mean_stm:.3f}")
    ax.set_xlabel(r"STM weight  $\alpha_{\mathrm{STM}}$  "
                  r"(LTM weight $= 1-\alpha_{\mathrm{STM}}$)")
    ax.set_ylabel("number of windows")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_acc_bars(rec: Dict[str, np.ndarray], out_path: str, title: str):
    sup = rec["support"]
    # STM-only accuracy is only meaningful where the STM has support; on
    # unsupported windows it is a uniform guess.
    overall = {
        "LTM-only":  float(np.mean(rec["ltm_hit"])),
        "STM-only\n(supported)": float(np.mean(rec["stm_hit"][sup])) if sup.any() else 0.0,
        "Combined":  float(np.mean(rec["cmb_hit"])),
    }
    fig, ax = plt.subplots(figsize=(6, 4.2))
    names = list(overall)
    vals = [overall[k] for k in names]
    colors = ["tab:blue", "tab:red", "tab:green"]
    bars = ax.bar(names, vals, color=colors, edgecolor="0.3", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)
    ax.set_ylabel("Acc@1")
    ax.set_ylim(0, max(vals) * 1.18 + 1e-6)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_acc_vs_weight(rec: Dict[str, np.ndarray], out_path: str, title: str,
                      n_bins: int = 10):
    a_stm = rec["alpha_stm"]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(a_stm, edges[1:-1]), 0, n_bins - 1)
    centers, ltm_acc, stm_acc, cmb_acc, counts = [], [], [], [], []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        ltm_acc.append(float(np.mean(rec["ltm_hit"][m])))
        cmb_acc.append(float(np.mean(rec["cmb_hit"][m])))
        sup_m = m & rec["support"]
        stm_acc.append(float(np.mean(rec["stm_hit"][sup_m])) if sup_m.any() else np.nan)
        counts.append(c)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(centers, ltm_acc, "o-", color="tab:blue", label="LTM-only")
    ax.plot(centers, stm_acc, "s--", color="tab:red",
            label="STM-only (supported)")
    ax.plot(centers, cmb_acc, "^-", color="tab:green", label="Combined")
    ax.set_xlabel(r"STM weight bin  $\alpha_{\mathrm{STM}}$")
    ax.set_ylabel("Acc@1")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=(1.0 / n_bins) * 0.85, color="0.85",
            zorder=0, alpha=0.6)
    ax2.set_ylabel("window count", color="0.5")
    ax2.tick_params(axis="y", colors="0.5")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_per_user(rec: Dict[str, np.ndarray], out_path: str, title: str,
                 min_windows: int = 20) -> List[dict]:
    users = rec["user"]
    uniq = np.unique(users)
    rows = []
    for u in uniq:
        m = users == u
        c = int(m.sum())
        if c < min_windows:
            # still record for the CSV, but flag it
            pass
        ltm_a = float(np.mean(rec["ltm_hit"][m]))
        cmb_a = float(np.mean(rec["cmb_hit"][m]))
        rows.append({
            "user": int(u),
            "n_windows": c,
            "mean_alpha_stm": float(np.mean(rec["alpha_stm"][m])),
            "support_frac": float(np.mean(rec["support"][m])),
            "ltm_acc": ltm_a,
            "stm_acc": float(np.mean(rec["stm_hit"][m & rec["support"]]))
                        if (m & rec["support"]).any() else float("nan"),
            "combined_acc": cmb_a,
            "gain": cmb_a - ltm_a,
        })

    plotted = [r for r in rows if r["n_windows"] >= min_windows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if plotted:
        x = np.array([r["mean_alpha_stm"] for r in plotted])
        g = np.array([r["gain"] for r in plotted])
        s = np.array([r["n_windows"] for r in plotted], dtype=float)
        s = 20 + 180 * (s - s.min()) / (s.max() - s.min() + 1e-9)
        col = np.where(g > 0, "tab:green", np.where(g < 0, "tab:red", "0.5"))
        ax.scatter(x, g, s=s, c=col, alpha=0.7, edgecolors="0.3", linewidths=0.4)
        helped = int(np.sum(g > 0))
        hurt = int(np.sum(g < 0))
        ax.text(0.02, 0.97,
                f"users helped: {helped}   hurt: {hurt}   "
                f"neutral: {len(plotted) - helped - hurt}",
                transform=ax.transAxes, va="top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"per-user mean STM weight  $\overline{\alpha_{\mathrm{STM}}}$")
    ax.set_ylabel("combined $-$ LTM-only  Acc@1")
    ax.set_title(title + f"\n(users with >= {min_windows} windows; "
                 "size proportional to #windows)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return rows


def write_csv(rows: List[dict], out_path: str):
    import csv
    if not rows:
        return
    fields = ["user", "n_windows", "mean_alpha_stm", "support_frac",
              "ltm_acc", "stm_acc", "combined_acc", "gain"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["gain"], reverse=True):
            w.writerow(r)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Track LTM vs STM weighting and per-memory accuracy of a "
                    "trained GraphIDyOM split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--split-dir",
                    default="saved_models/ds2_worldmove_539_SG/split_0",
                    help="Trained split directory (holds the .pt checkpoints "
                         "and test_paths.pkl).")
    ap.add_argument("--encoding", default="categorical_id",
                    choices=list(ENCODINGS),
                    help="Which node encoding / checkpoint family to analyse.")
    ap.add_argument("--order", type=int, default=10,
                    help="Markov order K of the checkpoint to load.")
    ap.add_argument("--sequence-length", type=int, default=10,
                    help="Context window length (must match training).")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="Optionally cap the number of evaluation windows "
                         "(random subsample) for speed.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--bins", type=int, default=10,
                    help="Number of alpha_STM bins for the accuracy-vs-weight "
                         "figure.")
    ap.add_argument("--min-user-windows", type=int, default=20,
                    help="Minimum windows for a user to appear in the per-user "
                         "scatter.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default=None,
                    help="Output path prefix (default: "
                         "ltm_stm_weights_<encoding>_ord<order>).")
    args = ap.parse_args()

    split_dir = os.path.join(_REPO, args.split_dir) \
        if not os.path.isabs(args.split_dir) else args.split_dir
    ckpt_path = os.path.join(split_dir, _checkpoint_name(args.encoding, args.order))
    tp_path = os.path.join(split_dir, "test_paths.pkl")
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 1
    if not os.path.isfile(tp_path):
        print(f"ERROR: test_paths.pkl not found: {tp_path}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Loading model {os.path.basename(ckpt_path)} on {device} …")
    model, graph_structure, vocab_size, _ = load_graphidyom(ckpt_path, device)

    print(f"[2] Loading test paths from {tp_path} …")
    with open(tp_path, "rb") as f:
        tp = pickle.load(f)
    paths_key, users_key, hist_key = ENCODINGS[args.encoding]
    test_paths = tp[paths_key]
    test_users = tp.get(users_key, list(range(len(test_paths))))
    history_by_user: Dict[int, List[int]] = tp.get(hist_key, {})
    hist_nodes = int(tp.get("user_history_nodes", 3000))
    print(f"    vocab={vocab_size}  test_paths={len(test_paths)}  "
          f"users_with_history={len(history_by_user)}  hist_nodes<={hist_nodes}")

    X, y, U = make_samples_with_users(test_paths, test_users,
                                      args.sequence_length)
    if len(y) == 0:
        print("ERROR: no evaluation windows could be built.")
        return 1
    print(f"[3] Built {len(y)} teacher-forced windows "
          f"({len(np.unique(U))} users).")

    if args.max_windows is not None and len(y) > args.max_windows:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(y), size=args.max_windows, replace=False)
        keep.sort()
        X, y, U = X[keep], y[keep], U[keep]
        print(f"    subsampled to {len(y)} windows (seed={args.seed}).")

    print("[4] Running forward_with_explanations over all windows …")
    rec = collect_window_records(
        model, graph_structure, X, y, U, history_by_user, hist_nodes,
        vocab_size, device, batch_size=args.batch_size)

    # ---- console summary -------------------------------------------------
    sup = rec["support"]
    print("\n=== summary ===")
    print(f"  windows                 : {len(y)}")
    print(f"  mean alpha_LTM          : {np.mean(rec['alpha_ltm']):.3f}")
    print(f"  mean alpha_STM          : {np.mean(rec['alpha_stm']):.3f}")
    print(f"  STM-supported windows   : {int(sup.sum())} "
          f"({100.0 * np.mean(sup):.1f}%)")
    print(f"  Acc@1 LTM-only          : {np.mean(rec['ltm_hit']):.4f}")
    print(f"  Acc@1 Combined          : {np.mean(rec['cmb_hit']):.4f}")
    if sup.any():
        print(f"  Acc@1 STM-only (support): {np.mean(rec['stm_hit'][sup]):.4f}")
        print(f"  Acc@1 LTM-only (support): {np.mean(rec['ltm_hit'][sup]):.4f}")
        print(f"  Acc@1 Combined (support): {np.mean(rec['cmb_hit'][sup]):.4f}")

    # ---- figures ---------------------------------------------------------
    prefix = args.out_prefix or os.path.join(
        _REPO, f"ltm_stm_weights_{args.encoding}_ord{args.order}")
    tag = f"{args.encoding} ord{args.order}  ({os.path.basename(split_dir)})"
    print("\n[5] Writing figures …")
    fig_weight_hist(rec, prefix + "_weight_hist.png",
                    f"STM vs LTM weighting — {tag}")
    fig_acc_bars(rec, prefix + "_acc_bars.png",
                 f"Per-memory Acc@1 — {tag}")
    fig_acc_vs_weight(rec, prefix + "_acc_vs_weight.png",
                      f"Accuracy vs STM weight — {tag}", n_bins=args.bins)
    rows = fig_per_user(rec, prefix + "_per_user.png",
                        f"Per-user STM benefit — {tag}",
                        min_windows=args.min_user_windows)
    write_csv(rows, prefix + "_per_user.csv")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
