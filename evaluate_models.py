#!/usr/bin/env python3
"""
Evaluation-only script.  Loads models saved by train_models.py and computes:

  Standard (teacher-forced):
      Acc@1   Acc@3   Acc@5   MRR

  Auto-regressive (closed-loop — top-1 fed back into context each step):
      AR@1    AR@3    AR@5    AR-MRR

Results are written to <output-dir>/ in the same format as train_and_evaluate.py
(unified_results.json, comparison_table.csv, comparison_table.tex + figures).

Usage
-----
    python evaluate_models.py \\
        --models-dir saved_models \\
        --output-dir results_unified

    # evaluate only specific models/splits:
    python evaluate_models.py \\
        --models-dir saved_models \\
        --output-dir results_unified \\
        --splits 0 1 \\
        --no-autoregressive
"""

import argparse
import ast
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)

warnings.filterwarnings("ignore")

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models import GraphIDyOMPredictor
from models.node_decode import (
    NodeSpaceAdapter,
    NodeMetricAccumulator,
    load_kgram_to_id,
    load_node_to_cluster,
    _haversine_km,
)
from train_and_evaluate import (
    TRAIN_FRACTION_DEFAULT,
    eval_autoregressive_node_space,
    make_samples_with_users,
    print_table,
    save_results,
    load_node_coordinates,
)
from plot_results import plot_comparison, plot_boxplots

AR_MAX_PATHS = 30   # max test trajectories for auto-regressive eval
                    # (lowered from 100: AR is closed-loop over every path x order
                    #  x encoding and dominates runtime; override with --ar-max-paths)
MAX_EVAL_WINDOWS = 2000  # default cap on standard teacher-forced windows per encoding
DATASET_FRACTION = 1

# ---------------------------------------------------------------------------
# Algorithm evaluation toggles
# Set to False to SKIP re-evaluating a model; its previous results are kept
# in the output JSON so the tables stay complete.
# ---------------------------------------------------------------------------
EVAL_ALGOS = {
    # Only the GraphIDyOM model changed, so re-evaluate just its variants and
    # keep every baseline's previous results in the output JSON.
    "graphidyom_categorical_id": True,  # categorical_id (baseline)
    "graphidyom_debruijn":       False,  # debruijn (baseline)
    "graphidyom_graph_embedding": False,  # graph_embedding (baseline)
    # Baselines (unchanged — keep previous results)
    
    "akom":  True,  # very slow to train; keep existing artifact
    "cpt":   True,  # very slow to train; keep existing artifact
    "mogen": True,  # very slow to train; keep existing artifact
    "iohmm": True,  # very slow to train; keep existing artifact
    
    "mc2p":  True,  # pure backoff order-2
    "mc5p":  True,  # pure backoff order-5
    "mc10p": True,  # pure backoff order-10

    "mc10_ge": False,  # GE cluster baseline (order-10 entropy-weighted)
    "mc2_ge": False,  # GE cluster baseline (order-1) – keep existing artifact
    "mc5_ge": False,  # GE cluster baseline (order-5) – keep existing artifact

    "mc2":   True,  # entropy-weighted order-2
    "mc5":   True,  # entropy-weighted order-5
    "mc10":  True,  # entropy-weighted order-10
}

# ---------------------------------------------------------------------------
# Network / dataset selector (mirror of RUN_NETWORKS in train_models.py)
# ---------------------------------------------------------------------------
# Global on/off switch per road network.  A trained dataset is evaluated only
# if its network key (matched as ``_<KEY>.`` in the data-path filename, or as a
# suffix of the dataset label) is True here.  Lets you evaluate the experiment
# on a single network (e.g. only US) without editing CLI arguments.  If none of
# the enabled keys match any dataset, all datasets are evaluated (fail-safe).
RUN_NETWORKS = {
    "US": True,
    "DE": True,
    "SG": True,
    "AE": True,
}


def _subsample_indices(n_items: int, fraction: float, seed: int) -> np.ndarray:
    """Deterministically choose a fraction of indices."""
    if fraction >= 1.0 or n_items <= 1:
        return np.arange(n_items, dtype=np.intp)

    n_keep = max(1, int(n_items * fraction))
    if n_keep >= n_items:
        return np.arange(n_items, dtype=np.intp)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_items, size=n_keep, replace=False))


def _subsample_eval_arrays(
    X_test: np.ndarray,
    y_test: np.ndarray,
    fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(y_test) == 0 or fraction >= 1.0:
        return X_test, y_test

    keep_idx = _subsample_indices(len(y_test), fraction, seed)
    return X_test[keep_idx], y_test[keep_idx]


def _subsample_eval_paths(
    test_paths: List[List[int]],
    fraction: float,
    seed: int,
) -> List[List[int]]:
    if not test_paths or fraction >= 1.0:
        return test_paths

    keep_idx = _subsample_indices(len(test_paths), fraction, seed)
    return [test_paths[i] for i in keep_idx]


def _filter_paths_by_id(
    paths: List[List[int]],
    users: List[int],
    path_ids: List[int],
    keep_ids: set,
) -> Tuple[List[List[int]], List[int], List[int]]:
    """Keep only rows whose path_id is in keep_ids (preserving original order)."""
    out_p, out_u, out_i = [], [], []
    for p, u, pid in zip(paths, users, path_ids):
        if int(pid) in keep_ids:
            out_p.append(p)
            out_u.append(int(u))
            out_i.append(int(pid))
    return out_p, out_u, out_i


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_graphidyom(
    path: str,
    device: torch.device,
) -> Tuple[object, dict, int, float]:
    """
    Load a GraphIDyOM checkpoint saved by train_models.py.

    Returns
    -------
    (model, graph_structure, vocab_size, train_time_s)
    """
    try:
        # PyTorch >= 2.6 defaults to weights_only=True; our checkpoints include
        # trusted NumPy metadata (e.g., sparse tables), so force full load.
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions.
        ckpt = torch.load(path, map_location=device)
    vocab_size = ckpt["vocab_size"]
    max_order  = ckpt["max_order"]

    graph_structure: Dict = {}
    ew = ckpt.get("edge_weights")
    if ew is not None:
        graph_structure["edge_weights"] = ew.to(device)
    fos = ckpt.get("first_order_sparse")
    if fos is not None:
        graph_structure["first_order_sparse"] = fos
    oc = ckpt.get("out_counts")
    if oc is not None:
        graph_structure["out_counts"] = oc.to(device)
    # Variable-order k-gram tables (sparse CSR numpy arrays). Restoring these is
    # what makes the LTM genuinely variable-order; without them every order
    # backs off to the 1st-order matrix and DM-MC degenerates to MC1.
    kt = ckpt.get("kgram_tables")
    if kt:
        graph_structure["kgram_tables"] = kt

    model = GraphIDyOMPredictor(vocab_size=vocab_size, max_order=max_order)
    smoothing = ckpt.get("smoothing")
    if smoothing is not None:
        model.smoothing = smoothing
    model = model.to(device).eval()
    train_time_s = float(ckpt.get("train_time_s", float("nan")))
    return model, graph_structure, vocab_size, train_time_s


def make_graphidyom_proba_fn(
    model,
    graph_structure: dict,
    device: torch.device,
):
    """Return a ``predict_proba_fn(context, user_history=None) -> np.ndarray`` closure."""
    def _fn(ctx: List[int], user_history: Optional[List[int]] = None) -> np.ndarray:
        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
        uh_t = uhl_t = None
        if user_history is not None and len(user_history) > 0:
            uh_t  = torch.tensor([list(user_history)], dtype=torch.long, device=device)
            uhl_t = torch.tensor([len(user_history)],  dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ctx_t, graph_structure=graph_structure,
                           user_history=uh_t, user_history_lengths=uhl_t)
        return torch.softmax(logits[0], dim=0).cpu().numpy()
    return _fn


def load_baseline(path: str) -> Tuple[object, int, float]:
    """
    Load a baseline pickled by train_models.py.

    Returns
    -------
    (model, vocab_size, train_time_s)
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["vocab_size"], payload.get("train_time_s", float("nan"))


def make_baseline_proba_fn(model, vocab_size: int):
    """Return a ``predict_proba_fn(context, user_history=None) -> np.ndarray`` closure.

    Baselines have no STM and ignore ``user_history``; it is accepted only so
    the closure is signature-compatible with the GraphIDyOM proba function.
    """
    def _fn(ctx: List[int], user_history: Optional[List[int]] = None) -> np.ndarray:
        return np.asarray(model.predict_proba(ctx, vocab_size=vocab_size), dtype=np.float64)
    return _fn


def _build_history_array(
    user_seq: np.ndarray,
    history_by_user: Dict[int, List[int]],
    hist_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a padded ``(N, hist_nodes)`` history matrix and ``(N,)`` length vector
    for a sequence of owning user ids.  Padding value is ``-1`` (ignored by the
    STM).  Users with no stored history get an all-padding row (length 0).
    """
    n = len(user_seq)
    arr     = np.full((n, hist_nodes), -1, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    for i, u in enumerate(user_seq):
        h = history_by_user.get(int(u))
        if not h:
            continue
        h = list(h)[-hist_nodes:]
        L = len(h)
        arr[i, :L] = h
        lengths[i] = L
    return arr, lengths


# ---------------------------------------------------------------------------
# Road-network neighbors (for AR endpoint hop-distance metric)
# ---------------------------------------------------------------------------

def load_cat_neighbors(graph_path: str,
                       node_to_idx: Dict) -> Optional[Dict[int, set]]:
    """
    Build an undirected adjacency keyed by encoded (categorical) node ID.

    Reads the same JSON road network format used by ``evaluate_baselines.py``
    and ``train_models.py`` (keys ``links`` or ``edges`` with
    ``source``/``target``).  Returns ``None`` if the file cannot be read.

    The mapping is undirected: each edge contributes both directions, which
    matches how a vehicle can traverse the road in either direction for the
    purpose of "how close is the predicted endpoint to the true one".
    """
    if not graph_path or not os.path.isfile(graph_path):
        return None
    try:
        with open(graph_path, "r") as f:
            graph_data = json.load(f)
    except Exception:
        return None

    # node_to_idx may have str or int keys depending on origin
    def _lookup(n):
        if n in node_to_idx:
            return node_to_idx[n]
        s = str(n)
        if s in node_to_idx:
            return node_to_idx[s]
        try:
            i = int(n)
            if i in node_to_idx:
                return node_to_idx[i]
        except (TypeError, ValueError):
            pass
        return None

    edges_key = "links" if "links" in graph_data else "edges"
    neighbors: Dict[int, set] = {}
    for edge in graph_data.get(edges_key, []):
        a = _lookup(edge.get("source"))
        b = _lookup(edge.get("target"))
        if a is None or b is None or a == b:
            continue
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    return neighbors if neighbors else None


def load_cat_outdegree(graph_path: str,
                       node_to_idx: Dict,
                       vocab_size: int) -> Optional[np.ndarray]:
    """
    Directed out-degree (number of distinct successor nodes) per encoded
    categorical node id, read from the same road-network JSON used elsewhere
    (keys ``links``/``edges`` with ``source``/``target``).

    Used to restrict teacher-forced evaluation to *decision* nodes
    (out-degree >= k): nodes where the next move is genuinely ambiguous, as
    opposed to forced single-successor moves that every model predicts
    trivially.  Returns ``None`` if the graph cannot be read.
    """
    if not graph_path or not os.path.isfile(graph_path):
        return None
    try:
        with open(graph_path, "r") as f:
            graph_data = json.load(f)
    except Exception:
        return None

    def _lookup(n):
        if n in node_to_idx:
            return node_to_idx[n]
        s = str(n)
        if s in node_to_idx:
            return node_to_idx[s]
        try:
            i = int(n)
            if i in node_to_idx:
                return node_to_idx[i]
        except (TypeError, ValueError):
            pass
        return None

    edges_key = "links" if "links" in graph_data else "edges"
    succ: Dict[int, set] = {}
    for edge in graph_data.get(edges_key, []):
        a = _lookup(edge.get("source"))
        b = _lookup(edge.get("target"))
        if a is None or b is None or a == b:
            continue
        succ.setdefault(a, set()).add(b)
    if not succ:
        return None
    deg = np.zeros(vocab_size, dtype=np.int64)
    for a, outs in succ.items():
        if 0 <= a < vocab_size:
            deg[a] = len(outs)
    return deg


# ---------------------------------------------------------------------------
# Standard (teacher-forced) evaluation
# ---------------------------------------------------------------------------

def _haversine_km_vec(coords: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised great-circle distance (km) between node-id arrays ``a``/``b``.

    Out-of-range ids or missing coordinates yield ``NaN`` so callers can mask
    them out.  ``coords`` is the ``(vocab, 2)`` ``[lon, lat]`` array (degrees).
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if coords is None:
        return out
    n = coords.shape[0]
    valid = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    if not valid.any():
        return out
    ai, bi = a[valid], b[valid]
    lon1, lat1 = coords[ai, 0], coords[ai, 1]
    lon2, lat2 = coords[bi, 0], coords[bi, 1]
    finite = (np.isfinite(lon1) & np.isfinite(lat1)
              & np.isfinite(lon2) & np.isfinite(lat2))
    R = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    h = (np.sin(dphi / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2)
    d = 2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    d[~finite] = np.nan
    sub = out[valid]
    sub[:] = d
    out[valid] = sub
    return out


def evaluate_standard(
    predict_proba_fn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    vocab_size: int,
    batch_size: int = 512,
    adaptive_batch: bool = True,
    device: Optional[torch.device] = None,
    desc: str = "eval",
    coords: Optional[np.ndarray] = None,
    predictions_out: Optional[dict] = None,
) -> Dict:
    """
    Compute Acc@1, Acc@3, Acc@5, MRR over teacher-forced (X_test, y_test) samples.

    ``predict_proba_fn`` must accept a single context list and return a numpy array,
    OR a numpy matrix of shape (B, V) when batching is not needed.  This function
    calls it sample-by-sample for generality.

    When ``predictions_out`` (a dict with ``"true"``/``"pred"``/``"rank"`` lists)
    is given, the per-sample ground-truth node, top-1 predicted node and rank of
    the true node are appended for offline re-analysis.
    """
    n = len(y_test)
    correct_top1 = correct_top3 = correct_top5 = 0
    rr_sum = 0.0
    nll_sum = 0.0
    brier_sum = 0.0
    ndcg5_sum = 0.0
    dist_km_sum = 0.0
    dist_km_n = 0

    for i in tqdm(range(n), desc=desc, leave=False):
        ctx       = X_test[i].tolist()
        true_next = int(y_test[i])
        proba     = np.asarray(predict_proba_fn(ctx), dtype=np.float64)
        ranked    = np.argsort(proba)[::-1]

        if ranked[0] == true_next:
            correct_top1 += 1
        if true_next in ranked[:3]:
            correct_top3 += 1
        if true_next in ranked[:5]:
            correct_top5 += 1
        pos = np.where(ranked == true_next)[0]
        rank = int(pos[0]) + 1 if len(pos) else (len(proba) + 1)
        if len(pos):
            rr_sum += 1.0 / rank

        # NLL: −log P(true next)
        p_true = max(float(proba[true_next]), 1e-300)
        nll_sum += -np.log(p_true)

        # Brier score: ||P − e_y||^2 = sum(P^2) − 2*P[y] + 1
        brier_sum += float(np.sum(proba ** 2)) - 2.0 * float(proba[true_next]) + 1.0

        # NDCG@5: discount=1/log2(rank+1) if rank<=5
        if rank <= 5:
            ndcg5_sum += 1.0 / np.log2(rank + 1)

        # Physical distance (km) between the top-1 predicted node and the true
        # next node.
        if coords is not None:
            d_km = _haversine_km(coords, int(ranked[0]), true_next)
            if d_km is not None:
                dist_km_sum += d_km
                dist_km_n += 1

        if predictions_out is not None:
            predictions_out["true"].append(true_next)
            predictions_out["pred"].append(int(ranked[0]))
            predictions_out["rank"].append(int(rank))

    out = {
        "accuracy_top1": correct_top1 / n,
        "accuracy_top3": correct_top3 / n,
        "accuracy_top5": correct_top5 / n,
        "mrr":           rr_sum / n,
        "nll":           nll_sum / n,
        "brier":         brier_sum / n,
        "ndcg5":         ndcg5_sum / n,
        "n_samples":     n,
    }
    if dist_km_n > 0:
        out["dist_km"] = dist_km_sum / dist_km_n
    return out


def evaluate_standard_batched(
    model,
    graph_structure: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    vocab_size: int,
    device: torch.device,
    batch_size: int = 512,
    adaptive_batch: bool = True,
    desc: str = "eval",
    user_hist_arr: Optional[np.ndarray] = None,
    user_hist_len: Optional[np.ndarray] = None,
    # --- online IM/STM update ----------------------------------------------
    user_seq_arr: Optional[np.ndarray] = None,
    online_stm_update: bool = False,
    history_dict: Optional[Dict[int, List[int]]] = None,
    hist_nodes: int = 500,
    coords: Optional[np.ndarray] = None,
    predictions_out: Optional[dict] = None,
) -> Dict:
    """
    Batched GPU evaluation for GraphIDyOM (much faster than sample-by-sample).

    When ``user_hist_arr`` (N, H) and ``user_hist_len`` (N,) are provided, the
    matching per-window user history slice is passed to the model so the STM
    can use it.

    When ``online_stm_update=True`` and ``user_seq_arr`` / ``history_dict`` are
    given, windows are sorted by user (stable sort, so intra-user temporal order
    is preserved) and the IM history buffer grows as evaluation proceeds: after
    each prediction the true next node is appended to the user's live history.
    This mirrors IDyOM's intended online-learning behaviour, giving the IM access
    to the user's actual test-time trajectories rather than only the frozen
    pre-eval training-time snapshot.
    """
    if adaptive_batch:
        target_toks = 2_000_000
        safe_bs = max(8, target_toks // max(1, vocab_size))
        batch_size = min(batch_size, safe_bs)

    n = len(y_test)
    correct_top1 = correct_top3 = correct_top5 = 0
    rr_sum = 0.0
    nll_sum = 0.0      # negative log-likelihood: −log P(true next)
    brier_sum = 0.0    # Brier score: ||P − e_y||^2  (lower = better calibrated)
    ndcg5_sum = 0.0    # NDCG@5: discount by 1/log2(rank+1) capped at k=5
    dist_km_sum = 0.0  # great-circle distance (km) of top-1 prediction
    dist_km_n = 0

    # Online IM update: sort windows by user (stable, preserving intra-user
    # temporal order), then after each prediction append the true next node to
    # the user's live history so subsequent windows for that user see it.
    live_history: Optional[Dict[int, List[int]]] = None
    if online_stm_update and user_seq_arr is not None and history_dict is not None:
        sort_idx     = np.argsort(user_seq_arr, kind="stable")
        X_test       = X_test[sort_idx]
        y_test       = y_test[sort_idx]
        user_seq_arr = user_seq_arr[sort_idx]
        live_history = {u: list(h) for u, h in history_dict.items()}
        user_hist_arr = user_hist_len = None   # managed dynamically below

    # Optional STM headroom diagnostic (set env DIAG_STM=1). Measures, on the
    # same eval windows, the LTM-only / STM-only / oracle Acc@1 so we can tell
    # whether the per-user STM carries any top-1 signal the LTM does not already
    # have, before tuning the combination further.
    diag_on = os.environ.get("DIAG_STM", "0") == "1" and (
        user_hist_arr is not None or live_history is not None)
    d_ltm = d_stm_right = d_disagree = d_disag_stm_right = d_disag_ltm_right = 0
    d_support = d_oracle = 0

    # Window-level progress bar: the eval is vectorised over teacher-forced
    # windows (not a per-user Python loop), so we surface progress in *windows*
    # -- the full set of test windows for all eval users -- rather than batch
    # count, which finishes in a handful of near-instant steps and is unreadable.
    _pbar = tqdm(total=n, desc=desc, unit="win", leave=False)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        bseq = torch.tensor(X_test[start:end], dtype=torch.long, device=device)
        btgt = torch.tensor(y_test[start:end], dtype=torch.long, device=device)

        uh_t = uhl_t = None
        if live_history is not None:
            _uh_arr, _uh_len = _build_history_array(
                user_seq_arr[start:end], live_history, hist_nodes)
            uh_t  = torch.tensor(_uh_arr, dtype=torch.long, device=device)
            uhl_t = torch.tensor(_uh_len, dtype=torch.long, device=device)
        elif user_hist_arr is not None:
            uh_t  = torch.tensor(user_hist_arr[start:end], dtype=torch.long, device=device)
            uhl_t = torch.tensor(user_hist_len[start:end], dtype=torch.long, device=device)

        with torch.no_grad():
            if diag_on:
                logits, p_ltm, p_stm, stm_sup = model(
                    bseq, graph_structure=graph_structure,
                    user_history=uh_t, user_history_lengths=uhl_t,
                    return_components=True)
                ltm_arg = p_ltm.argmax(dim=1)
                stm_arg = p_stm.argmax(dim=1)
                has_sup = stm_sup > 0
                ltm_ok  = ltm_arg == btgt
                stm_ok  = stm_arg == btgt
                disagree = has_sup & (ltm_arg != stm_arg)
                d_ltm            += ltm_ok.sum().item()
                d_support        += has_sup.sum().item()
                d_stm_right      += (has_sup & stm_ok).sum().item()
                d_disagree       += disagree.sum().item()
                d_disag_stm_right += (disagree & stm_ok).sum().item()
                d_disag_ltm_right += (disagree & ltm_ok).sum().item()
                # Oracle: right if LTM is right, OR (has support AND STM is right)
                d_oracle += (ltm_ok | (has_sup & stm_ok)).sum().item()
            else:
                logits = model(bseq, graph_structure=graph_structure,
                               user_history=uh_t, user_history_lengths=uhl_t)   # (B, V)

        # Online IM update: append the true next node to each user's live
        # history so future windows for the same user benefit from it.
        if live_history is not None:
            for _u, _y in zip(user_seq_arr[start:end], y_test[start:end]):
                _u = int(_u)
                _h = live_history.setdefault(_u, [])
                _h.append(int(_y))
                if len(_h) > hist_nodes:
                    live_history[_u] = _h[-hist_nodes:]

        _, top5_idx = logits.topk(5, dim=1)
        btgt_exp    = btgt.unsqueeze(1)

        correct_top1 += (top5_idx[:, :1] == btgt_exp).any(dim=1).sum().item()
        correct_top3 += (top5_idx[:, :3] == btgt_exp).any(dim=1).sum().item()
        correct_top5 += (top5_idx         == btgt_exp).any(dim=1).sum().item()

        # Rank of the true label == (#logits strictly greater) + 1.  This avoids
        # a full O(V log V) argsort over the (huge, for high-order De Bruijn)
        # vocabulary and the per-row Python loop, while staying identical across
        # all models so the comparison remains fair.
        true_logit = logits.gather(1, btgt_exp)              # (B, 1)
        rank       = (logits > true_logit).sum(dim=1) + 1    # (B,)
        rank_f     = rank.to(torch.float64)
        rr_sum    += (1.0 / rank_f).sum().item()

        # NLL: logits are already log-probs (output of log-softmax)
        nll_sum   += (-true_logit.squeeze(1).to(torch.float64)).sum().item()

        # Brier score: ||softmax(logits) − one_hot(y)||^2
        probs      = torch.softmax(logits.to(torch.float64), dim=1)
        brier_row  = (probs ** 2).sum(dim=1) - 2.0 * probs.gather(1, btgt_exp).squeeze(1) + 1.0
        brier_sum += brier_row.sum().item()

        # NDCG@5: gain=1 only at true label; discount=1/log2(rank+1) for rank<=5
        in_top5    = rank_f <= 5
        ndcg5_sum += (in_top5 / torch.log2(rank_f + 1.0)).sum().item()

        if predictions_out is not None:
            predictions_out["true"].extend(
                int(x) for x in btgt.detach().cpu().numpy())
            predictions_out["pred"].extend(
                int(x) for x in top5_idx[:, 0].detach().cpu().numpy())
            predictions_out["rank"].extend(
                int(x) for x in rank.detach().cpu().numpy())

        # Physical distance (km) between the top-1 predicted node and the true
        # next node, averaged over windows with known coordinates.
        if coords is not None:
            top1_np = top5_idx[:, 0].detach().cpu().numpy()
            true_np = btgt.detach().cpu().numpy()
            d = _haversine_km_vec(coords, top1_np, true_np)
            finite = np.isfinite(d)
            dist_km_sum += float(d[finite].sum())
            dist_km_n   += int(finite.sum())

        _pbar.update(end - start)
    _pbar.close()

    if diag_on:
        def pct(a, b):
            return (100.0 * a / b) if b else 0.0
        print(f"\n    [DIAG_STM] {desc}  n={n}")
        print(f"      combined Acc@1 : {correct_top1/n:.4f}")
        print(f"      LTM-only Acc@1 : {d_ltm/n:.4f}")
        print(f"      oracle   Acc@1 : {d_oracle/n:.4f}   (LTM right OR supported-STM right)")
        print(f"      rows w/ STM support : {d_support}/{n} ({pct(d_support,n):.1f}%)")
        print(f"      STM argmax right (of supported) : {d_stm_right}/{d_support} ({pct(d_stm_right,d_support):.1f}%)")
        print(f"      LTM!=STM disagreements (supported) : {d_disagree}")
        print(f"        of those, STM right : {d_disag_stm_right} ({pct(d_disag_stm_right,d_disagree):.1f}%)")
        print(f"        of those, LTM right : {d_disag_ltm_right} ({pct(d_disag_ltm_right,d_disagree):.1f}%)")
        net = d_disag_stm_right - d_disag_ltm_right
        print(f"      => max net top-1 gain if STM always overrode on disagreement: {net:+d} ({pct(net,n):+.2f} pts)")

    out = {
        "accuracy_top1": correct_top1 / n,
        "accuracy_top3": correct_top3 / n,
        "accuracy_top5": correct_top5 / n,
        "mrr":           rr_sum / n,
        "nll":           nll_sum / n,
        "brier":         brier_sum / n,
        "ndcg5":         ndcg5_sum / n,
        "n_samples":     n,
    }
    if dist_km_n > 0:
        out["dist_km"] = dist_km_sum / dist_km_n
    return out


# ---------------------------------------------------------------------------
# Node-ID (node-space) teacher-forced evaluation
# ---------------------------------------------------------------------------

def evaluate_standard_node_space(
    predict_proba_enc_fn,
    adapter: NodeSpaceAdapter,
    X_enc: np.ndarray,
    true_nodes: np.ndarray,
    current_nodes: Optional[np.ndarray] = None,
    prev_nodes: Optional[np.ndarray] = None,
    user_hist_list: Optional[List[List[int]]] = None,
    desc: str = "eval",
    coords: Optional[np.ndarray] = None,
    predictions_out: Optional[dict] = None,
) -> Dict:
    """Teacher-forced evaluation in *node-ID* space.

    ``predict_proba_enc_fn(ctx, user_history)`` returns the model's distribution
    over its **encoded** symbol space; the adapter then decodes it to a node-ID
    distribution which is scored against the true next node.  Used for the
    ``debruijn`` and ``graph_embedding`` encodings, whose native prediction is
    not a node ID (``categorical_id`` is already node-native and uses the fast
    batched path).

    ``prev_nodes`` (optional) is the node immediately before ``current_nodes``
    in the context window (i.e. ``X_test_cat[:, -2]`` for graph_embedding).
    When provided, the adapter uses bigram transitions for candidate generation.
    """
    acc = NodeMetricAccumulator(coords=coords, predictions_out=predictions_out)
    n = len(true_nodes)
    for i in tqdm(range(n), desc=desc, unit="win", leave=False):
        ctx = X_enc[i].tolist()
        uh = user_hist_list[i] if user_hist_list is not None else None
        enc_proba = np.asarray(predict_proba_enc_fn(ctx, uh), dtype=np.float64)
        cur = int(current_nodes[i]) if current_nodes is not None else None
        prev = int(prev_nodes[i]) if prev_nodes is not None else None
        cand, probs = adapter.decode(enc_proba, cur, prev)
        acc.add(cand, probs, int(true_nodes[i]))
    return acc.result()


# ---------------------------------------------------------------------------
# Split evaluation orchestrator
# ---------------------------------------------------------------------------

def _save_ar_predictions(
    ar_predictions_all: Dict[str, list],
    split_dir: str,
    split_info: dict,
    args,
    ds_info: Optional[dict],
    graph_path: Optional[str],
    node_to_idx_path: str,
    seq_len: int,
) -> None:
    """Save per-path AR traces (seed/true/pred/ranks/probs) for offline re-analysis.

    Node ids are in the encoded categorical-id space; the saved ``graph_path``
    and ``node_to_idx_path`` let an analysis script recover coordinates / road
    adjacency and recompute any AR metric without re-running evaluation. Each
    trace also stores ``probs`` — the model probability assigned to the true
    next node at every closed-loop step — so the AR NLL can be recomputed
    offline.
    """
    ds_label  = (ds_info or {}).get("label", "dataset")
    split_idx = int(split_info.get("split_idx", 0))
    out_dir   = os.path.join(args.output_dir, ds_label, "ar_predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, f"split{split_idx}.json")
    payload = {
        "meta": {
            "dataset": ds_label,
            "split_idx": split_idx,
            "split_dir": split_dir,
            "graph_path": graph_path,
            "node_to_idx_path": (node_to_idx_path
                                 if os.path.isfile(node_to_idx_path) else None),
            "sequence_length": int(seq_len),
            "accept_acc": float(args.accept_acc),
            "id_space": "encoded_categorical_id",
            "note": ("seed/true/pred are encoded categorical-id node ids; "
                     "recover coordinates/adjacency via graph_path + "
                     "node_to_idx_path to recompute spatial AR metrics."),
        },
        "predictions": ar_predictions_all,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"  [ar-predictions] saved {len(ar_predictions_all)} model trace(s) "
          f"→ {out_path}")


def _save_nn_predictions(
    nn_predictions_all: Dict[str, dict],
    split_dir: str,
    split_info: dict,
    args,
    ds_info: Optional[dict],
    graph_path: Optional[str],
    node_to_idx_path: str,
) -> None:
    """Save per-window next-node (teacher-forced) ground truth / prediction.

    Columnar ``true``/``pred``/``rank`` arrays per model (node ids in the
    encoded categorical-id space).  Saved next to the AR traces so the
    teacher-forced results can be re-scored offline without re-running eval.
    """
    ds_label  = (ds_info or {}).get("label", "dataset")
    split_idx = int(split_info.get("split_idx", 0))
    out_dir   = os.path.join(args.output_dir, ds_label, "nextnode_predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, f"split{split_idx}.json")
    payload = {
        "meta": {
            "dataset": ds_label,
            "split_idx": split_idx,
            "split_dir": split_dir,
            "graph_path": graph_path,
            "node_to_idx_path": (node_to_idx_path
                                 if os.path.isfile(node_to_idx_path) else None),
            "id_space": "encoded_categorical_id",
            "task": "next_node_teacher_forced",
            "fields": ["true", "pred", "rank"],
            "note": ("true/pred are encoded categorical-id node ids, one entry "
                     "per evaluated window; rank is the 1-based rank of the true "
                     "node (-1 = degenerate/no candidates). For online-STM runs "
                     "entries follow per-user evaluation order, not the original "
                     "window order. Recover coordinates/adjacency via graph_path "
                     "+ node_to_idx_path."),
        },
        "predictions": nn_predictions_all,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"  [nextnode-predictions] saved {len(nn_predictions_all)} model "
          f"trace(s) → {out_path}")


def evaluate_split(
    split_dir: str,
    split_info: dict,
    args,
    device: torch.device,
    ds_info: Optional[dict] = None,
) -> Dict[str, Dict]:
    """
    Load all models for one split and return a dict {model_name: metrics}.
    """
    # Per-path auto-regressive traces collected this split, keyed by model name.
    # Saved to disk (when --save-predictions) so AR metrics can be recomputed
    # offline without re-running the full evaluation.
    ar_predictions_all: Dict[str, list] = {}
    # Per-window next-node (teacher-forced) traces: columnar true/pred/rank per
    # model, saved alongside the AR traces for the same offline-reanalysis goal.
    nn_predictions_all: Dict[str, dict] = {}
    active_encodings = split_info.get("active_encodings", args.encodings)
    vocab_size_cat   = split_info["vocab_size_cat"]
    vocab_size_db    = split_info.get("vocab_size_db", 0)
    vocab_size_ge    = split_info.get("vocab_size_ge", 0)
    split_seed       = int(split_info.get("split_seed", split_info.get("split_idx", 0)))

    # Keep an existence/format touchpoint for legacy artifacts; fairness now
    # relies on per-path IDs from test_paths.pkl.
    with np.load(os.path.join(split_dir, "test_data.npz"), allow_pickle=False):
        pass

    with open(os.path.join(split_dir, "test_paths.pkl"), "rb") as f:
        tp = pickle.load(f)
    cat_test = tp["cat_test"]
    db_test  = tp["db_test"]
    ge_test  = tp["ge_test"]

    # Per-path owning user id + per-user encoded STM history buffers.
    cat_test_users = tp.get("cat_test_users", list(range(len(cat_test))))
    db_test_users  = tp.get("db_test_users",  list(range(len(db_test))))
    ge_test_users  = tp.get("ge_test_users",  list(range(len(ge_test))))
    # Stable per-path IDs across encodings (saved by train_models.py).  These
    # are required to guarantee all models are evaluated on the same path pool.
    cat_test_path_ids = tp.get("cat_test_path_ids")
    db_test_path_ids  = tp.get("db_test_path_ids")
    ge_test_path_ids  = tp.get("ge_test_path_ids")
    if cat_test_path_ids is None:
        cat_test_path_ids = list(range(len(cat_test)))
    if db_test_path_ids is None:
        db_test_path_ids = list(range(len(db_test)))
    if ge_test_path_ids is None:
        ge_test_path_ids = list(range(len(ge_test)))
    history_cat: Dict[int, List[int]] = tp.get("history_cat", {})
    history_db:  Dict[int, List[int]] = tp.get("history_db",  {})
    history_ge:  Dict[int, List[int]] = tp.get("history_ge",  {})
    # Training node visit-frequency prior (node popularity) used to decode
    # graph_embedding cluster predictions to specific node ids.
    node_prior = tp.get("node_prior")
    # Training first-order successor counts (data-driven adjacency) used to
    # generate node candidates for graph_embedding decoding.
    node_transitions = tp.get("node_transitions")
    # Training second-order (bigram) successor counts; provides much more
    # specific candidates for the cluster→node decode step, especially
    # benefiting low-order GE models (MC1_GE, ord2) where the cluster
    # prediction alone is not accurate enough.
    node_transitions2 = tp.get("node_transitions2")
    hist_nodes = int(tp.get("user_history_nodes",
                            split_info.get("user_history_nodes", 200)))
    if history_cat:
        print(f"  [history] per-user STM buffers for {len(history_cat)} users "
              f"(<= {hist_nodes} nodes each)")

    # Determine which encodings are actually needed by enabled models and
    # evaluate them on the same test path IDs for maximum fairness.
    required_encodings = set()
    for enc in active_encodings:
        if EVAL_ALGOS.get(f"graphidyom_{enc}", True):
            required_encodings.add(enc)
    if not args.skip_baselines:
        cat_baselines = {"akom", "cpt", "mogen", "iohmm", "mc2", "mc5p", "mc10p", "mc5", "mc10"}
        if any((b in args.baselines) and EVAL_ALGOS.get(b.lower(), True)
               for b in cat_baselines):
            required_encodings.add("categorical_id")
        if any(b.endswith("_ge") and (b in args.baselines) and EVAL_ALGOS.get(b.lower(), True)
               for b in args.baselines):
            required_encodings.add("graph_embedding")

    enc_path_data = {
        "categorical_id": (cat_test, cat_test_users, cat_test_path_ids),
        "debruijn":       (db_test,  db_test_users,  db_test_path_ids),
        "graph_embedding":(ge_test,  ge_test_users,  ge_test_path_ids),
    }
    common_ids = None
    for enc in sorted(required_encodings):
        paths_e, users_e, ids_e = enc_path_data.get(enc, ([], [], []))
        if len(paths_e) == 0:
            continue
        ids_set = set(int(i) for i in ids_e)
        common_ids = ids_set if common_ids is None else (common_ids & ids_set)

    if common_ids is None:
        common_ids = set(int(i) for i in cat_test_path_ids)

    # One single shared path subsample for all models.
    if args.dataset_fraction < 1.0 and len(common_ids) > 0:
        ordered_ids = sorted(common_ids)
        keep_idx = _subsample_indices(len(ordered_ids), args.dataset_fraction, split_seed + 707)
        common_ids = {ordered_ids[i] for i in keep_idx}

    cat_test, cat_test_users, cat_test_path_ids = _filter_paths_by_id(
        cat_test, cat_test_users, cat_test_path_ids, common_ids)
    db_test, db_test_users, db_test_path_ids = _filter_paths_by_id(
        db_test, db_test_users, db_test_path_ids, common_ids)
    ge_test, ge_test_users, ge_test_path_ids = _filter_paths_by_id(
        ge_test, ge_test_users, ge_test_path_ids, common_ids)

    # Rebuild teacher-forced windows from the now-common path pools.
    X_test_cat, y_test_cat, user_test_cat = make_samples_with_users(
        cat_test, cat_test_users, args.sequence_length)
    X_test_db, y_test_db, user_test_db = make_samples_with_users(
        db_test, db_test_users, args.sequence_length)
    X_test_ge, y_test_ge, user_test_ge = make_samples_with_users(
        ge_test, ge_test_users, args.sequence_length)

    print("  [fairness] shared path pool: "
          f"paths={len(common_ids)} "
          f"users={len(set(cat_test_users))} "
          f"cat/db/ge={len(cat_test)}/{len(db_test)}/{len(ge_test)}")

    # --- leakage / triviality diagnostic ----------------------------------
    # "Persistence" baseline: predict next token == last context token.
    # If this is already high for an encoding, that encoding's accuracy is
    # dominated by trivial self-repetition (coarse label space), NOT by the
    # model learning anything -- and is not comparable across encodings.
    def _persistence_acc(X: np.ndarray, y: np.ndarray) -> float:
        if X.size == 0 or y.size == 0:
            return float("nan")
        return float(np.mean(y == X[:, -1]))

    print("  [diagnostic] persistence Acc@1 (predict next == last token): "
          f"cat={_persistence_acc(X_test_cat, y_test_cat):.4f} "
          f"db={_persistence_acc(X_test_db, y_test_db):.4f} "
          f"ge={_persistence_acc(X_test_ge, y_test_ge):.4f}  "
          f"| label-space sizes: "
          f"cat={len(np.unique(y_test_cat)) if y_test_cat.size else 0} "
          f"ge={len(np.unique(y_test_ge)) if y_test_ge.size else 0}")

    # --- categorical-id undirected adjacency for AR endpoint hop metric ----
    neighbors_cat: Optional[Dict[int, set]] = None
    coords_cat: Optional[np.ndarray] = None
    outdeg_cat: Optional[np.ndarray] = None  # directed out-degree per cat-id node
    graph_path = (ds_info or {}).get("graph_path") if ds_info else None
    node_to_idx_path = os.path.join(split_dir, "node_to_idx.json")
    if graph_path and os.path.isfile(node_to_idx_path):
        try:
            with open(node_to_idx_path) as _f:
                raw_n2i = json.load(_f)
            # JSON keys are strings; keep both str and int views for robustness
            node_to_idx: Dict = {}
            for k, v in raw_n2i.items():
                node_to_idx[k] = int(v)
                try:
                    node_to_idx[int(k)] = int(v)
                except (TypeError, ValueError):
                    pass
            neighbors_cat = load_cat_neighbors(graph_path, node_to_idx)
            if neighbors_cat:
                print(f"  [endpoint] loaded neighbors for {len(neighbors_cat)} "
                      f"cat-id nodes from {os.path.basename(graph_path)}")
            outdeg_cat = load_cat_outdegree(graph_path, node_to_idx, vocab_size_cat)
            if outdeg_cat is not None:
                print(f"  [decision-node] road out-degree loaded; "
                      f"{int((outdeg_cat >= 2).sum())}/{vocab_size_cat} nodes "
                      f"have out-degree >= 2")
            # Per-node spatial coordinates ([lon, lat], indexed by encoded cat-id
            # node) for the physical-distance metrics.  node_to_idx maps raw
            # node id -> dense cat-id, so coords align with prediction ids.
            coords_cat = load_node_coordinates(graph_path, node_to_idx,
                                               vocab_size_cat)
            if coords_cat is not None:
                _n_xy = int(np.isfinite(coords_cat).all(axis=1).sum())
                print(f"  [distance] loaded coordinates for {_n_xy} cat-id nodes")
        except Exception as exc:
            print(f"  [endpoint] could not build neighbors: {exc}")

    encoding_test = {
        "categorical_id": (X_test_cat, y_test_cat, cat_test, vocab_size_cat),
        "debruijn":       (X_test_db,  y_test_db,  db_test,  vocab_size_db),
        "graph_embedding":(X_test_ge,  y_test_ge,  ge_test,  vocab_size_ge),
    }
    # Per-encoding owning-user ids + STM history dicts (GraphIDyOM only).
    enc_window_users = {
        "categorical_id": user_test_cat,
        "debruijn":       user_test_db,
        "graph_embedding":user_test_ge,
    }
    enc_path_users = {
        "categorical_id": cat_test_users,
        "debruijn":       db_test_users,
        "graph_embedding":ge_test_users,
    }
    enc_history = {
        "categorical_id": history_cat,
        "debruijn":       history_db,
        "graph_embedding":history_ge,
    }

    results: Dict[str, Dict] = {}
    seq_len = X_test_cat.shape[1] if len(X_test_cat) > 0 else args.sequence_length

    # ---- node-ID decoding adapters ----------------------------------------
    # Every model predicts a *node id*; these adapters convert each encoding's
    # native prediction space back to node ids (categorical = identity,
    # debruijn = last node of the k-gram, graph_embedding = a graph neighbour
    # of the current node within the predicted structural cluster).
    adapters: Dict[str, NodeSpaceAdapter] = {
        "categorical_id": NodeSpaceAdapter("categorical_id", vocab_size_cat, seq_len),
    }
    _k2i_path = os.path.join(split_dir, "kgram_to_id.json")
    if vocab_size_db and os.path.isfile(_k2i_path):
        try:
            adapters["debruijn"] = NodeSpaceAdapter(
                "debruijn", vocab_size_cat, seq_len,
                kgram_to_id=load_kgram_to_id(_k2i_path),
                debruijn_order=getattr(args, "debruijn_order", 2))
        except Exception as exc:
            print(f"  [node-decode] debruijn adapter unavailable: {exc}")
    _n2c_path = os.path.join(split_dir, "node_to_cluster.json")
    if vocab_size_ge and os.path.isfile(_n2c_path):
        try:
            adapters["graph_embedding"] = NodeSpaceAdapter(
                "graph_embedding", vocab_size_cat, seq_len,
                node_to_cluster=load_node_to_cluster(_n2c_path, vocab_size_cat),
                neighbors=neighbors_cat,
                node_prior=node_prior,
                transitions=node_transitions,
                transitions2=node_transitions2)
        except Exception as exc:
            print(f"  [node-decode] graph_embedding adapter unavailable: {exc}")

    # ---- Decision-node restriction (optional, re-eval only) ---------------
    # Restrict teacher-forced (next-node) evaluation to *decision* windows:
    # those whose current node (last context token, in categorical-id space)
    # has road-network out-degree >= --decision-min-outdeg.  Forced
    # single-successor moves are excluded -- every model predicts them
    # trivially, washing out per-user routing preference.  Branch points are
    # exactly where the personalised STM can beat the population model.
    # Applied identically to every model (cat directly, ge via 1:1 alignment,
    # db via the adapter's last-node map) so the comparison stays fair.
    # Auto-regressive eval (over full paths) is deliberately left untouched.
    _thr = int(getattr(args, "decision_min_outdeg", 0) or 0)
    if _thr > 0 and outdeg_cat is not None:
        _deg = outdeg_cat
        _n_cat0, _n_db0, _n_ge0 = len(y_test_cat), len(y_test_db), len(y_test_ge)

        def _curr_deg(curr_nodes):
            cn = np.asarray(curr_nodes, dtype=np.int64)
            d = np.zeros(cn.shape, dtype=np.int64)
            ok = (cn >= 0) & (cn < len(_deg))
            d[ok] = _deg[cn[ok]]
            return d

        # categorical_id: current node = last context token (node-native).
        if _n_cat0 > 0:
            m_cat = _curr_deg(X_test_cat[:, -1]) >= _thr
            X_test_cat, y_test_cat = X_test_cat[m_cat], y_test_cat[m_cat]
            user_test_cat = np.asarray(user_test_cat)[m_cat]
        else:
            m_cat = np.zeros(0, dtype=bool)

        # graph_embedding: windows are 1:1 index-aligned with categorical, so
        # the current node is the same -> reuse the categorical mask exactly.
        if _n_ge0 > 0 and _n_ge0 == _n_cat0:
            X_test_ge, y_test_ge = X_test_ge[m_cat], y_test_ge[m_cat]
            user_test_ge = np.asarray(user_test_ge)[m_cat]
        elif _n_ge0 > 0:
            print("    [decision-node] ge/cat window misalignment "
                  f"({_n_ge0} vs {_n_cat0}) -- ge left unfiltered.")

        # debruijn: map the last token to its cat-id current node via the
        # adapter's k-gram last-node lookup, then filter on out-degree.
        _db_ad = adapters.get("debruijn")
        if _n_db0 > 0 and _db_ad is not None and hasattr(_db_ad, "_id_to_lastnode"):
            m_db = _curr_deg(_db_ad._id_to_lastnode[X_test_db[:, -1]]) >= _thr
            X_test_db, y_test_db = X_test_db[m_db], y_test_db[m_db]
            user_test_db = np.asarray(user_test_db)[m_db]
        elif _n_db0 > 0:
            print("    [decision-node] debruijn adapter unavailable -- "
                  "db left unfiltered.")

        # Rebuild the per-encoding views that reference the filtered windows.
        encoding_test["categorical_id"] = (X_test_cat, y_test_cat, cat_test, vocab_size_cat)
        encoding_test["graph_embedding"] = (X_test_ge, y_test_ge, ge_test, vocab_size_ge)
        encoding_test["debruijn"] = (X_test_db, y_test_db, db_test, vocab_size_db)
        enc_window_users["categorical_id"] = user_test_cat
        enc_window_users["graph_embedding"] = user_test_ge
        enc_window_users["debruijn"] = user_test_db

        print(f"  [decision-node] out-degree >= {_thr}: "
              f"cat {_n_cat0}->{len(y_test_cat)}  "
              f"db {_n_db0}->{len(y_test_db)}  "
              f"ge {_n_ge0}->{len(y_test_ge)}")
    elif _thr > 0:
        print(f"  [decision-node] requested out-degree >= {_thr} but no road "
              f"out-degree available for this split -- scoring all windows.")

    # ---- GraphIDyOM models -------------------------------------------------
    print(f"\n  [eval] GraphIDyOM …")
    for enc in active_encodings:
        toggle_key = f"graphidyom_{enc}"
        if not EVAL_ALGOS.get(toggle_key, True):
            print(f"    graphidyom_{enc} (all orders): disabled by EVAL_ALGOS – skipped.")
            continue
        if enc not in encoding_test:
            continue
        X_te, y_te, test_paths_enc, vs = encoding_test[enc]
        if len(X_te) == 0 or vs == 0:
            continue

        adapter = adapters.get(enc)
        # debruijn / graph_embedding predict in their own encoded label space;
        # decode every prediction to a node id so the metrics are comparable
        # to the categorical (node-native) models.
        node_space = enc in ("debruijn", "graph_embedding") and adapter is not None

        # Build per-user STM history for this encoding (GraphIDyOM only).
        hist_dict   = enc_history.get(enc, {})
        win_users   = enc_window_users.get(enc)
        path_users  = enc_path_users.get(enc)
        uh_arr = uh_len = None
        if hist_dict and win_users is not None and len(win_users) == len(y_te):
            uh_arr, uh_len = _build_history_array(win_users, hist_dict, hist_nodes)
        ar_histories = None
        if hist_dict and path_users is not None and len(path_users) == len(test_paths_enc):
            ar_histories = [hist_dict.get(int(u), []) for u in path_users]

        # Node-ID ground truth (true next node + current node) for node-space
        # encodings.  graph_embedding windows are 1:1 index-aligned with the
        # categorical windows (node->cluster is bijective per position), so the
        # categorical target is the true next node and the last categorical
        # context token is the current node (used to restrict the cluster
        # prediction to graph neighbours).  For debruijn the true next node is
        # the last node of the target k-gram, recovered via the adapter.
        node_true = node_curr = node_prev = None
        if node_space:
            if enc == "graph_embedding":
                if len(y_test_cat) != len(y_te):
                    print(f"    graph_embedding: cat/ge window misalignment "
                          f"({len(y_test_cat)} vs {len(y_te)}) – node scoring off.")
                    node_space = False
                else:
                    node_true = y_test_cat.copy()
                    node_curr = X_test_cat[:, -1].copy()
                    # Second-to-last context node for bigram candidate generation.
                    if X_test_cat.shape[1] >= 2:
                        node_prev = X_test_cat[:, -2].copy()
            else:  # debruijn
                node_true = adapter._id_to_lastnode[y_te]
                node_curr = None

        # Optional cap on standard-eval windows (shared across all orders of
        # this encoding for fairness; AR eval is left untouched).
        win_users_te = list(win_users) if win_users is not None else None
        if args.max_eval_windows and len(y_te) > args.max_eval_windows:
            win_idx = _subsample_indices(len(y_te), args.max_eval_windows / len(y_te),
                                         split_seed + 909)
            X_te, y_te = X_te[win_idx], y_te[win_idx]
            if uh_arr is not None:
                uh_arr, uh_len = uh_arr[win_idx], uh_len[win_idx]
            if win_users_te is not None:
                win_users_te = [win_users_te[i] for i in win_idx]
            if node_true is not None:
                node_true = node_true[win_idx]
            if node_curr is not None:
                node_curr = node_curr[win_idx]
            if node_prev is not None:
                node_prev = node_prev[win_idx]

        # Per-window encoded STM history list for the node-space path.
        uh_list = None
        if node_space and hist_dict and win_users_te is not None \
                and len(win_users_te) == len(y_te):
            uh_list = [hist_dict.get(int(u), []) for u in win_users_te]

        for order in args.orders:
            cfg_name  = f"graphidyom_{enc}_ord{order}"
            pt_path   = os.path.join(split_dir, f"{cfg_name}.pt")
            model_key = f"GraphIDyOM_{enc}_ord{order}"
            if not os.path.exists(pt_path):
                print(f"    {model_key}: checkpoint not found – skipped.")
                continue
            try:
                model, graph_structure, _, ckpt_train_time = load_graphidyom(pt_path, device)
                print(f"    {model_key} …", end="", flush=True)
                t0 = time.time()
                nn_pred = {"true": [], "pred": [], "rank": []} if args.save_predictions else None
                if node_space:
                    proba_enc_fn = make_graphidyom_proba_fn(model, graph_structure, device)
                    metrics = evaluate_standard_node_space(
                        proba_enc_fn, adapter, X_te, node_true,
                        current_nodes=node_curr, prev_nodes=node_prev,
                        user_hist_list=uh_list, desc=model_key,
                        coords=coords_cat, predictions_out=nn_pred)
                else:
                    # STM mode.  By default every encoding uses the *static*
                    # per-user STM buffer (the pickled last-3000-node history),
                    # so the prediction is uniformly "population LTM + that
                    # user's personal STM" across categorical_id / debruijn /
                    # graph_embedding.  The optional online variant (--online-stm)
                    # additionally grows each user's STM with their true next
                    # nodes as evaluation proceeds; it is opt-in for all
                    # encodings so the default comparison stays consistent.
                    _online_stm = args.online_stm and bool(hist_dict)
                    metrics = evaluate_standard_batched(
                        model, graph_structure, X_te, y_te, vs, device,
                        batch_size=args.batch_size,
                        adaptive_batch=True,
                        desc=model_key,
                        user_hist_arr=None if _online_stm else uh_arr,
                        user_hist_len=None if _online_stm else uh_len,
                        user_seq_arr=(np.array(win_users_te, dtype=np.int64)
                                      if win_users_te is not None
                                      and len(win_users_te) == len(y_te) else None),
                        online_stm_update=_online_stm,
                        history_dict=hist_dict if hist_dict else None,
                        hist_nodes=hist_nodes,
                        coords=coords_cat if enc == "categorical_id" else None,
                        predictions_out=nn_pred,
                    )
                metrics["train_time_s"] = ckpt_train_time
                if nn_pred is not None:
                    nn_predictions_all[model_key] = nn_pred
                print(f" Acc@1={metrics['accuracy_top1']:.4f}  "
                      f"IC={metrics['nll']/np.log(2):.3f}bits  "
                      f"MRR={metrics['mrr']:.4f}  ({time.time()-t0:.1f}s)")

                if args.autoregressive and len(cat_test) > 0 and adapter is not None:
                    predict_proba_fn = make_graphidyom_proba_fn(model, graph_structure, device)
                    print(f"    {model_key} AR …", end="", flush=True)
                    ar_t0 = time.time()
                    ar_pred_list = [] if args.save_predictions else None
                    # cat_test is the shared node-id path pool (every encoding
                    # has the same paths in the same order after common-id
                    # filtering); the adapter re-encodes the node context each
                    # closed-loop step into this model's input space.
                    ar = eval_autoregressive_node_space(
                        predict_proba_fn, adapter, cat_test, seq_len,
                        desc=f"AR {model_key}", max_paths=args.ar_max_paths,
                        neighbors=neighbors_cat, user_histories=ar_histories,
                        coords=coords_cat,
                        accept_acc=args.accept_acc,
                        predictions_out=ar_pred_list)
                    print(f" AR@1={ar['ar_accuracy_top1']:.4f}  "
                          f"AR@3={ar['ar_accuracy_top3']:.4f}  "
                          f"AR@5={ar['ar_accuracy_top5']:.4f}  "
                          f"AR-MRR={ar['ar_mrr']:.4f}  ({time.time()-ar_t0:.1f}s)")
                    if 'ar_reliable_horizon_mean' in ar:
                        _ak = f"ar_horizon_acc{int(round(args.accept_acc*100))}_mean"
                        print(f"      horizon(reliable/acc{int(round(args.accept_acc*100))}): "
                              f"{ar['ar_reliable_horizon_mean']:.1f}/"
                              f"{ar.get(_ak, float('nan')):.1f} steps")
                    if 'ar_rel_distance_mean' in ar:
                        print(f"      rel-dist(pred vs truth): "
                              f"{ar['ar_rel_distance_mean']:.3f}  "
                              f"len-ratio={ar.get('ar_pred_path_len_ratio_mean', float('nan')):.3f}")
                    if ar_pred_list is not None:
                        ar_predictions_all[model_key] = ar_pred_list
                    metrics.update(ar)

                results[model_key] = metrics
            except Exception as exc:
                print(f" ERROR: {exc}")

    # ---- Baselines ---------------------------------------------------------
    if not args.skip_baselines:
        print(f"\n  [eval] Baselines …")
        for bname in args.baselines:
            if not EVAL_ALGOS.get(bname.lower(), True):
                print(f"    {bname.upper()}: disabled by EVAL_ALGOS – skipped.")
                continue
            pkl_path  = os.path.join(split_dir, f"baseline_{bname}.pkl")
            model_key = bname.upper()
            if not os.path.exists(pkl_path):
                print(f"    {model_key}: model file not found – skipped.")
                continue
            try:
                model, vs, train_time = load_baseline(pkl_path)
                # *_ge baselines (mc1_ge, mc2_ge, mc10_ge) predict in the
                # graph-embedding cluster space; decode their cluster prediction
                # to a node id (graph neighbour of the current node).  Every
                # other baseline is already node-native (categorical-id).
                node_true = node_curr = node_prev = None
                if bname.endswith("_ge"):
                    adapter   = adapters.get("graph_embedding")
                    X_te, y_te = X_test_ge, y_test_ge
                    paths_te   = cat_test
                    node_space = adapter is not None and len(y_test_cat) == len(y_te)
                    if node_space:
                        node_true = y_test_cat.copy()
                        node_curr = X_test_cat[:, -1].copy()
                        if X_test_cat.shape[1] >= 2:
                            node_prev = X_test_cat[:, -2].copy()
                else:
                    adapter    = adapters.get("categorical_id")
                    X_te, y_te = X_test_cat, y_test_cat
                    paths_te   = cat_test
                    node_space = False
                if len(X_te) == 0:
                    print(f"    {model_key}: no test samples – skipped.")
                    continue
                # Optional cap on standard-eval windows (deterministic; AR
                # eval over paths_te is left untouched).
                if args.max_eval_windows and len(y_te) > args.max_eval_windows:
                    win_idx = _subsample_indices(
                        len(y_te), args.max_eval_windows / len(y_te), split_seed + 909)
                    X_te, y_te = X_te[win_idx], y_te[win_idx]
                    if node_true is not None:
                        node_true = node_true[win_idx]
                    if node_curr is not None:
                        node_curr = node_curr[win_idx]
                    if node_prev is not None:
                        node_prev = node_prev[win_idx]
                print(f"    {model_key} …", end="", flush=True)
                predict_proba_fn = make_baseline_proba_fn(model, vs)
                t0 = time.time()
                nn_pred = {"true": [], "pred": [], "rank": []} if args.save_predictions else None
                if node_space:
                    metrics = evaluate_standard_node_space(
                        predict_proba_fn, adapter, X_te, node_true,
                        current_nodes=node_curr, prev_nodes=node_prev,
                        desc=model_key, coords=coords_cat,
                        predictions_out=nn_pred)
                else:
                    metrics = evaluate_standard(
                        predict_proba_fn, X_te, y_te, vs, desc=model_key,
                        coords=coords_cat, predictions_out=nn_pred)
                metrics["train_time_s"] = train_time
                if nn_pred is not None:
                    nn_predictions_all[model_key] = nn_pred
                print(f" Acc@1={metrics['accuracy_top1']:.4f}  "
                      f"IC={metrics['nll']/np.log(2):.3f}bits  "
                      f"MRR={metrics['mrr']:.4f}  ({time.time()-t0:.1f}s)")

                if args.autoregressive and len(paths_te) > 0 and adapter is not None:
                    print(f"    {model_key} AR …", end="", flush=True)
                    ar_t0 = time.time()
                    ar_pred_list = [] if args.save_predictions else None
                    ar = eval_autoregressive_node_space(
                        predict_proba_fn, adapter, paths_te, seq_len,
                        desc=f"AR {model_key}", max_paths=args.ar_max_paths,
                        neighbors=neighbors_cat, coords=coords_cat,
                        accept_acc=args.accept_acc,
                        predictions_out=ar_pred_list)
                    if ar_pred_list is not None:
                        ar_predictions_all[model_key] = ar_pred_list
                    print(f" AR@1={ar['ar_accuracy_top1']:.4f}  "
                          f"AR@3={ar['ar_accuracy_top3']:.4f}  "
                          f"AR@5={ar['ar_accuracy_top5']:.4f}  "
                          f"AR-MRR={ar['ar_mrr']:.4f}  ({time.time()-ar_t0:.1f}s)")
                    if 'ar_endpoint_hops_median' in ar:
                        print(f"      endpoint: match={ar['ar_endpoint_match']:.3f}  "
                              f"hops(med/mean)={ar['ar_endpoint_hops_median']:.1f}/"
                              f"{ar['ar_endpoint_hops_mean']:.1f}  "
                              f"jaccard={ar['ar_path_jaccard']:.3f}")
                    else:
                        print(f"      endpoint: match={ar['ar_endpoint_match']:.3f}  "
                              f"jaccard={ar['ar_path_jaccard']:.3f}")
                    if 'ar_endpoint_dist_km_median' in ar:
                        print(f"      endpoint dist (km, med/mean): "
                              f"{ar['ar_endpoint_dist_km_median']:.2f}/"
                              f"{ar['ar_endpoint_dist_km_mean']:.2f}")
                    if 'dist_km' in metrics:
                        print(f"      next-node dist (km, mean): "
                              f"{metrics['dist_km']:.2f}")
                    if 'ar_reliable_horizon_mean' in ar:
                        _ak = f"ar_horizon_acc{int(round(args.accept_acc*100))}_mean"
                        print(f"      horizon(reliable/acc{int(round(args.accept_acc*100))}): "
                              f"{ar['ar_reliable_horizon_mean']:.1f}/"
                              f"{ar.get(_ak, float('nan')):.1f} steps")
                    if 'ar_rel_distance_mean' in ar:
                        print(f"      rel-dist(pred vs truth): "
                              f"{ar['ar_rel_distance_mean']:.3f}  "
                              f"len-ratio={ar.get('ar_pred_path_len_ratio_mean', float('nan')):.3f}")
                    metrics.update(ar)

                results[model_key] = metrics
            except Exception as exc:
                print(f" ERROR: {exc}")

    # Persist AR traces for offline metric re-analysis.
    if args.save_predictions and ar_predictions_all:
        _save_ar_predictions(ar_predictions_all, split_dir, split_info, args,
                             ds_info, graph_path, node_to_idx_path, seq_len)
    # Persist next-node (teacher-forced) ground truth / predictions too.
    if args.save_predictions and nn_predictions_all:
        _save_nn_predictions(nn_predictions_all, split_dir, split_info, args,
                             ds_info, graph_path, node_to_idx_path)

    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate models saved by train_models.py (standard + AR@1/3/5/MRR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models-dir", default="saved_models",
                   help="Root directory written by train_models.py.")
    p.add_argument("--output-dir", default="results_unified",
                   help="Directory to write results JSON/CSV/figures.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size for GraphIDyOM forward passes.")
    p.add_argument("--ar-max-paths", type=int, default=AR_MAX_PATHS,
                   help="Maximum number of test trajectories used for AR eval.")
    p.add_argument("--dataset-fraction", type=float, default=DATASET_FRACTION,
                   help="Fraction of the saved evaluation set to score per split.")
    p.add_argument("--max-eval-windows", type=int, default=MAX_EVAL_WINDOWS,
                   help="Cap the number of teacher-forced windows scored in the "
                        "standard eval (deterministic, shared across models for "
                        "fairness). Does not affect AR eval. Use 0 to disable the cap.")
    p.add_argument("--splits", type=int, nargs="+", default=None,
                   help="Which split indices to evaluate (default: all).")
    p.add_argument("--baselines", nargs="+", default=None,
                   help="Which baselines to evaluate (default: all found in models-dir).")
    p.add_argument("--orders", type=int, nargs="+", default=None,
                   help="Which GraphIDyOM orders to evaluate (default: from metadata).")
    p.add_argument("--encodings", nargs="+", default=None,
                   help="Which encodings to evaluate (default: from metadata).")
    p.add_argument("--no-autoregressive", dest="autoregressive", action="store_false")
    p.set_defaults(autoregressive=True)
    p.add_argument("--accept-acc", type=float, default=0.75,
                   help="Acceptable per-prefix top-1 accuracy threshold for the "
                        "AR reliable-prediction-horizon metric (ar_horizon_accNN): "
                        "the longest prediction horizon whose cumulative Acc@1 "
                        "stays at or above this value.")
    p.add_argument("--save-predictions", dest="save_predictions",
                   action="store_true",
                   help="Save per-path AR traces (seed/true/pred/ranks) to "
                        "<output-dir>/<dataset>/ar_predictions/ so metrics can be "
                        "recomputed offline without re-running evaluation (default on).")
    p.add_argument("--no-save-predictions", dest="save_predictions",
                   action="store_false")
    p.set_defaults(save_predictions=True)
    p.add_argument("--online-stm", dest="online_stm", action="store_true",
                   help="Enable online IM history accumulation during evaluation: "
                        "windows are sorted by user and the true next node is appended "
                        "to the IM buffer after each prediction. Disabled by default "
                        "because on coarse encodings (e.g. 64 graph-embedding clusters) "
                        "the IM is rarely more accurate than the LTM when they disagree, "
                        "so growing the IM buffer makes it more confidently wrong and "
                        "slightly reduces Acc@1.")
    p.set_defaults(online_stm=False)
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--decision-min-outdeg", type=int, default=0,
                   help="If > 0, restrict teacher-forced (next-node) evaluation "
                        "to decision nodes whose current node has road-network "
                        "out-degree >= this value (e.g. 2 or 3). Forced "
                        "single-successor moves are excluded. Applied identically "
                        "to every model; auto-regressive eval is unaffected. "
                        "0 = score all windows (default). Use a fresh --output-dir "
                        "and enable the baselines in EVAL_ALGOS for a fair "
                        "decision-node comparison table.")
    p.add_argument("--sequence-length", type=int, default=None,
                   help="Override sequence length (default: from metadata).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not (0.0 < args.dataset_fraction <= 1.0):
        raise ValueError("--dataset-fraction must be in (0, 1].")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.dataset_fraction < 1.0:
        print(f"Evaluating on dataset_fraction={args.dataset_fraction:.3f} of each saved split")

    # ---- load metadata ------------------------------------------------------
    meta_path = os.path.join(args.models_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: metadata.json not found in '{args.models_dir}'.")
        print("  Run train_models.py first.")
        return 1

    with open(meta_path) as f:
        meta = json.load(f)

    # Apply CLI overrides on top of saved metadata
    if args.orders is None:
        args.orders    = meta.get("orders",    [2, 5, 10])
    if args.encodings is None:
        args.encodings = meta.get("encodings", ["categorical_id", "debruijn"])
    if args.baselines is None:
        args.baselines = meta.get("baselines", [])
    if args.sequence_length is None:
        args.sequence_length = meta.get("sequence_length", 10)
    # k-gram order used by the de-Bruijn encoding (needed to decode k-gram
    # predictions back to node ids).
    args.debruijn_order = meta.get("debruijn_order", 2)

    all_splits_flat = meta.get("splits", [])
    if args.splits is not None:
        all_splits_flat = [s for s in all_splits_flat if s["split_idx"] in args.splits]

    if not all_splits_flat:
        print("ERROR: no splits found.  Check --models-dir and --splits.")
        return 1

    # Resolve dataset groupings (new per-dataset block, or backward-compat flat list)
    datasets_meta = meta.get("datasets", [])
    if not datasets_meta:
        datasets_meta = [{"ds_idx": 0, "label": "dataset",
                          "data_path": meta.get("data_paths", [""])[0],
                          "splits": all_splits_flat}]

    if args.splits is not None:
        for ds in datasets_meta:
            ds["splits"] = [s for s in ds["splits"] if s["split_idx"] in args.splits]

    # ---- apply the RUN_NETWORKS global selector -----------------------------
    # Keep only datasets whose network key is enabled in RUN_NETWORKS.  The key
    # is matched as ``_<KEY>.`` in the data-path basename (e.g.
    # worldmove_380_US.csv -> "_US.") or as a suffix of the dataset label
    # (e.g. "..._US").  If nothing matches, fall back to evaluating every dataset.
    _enabled_keys = [k for k, v in RUN_NETWORKS.items() if v]

    def _ds_matches(ds: dict) -> bool:
        _dp = os.path.basename(str(ds.get("data_path", "")))
        _lbl = str(ds.get("label", ""))
        return any(f"_{k}." in _dp or _lbl.endswith(f"_{k}") or _lbl == k
                   for k in _enabled_keys)

    _selected = [ds for ds in datasets_meta if _ds_matches(ds)]
    if _selected:
        datasets_meta = _selected
        print(f"[network-select] RUN_NETWORKS -> "
              f"{[ds.get('label') for ds in datasets_meta]}")
    else:
        print("[network-select] RUN_NETWORKS matched no dataset "
              "-> evaluating all networks.")

    n_datasets = len(datasets_meta)

    # ---- per-dataset evaluation ---------------------------------------------
    # Key: dataset label → per-dataset mean metrics (used for cross-DS average)
    all_ds_means: Dict[str, Dict[str, Dict]] = {}

    for ds_idx, ds_info in enumerate(datasets_meta):
        ds_label  = ds_info.get("label", f"ds{ds_idx}")
        ds_splits = ds_info.get("splits", [])
        if not ds_splits:
            print(f"\n  Dataset {ds_label}: no splits – skipped.")
            continue

        print(f"\n{'#'*60}\n  Dataset {ds_idx+1}/{n_datasets}: {ds_label}\n{'#'*60}")

        ds_split_results: Dict[str, List[Dict]] = {}

        for split_info in ds_splits:
            split_idx = split_info["split_idx"]
            split_dir = split_info["split_dir"]
            print(f"\n{'='*60}\n  Split {split_idx+1}/{len(ds_splits)}"
                  f"  ({split_dir})\n{'='*60}")
            try:
                split_res = evaluate_split(split_dir, split_info, args, device,
                                           ds_info=ds_info)
            except Exception as exc:
                print(f"  Split {split_idx} failed: {exc}")
                continue
            for model_name, metrics in split_res.items():
                ds_split_results.setdefault(model_name, []).append(metrics)

        # Merge existing per-split results for models skipped by EVAL_ALGOS
        ds_out_dir = os.path.join(args.output_dir, ds_label)
        existing_json = os.path.join(ds_out_dir, "all_splits_results.json")
        if os.path.exists(existing_json):
            with open(existing_json) as _f:
                _existing = json.load(_f)
            for _mname, _split_list in _existing.get("per_split_results", {}).items():
                if _mname not in ds_split_results:  # not re-evaluated → keep old
                    ds_split_results[_mname] = _split_list

        if not ds_split_results:
            continue

        # Per-dataset: aggregate over splits, write to <output_dir>/<ds_label>/
        os.makedirs(ds_out_dir, exist_ok=True)
        ds_results = _aggregate(ds_split_results)
        all_ds_means[ds_label] = ds_results
        _write_results(ds_results, ds_split_results, ds_out_dir, meta, args,
                       len(ds_splits), label=ds_label)
        print(f"\n  [{ds_label}] Summary (mean over {len(ds_splits)} split(s)):")
        print_table(ds_results)

    if not all_ds_means:
        print("No results collected – nothing to save.")
        return 1

    # ---- cross-dataset aggregate (average of per-dataset means) -------------
    # Each dataset contributes one averaged data point → equal weight per dataset.
    print(f"\n{'#'*60}\n  Cross-dataset average  "
          f"({len(all_ds_means)} dataset(s))\n{'#'*60}")
    global_results = _aggregate_across_datasets(all_ds_means)
    _write_results(global_results, {}, args.output_dir, meta, args,
                   len(all_ds_means), tag="all_datasets", label="all_datasets")
    print(f"\n  [All datasets – averaged] Summary:")
    print_table(global_results)

    print(f"\n✓  Evaluation complete.  Results in: {args.output_dir}/")
    return 0


# ---------------------------------------------------------------------------
# Helpers called by main
# ---------------------------------------------------------------------------

def _aggregate(split_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Mean (and std) of all numeric metrics across splits."""
    results: Dict[str, Dict] = {}
    for model_name, splits in split_results.items():
        float_keys = [k for k in splits[0] if isinstance(splits[0][k], (int, float))]
        results[model_name] = {
            k: float(np.mean([s[k] for s in splits if k in s])) for k in float_keys
        }
        if len(splits) > 1:
            for k in [
                "accuracy_top1", "accuracy_top3", "accuracy_top5", "mrr",
                "nll", "brier", "ndcg5",
                "ar_accuracy_top1", "ar_accuracy_top3",
                "ar_accuracy_top5", "ar_mrr", "ar_nll",
                "ar_endpoint_match", "ar_path_jaccard",
                "ar_endpoint_hops_mean", "ar_endpoint_hops_median",
                "dist_km", "ar_endpoint_dist_km_mean",
                "ar_endpoint_dist_km_median",
                "ar_reliable_horizon_mean", "ar_reliable_horizon_median",
                "ar_rel_distance_mean", "ar_rel_distance_median",
                "ar_pred_path_len_ratio_mean", "ar_pred_path_len_ratio_median",
            ]:
                if k in results[model_name]:
                    results[model_name][f"{k}_std"] = float(
                        np.std([s[k] for s in splits if k in s]))
    return results


def _aggregate_across_datasets(
    per_dataset_means: Dict[str, Dict[str, Dict]],
) -> Dict[str, Dict]:
    """
    Average per-dataset mean results across datasets.

    Each dataset contributes one averaged data point regardless of how many
    splits it contained, giving equal weight to every dataset.
    """
    all_model_names: set = set()
    for ds_means in per_dataset_means.values():
        all_model_names.update(ds_means.keys())

    results: Dict[str, Dict] = {}
    for model_name in sorted(all_model_names):
        ds_values = [
            ds_means[model_name]
            for ds_means in per_dataset_means.values()
            if model_name in ds_means
        ]
        if not ds_values:
            continue
        # Average only primary metrics, not existing _std keys
        float_keys = [
            k for k in ds_values[0]
            if isinstance(ds_values[0][k], (int, float)) and not k.endswith("_std")
        ]
        results[model_name] = {
            k: float(np.mean([v[k] for v in ds_values if k in v]))
            for k in float_keys
        }
        # Std across datasets (inter-dataset variability)
        if len(ds_values) > 1:
            for k in float_keys:
                vals = [v[k] for v in ds_values if k in v]
                results[model_name][f"{k}_std"] = float(np.std(vals))
    return results


def _write_results(
    results: Dict[str, Dict],
    split_results: Dict[str, List[Dict]],
    out_dir: str,
    meta: dict,
    args,
    n_splits: int,
    tag: str = "",
    label: str = "",
) -> None:
    """Save JSON + CSV + LaTeX + plots for one result block."""
    os.makedirs(out_dir, exist_ok=True)
    out_meta = {
        **meta,
        "n_evaluated_splits": n_splits,
        "autoregressive": args.autoregressive,
        "dataset_fraction": args.dataset_fraction,
    }
    if label:
        out_meta["label"] = label
    fname = f"all_splits_results{'_' + tag if tag else ''}.json"
    all_splits_path = os.path.join(out_dir, fname)
    with open(all_splits_path, "w") as f:
        json.dump({"meta": out_meta,
                   "per_split_results": split_results,
                   "mean_results":      results}, f, indent=2)
    print(f"  Saved → {all_splits_path}")
    save_results(results, out_dir, out_meta)
    plot_comparison(results, out_dir,
                    all_split_results=split_results if split_results else None)
    if split_results and n_splits > 1:
        plot_boxplots(split_results, out_dir)


if __name__ == "__main__":
    sys.exit(main())
