#!/usr/bin/env python3
"""
Unified training & evaluation for GraphIDyOM (pure Markov) and baselines.

GraphIDyOM variants (no gradient descent – building corpus statistics IS training):
  • categorical_id  × Markov orders {2, 5, 10}  →  3 configs
  • debruijn-2gram  × Markov orders {2, 5, 10}  →  3 configs

  NOTE: graph_embedding / node2vec produces continuous embeddings, which are
  incompatible with the discrete Markov chain model and is therefore excluded.

Baseline models (categorical_id, same train/test split):
  • AKOM, CPT+, MOGen, IOHMM

Data split is done ONCE at the trajectory level before any model sees any data:
  train 60 % | val 20 % | test 20 %
  (split by full trajectory, not by sample, to prevent context leakage)

Outputs saved to <output_dir>/:
  unified_results.json    – all accuracy / top-k / MRR numbers
  comparison_table.csv    – easy to paste into paper
  comparison_table.tex    – ready LaTeX table
  comparison.png          – IEEE-style bar chart

Usage
-----
    python train_and_evaluate.py \\
        --data-path  data/worldmove_380_NY_test.csv \\
        --graph-path data/worldmove_380_NY_network.json \\
        --output-dir results_unified
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

warnings.filterwarnings("ignore")

# UTF-8 on Windows
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# Make sure GraphIDyOMo repo root is importable regardless of cwd
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models import GraphIDyOMPredictor
from models.baselines import AKOM, CPTPlus, MOGen, IOHMM, SimpleMarkov
from models.graph_idyom import build_graph_structure_from_sequences
from plot_results import plot_comparison, plot_boxplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPHIDYOM_ENCODINGS = ["categorical_id", "debruijn", "graph_embedding"]
GRAPHIDYOM_ORDERS    = [2, 5, 10]
DEBRUIJN_KGRAM_ORDER = 2        # fixed bigram tokenisation for DeBruijn
# Number of clusters for the graph_embedding discretisation.  Raised from 64 to
# 1024: with ~63k nodes, 64 clusters put ~1000 nodes per cluster, an extreme
# many-to-one bottleneck that discards almost all node identity.  1024 clusters
# (~60 nodes each) keep far more spatial/structural resolution so a predicted
# cluster is much more informative when decoded back to a node.
GRAPH_EMBEDDING_CLUSTERS = 1024
# How node features are computed before k-means clustering:
#   'spatial'    – cluster on (x, y) node coordinates → locally coherent
#                  clusters (geographically adjacent nodes group together).
#                  This is the most informative for next-node prediction and is
#                  the default; requires coordinates in the network JSON.
#   'node2vec'   – cluster on Node2Vec graph embeddings (requires the optional
#                  `node2vec` package; falls back to spatial then structural).
#   'structural' – legacy degree/in-degree/out-degree/PageRank features.
GRAPH_EMBEDDING_METHOD = "spatial"
TRAIN_FRACTION_DEFAULT = 1.0    # fraction of trajectories kept in each split subset
AR_MAX_PATHS = 100              # max number of test paths used for auto-regressive eval

BASELINE_NAMES = ["akom", "cpt", "mogen", "iohmm", "mc2", "mc2p", "mc5p", "mc10p", "mc5", "mc10", "mc1_ge", "mc2_ge", "mc5_ge", "mc10_ge"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_trajectories(data_path: str, sequence_length: int,
                      return_users: bool = False):
    """
    Load CSV and return ``(full_paths, sorted_node_vocab)``.

    Each path is a list of raw node IDs.  Only paths long enough to yield at
    least one (context, target) sample are kept.

    If ``return_users`` is True, also return a parallel list ``user_ids`` (one
    entry per kept path) taken from the CSV's ``user_id`` column.  When that
    column is absent, each trajectory is treated as its own user.
    """
    df = pd.read_csv(data_path)

    if "q_path" in df.columns:
        def _parse(s):
            return [int(x.strip()) for x in str(s).split(",") if x.strip()]
        df["route_nodes"] = df["q_path"].apply(_parse)
    elif "route_taken" in df.columns:
        import ast
        df["route_nodes"] = df["route_taken"].apply(ast.literal_eval)
    else:
        raise ValueError("CSV must have column 'q_path' or 'route_taken'.")

    min_len = sequence_length + 1
    df = df[df["route_nodes"].apply(len) >= min_len]
    print(f"  Loaded {len(df)} trajectories (min length {min_len})")

    full_paths  = df["route_nodes"].tolist()
    node_vocab  = sorted({n for p in full_paths for n in p})
    if return_users:
        if "user_id" in df.columns:
            raw_users = df["user_id"].tolist()
            try:
                user_ids = [int(u) for u in raw_users]
            except (ValueError, TypeError):
                # Non-integer IDs (e.g. 'user_00000') → map to dense ints.
                uid_to_idx = {}
                user_ids = []
                for u in raw_users:
                    if u not in uid_to_idx:
                        uid_to_idx[u] = len(uid_to_idx)
                    user_ids.append(uid_to_idx[u])
        else:
            user_ids = list(range(len(full_paths)))
        return full_paths, node_vocab, user_ids
    return full_paths, node_vocab


def split_trajectories(paths: List[List[int]],
                       test_ratio:  float = 0.20,
                       val_ratio:   float = 0.20,
                       seed:        int   = 42):
    """
    Split by TRAJECTORY (not by sample) to prevent context leakage.

    Returns (train_paths, val_paths, test_paths).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    n        = len(paths)
    n_test   = max(1, int(n * test_ratio))
    n_val    = max(1, int(n * val_ratio))
    n_train  = n - n_test - n_val

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    to_list = lambda idxs: [paths[i] for i in idxs]
    return to_list(train_idx), to_list(val_idx), to_list(test_idx)


def subsample_trajectories(paths: List[List[int]],
                          fraction: float,
                          seed: int) -> List[List[int]]:
    """
    Deterministically subsample a fraction of trajectories.

    Used to reduce training time while preserving the same val/test sets.
    """
    if fraction >= 1.0 or len(paths) <= 1:
        return paths

    n_keep = max(1, int(len(paths) * fraction))
    if n_keep >= len(paths):
        return paths

    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(len(paths), size=n_keep, replace=False))
    return [paths[i] for i in keep_idx]


def split_trajectories_by_user(
    paths:              List[List[int]],
    user_ids:           List[int],
    eval_n_users:       int   = 100,
    user_test_ratio:    float = 0.10,
    user_history_nodes: int   = 3000,
    val_ratio:          float = 0.10,
    seed:               int   = 42,
    min_user_trips:     int   = 3,
) -> Dict:
    """
    User-aware split for the per-user history evaluation protocol.

    For ``eval_n_users`` randomly chosen users (each with at least
    ``min_user_trips`` trips), every user's trips are partitioned into three
    *disjoint* groups:

      * **test trips**  – a fraction ``user_test_ratio`` of the user's trips
        (at least one).  All of these are predicted by every model.
      * **history trips** – *every* remaining (non-test) trip of that user.
        Their concatenated node sequence (truncated to the last
        ``user_history_nodes`` nodes) becomes that user's STM history buffer.
        Using all of the user's own moves makes the STM as personally complete
        as possible while staying strictly user-specific.  These trips are held
        out of the training corpus so the history is genuinely unseen (no
        leakage of the eval user's own data into the population LTM).

    All trips of non-eval users go to the training corpus.

    Returns a dict with keys: ``train_paths``, ``val_paths``, ``test_paths``,
    ``test_user_ids`` (aligned with ``test_paths``), ``history_by_user``
    (``{user_id: [raw node ids]}``), ``eval_user_ids``.
    """
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    by_user: Dict[int, List[int]] = defaultdict(list)
    for i, u in enumerate(user_ids):
        by_user[u].append(i)

    eligible = sorted(u for u, idxs in by_user.items() if len(idxs) >= min_user_trips)
    rng.shuffle(eligible)
    eval_users = eligible[:eval_n_users]

    test_paths:    List[List[int]] = []
    test_user_ids: List[int]       = []
    history_by_user: Dict[int, List[int]] = {}
    held_out: set = set()

    for u in eval_users:
        idxs = list(by_user[u])
        rng.shuffle(idxs)
        n_test = max(1, int(len(idxs) * user_test_ratio))
        test_trip_idxs = idxs[:n_test]
        rest           = idxs[n_test:]

        # STM history = ALL of this user's remaining (non-test) trips,
        # concatenated and truncated to the last user_history_nodes nodes.
        # Every one of these trips is held out of the training corpus, so the
        # user's own data never leaks into the population LTM.
        hist_nodes: List[int] = []
        for j in rest:
            hist_nodes.extend(paths[j])
        history_by_user[u] = hist_nodes[-user_history_nodes:]

        for j in test_trip_idxs:
            test_paths.append(paths[j])
            test_user_ids.append(u)
            held_out.add(j)
        for j in rest:
            held_out.add(j)

    train_all_idx = [i for i in range(len(paths)) if i not in held_out]
    rng.shuffle(train_all_idx)
    n_val     = max(1, int(len(train_all_idx) * val_ratio))
    val_idx   = train_all_idx[:n_val]
    train_idx = train_all_idx[n_val:]

    return {
        "train_paths":     [paths[i] for i in train_idx],
        "val_paths":       [paths[i] for i in val_idx],
        "test_paths":      test_paths,
        "test_user_ids":   test_user_ids,
        "history_by_user": history_by_user,
        "eval_user_ids":   list(eval_users),
    }


def encode_paths_categorical(paths: List[List[int]],
                             node_to_idx: Dict[int, int]) -> List[List[int]]:
    """Remap raw node IDs → dense [0, V) integers."""
    out = []
    for p in paths:
        enc = [node_to_idx[n] for n in p if n in node_to_idx]
        if enc:
            out.append(enc)
    return out


def build_debruijn(cat_train: List[List[int]],
                   kgram_order: int = 2
                   ) -> Tuple[Dict[tuple, int], int]:
    """
    Build a k-gram vocabulary from the training set.

    Returns (kgram_to_id mapping, vocab_size).
    Unknown k-grams encountered at test time are silently skipped.
    """
    kgram_to_id: Dict[tuple, int] = {}
    for path in cat_train:
        for i in range(len(path) - kgram_order + 1):
            kg = tuple(path[i: i + kgram_order])
            if kg not in kgram_to_id:
                kgram_to_id[kg] = len(kgram_to_id)
    return kgram_to_id, len(kgram_to_id)


def apply_debruijn(cat_paths: List[List[int]],
                   kgram_to_id: Dict[tuple, int],
                   kgram_order: int = 2) -> List[List[int]]:
    """Encode categorical paths as k-gram ID sequences."""
    out = []
    for path in cat_paths:
        enc = []
        for i in range(len(path) - kgram_order + 1):
            kg = tuple(path[i: i + kgram_order])
            if kg in kgram_to_id:
                enc.append(kgram_to_id[kg])
        if enc:
            out.append(enc)
    return out


def make_samples(encoded_paths: List[List[int]],
                 sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window extraction of (context, next-node) pairs."""
    X, y = [], []
    for path in encoded_paths:
        for i in range(len(path) - sequence_length):
            X.append(path[i: i + sequence_length])
            y.append(path[i + sequence_length])
    if not X:
        return np.empty((0, sequence_length), dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


def make_samples_with_users(
    encoded_paths:   List[List[int]],
    users_per_path:  List[int],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like :func:`make_samples` but also returns the owning user id of every
    sliding-window sample, so teacher-forced evaluation can attach each
    sample's per-user STM history buffer.
    """
    X, y, U = [], [], []
    for path, uid in zip(encoded_paths, users_per_path):
        for i in range(len(path) - sequence_length):
            X.append(path[i: i + sequence_length])
            y.append(path[i + sequence_length])
            U.append(uid)
    if not X:
        return (np.empty((0, sequence_length), dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int64))
    return (np.array(X, dtype=np.int32),
            np.array(y, dtype=np.int32),
            np.array(U, dtype=np.int64))


# ---------------------------------------------------------------------------
# Auto-regressive evaluation helper
# ---------------------------------------------------------------------------

def _bfs_hops(neighbors: Dict[int, set], src: int, dst: int,
              max_hops: int = 30) -> int:
    """
    Undirected BFS shortest-path hop count between ``src`` and ``dst``.

    Returns ``0`` if equal, ``max_hops + 1`` if unreachable within ``max_hops``
    (so the metric is bounded and aggregations remain finite).
    """
    if src == dst:
        return 0
    if src not in neighbors or dst not in neighbors:
        return max_hops + 1
    from collections import deque
    visited = {src}
    frontier = deque([(src, 0)])
    while frontier:
        node, d = frontier.popleft()
        if d >= max_hops:
            continue
        for nb in neighbors.get(node, ()):
            if nb in visited:
                continue
            if nb == dst:
                return d + 1
            visited.add(nb)
            frontier.append((nb, d + 1))
    return max_hops + 1


def _haversine_km(coords: np.ndarray, a: int, b: int) -> Optional[float]:
    """
    Great-circle distance in kilometres between two nodes.

    ``coords`` is the ``(vocab_size, 2)`` ``[x=lon, y=lat]`` array returned by
    :func:`load_node_coordinates` (degrees).  Returns ``None`` if either node
    id is out of range or has no finite coordinate.
    """
    if coords is None:
        return None
    n = coords.shape[0]
    if not (0 <= a < n and 0 <= b < n):
        return None
    lon1, lat1 = coords[a]
    lon2, lat2 = coords[b]
    if not (np.isfinite(lon1) and np.isfinite(lat1)
            and np.isfinite(lon2) and np.isfinite(lat2)):
        return None
    R = 6371.0088  # mean Earth radius in km
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    h = (np.sin(dphi / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2)
    return float(2.0 * R * np.arcsin(np.sqrt(min(1.0, h))))


def eval_autoregressive(
    predict_proba_fn,
    test_paths: List[List[int]],
    sequence_length: int,
    desc: Optional[str] = None,
    max_paths: Optional[int] = AR_MAX_PATHS,
    neighbors: Optional[Dict[int, set]] = None,
    endpoint_max_hops: int = 30,
    user_histories: Optional[List[List[int]]] = None,
    coords: Optional[np.ndarray] = None,
) -> Dict:
    """
    Auto-regressive (closed-loop) evaluation.

    For each test trajectory:
      1. Seed the context with the first ``sequence_length`` ground-truth nodes.
      2. Call ``predict_proba_fn(context)`` to get a score/probability vector.
      3. Feed **top-1** prediction back into the context (closed loop).
      4. Repeat until the end of the trajectory.

    Reports per-token ranking metrics (AR@1, AR@3, AR@5, AR-MRR) and
    per-trajectory endpoint quality metrics:

      * ``ar_endpoint_match`` – fraction of paths whose final predicted node
        equals the true final node.
      * ``ar_path_jaccard``   – mean Jaccard overlap between the predicted and
        true generated suffix (set of visited node IDs, ignoring order).
      * ``ar_endpoint_hops_mean`` / ``_median`` – BFS shortest-path hop
        distance between predicted and true final nodes on the graph defined
        by ``neighbors`` (capped at ``endpoint_max_hops + 1``).  Only emitted
        when ``neighbors`` is provided and the encoded IDs match the graph
        vocabulary (e.g. categorical-id encoding).

    Parameters
    ----------
    predict_proba_fn : callable(context: List[int]) -> np.ndarray
        Returns a score/probability vector of shape ``(vocab_size,)``.
    test_paths : list of list of int
        Encoded test trajectories (full paths, not sliding-window samples).
    sequence_length : int
        Warm-up / context length.
    neighbors : dict, optional
        Undirected adjacency keyed by encoded node id (encoded id → iterable
        of encoded neighbour ids).  Required for hop-distance metrics.
    endpoint_max_hops : int
        Cap on BFS depth used both as the search limit and as the
        "unreachable" sentinel value (``max_hops + 1``).
    """
    if max_paths is not None and max_paths > 0:
        eval_paths = test_paths[:max_paths]
        eval_histories = (user_histories[:max_paths]
                          if user_histories is not None else None)
    else:
        eval_paths = test_paths
        eval_histories = user_histories

    correct_top1 = correct_top3 = correct_top5 = 0
    rr_sum = 0.0
    total  = 0

    n_paths_scored   = 0
    endpoint_matches = 0
    jaccard_sum      = 0.0
    hop_distances: List[int] = []
    endpoint_dists_km: List[float] = []

    path_iter = tqdm(eval_paths, desc=desc or "auto-regressive", leave=False)
    for p_idx, path in enumerate(path_iter):
        if len(path) <= sequence_length:
            continue
        user_hist = eval_histories[p_idx] if eval_histories is not None else None
        context = list(path[:sequence_length])
        true_suffix: List[int] = list(path[sequence_length:])
        pred_suffix: List[int] = []
        for true_next in true_suffix:
            if user_hist is not None:
                proba = predict_proba_fn(context, user_history=user_hist)
            else:
                proba = predict_proba_fn(context)
            ranked = np.argsort(proba)[::-1]
            pred   = int(ranked[0])           # top-1 fed back into context
            if pred == true_next:
                correct_top1 += 1
            if true_next in ranked[:3]:
                correct_top3 += 1
            if true_next in ranked[:5]:
                correct_top5 += 1
            pos = np.where(ranked == true_next)[0]
            if len(pos):
                rr_sum += 1.0 / (pos[0] + 1)
            total   += 1
            pred_suffix.append(pred)
            context  = context[1:] + [pred]   # feed top-1 prediction back

        # ---- per-path endpoint / overlap metrics ----------------------
        if pred_suffix:
            n_paths_scored += 1
            true_end = int(true_suffix[-1])
            pred_end = int(pred_suffix[-1])
            if pred_end == true_end:
                endpoint_matches += 1
            true_set = set(int(x) for x in true_suffix)
            pred_set = set(int(x) for x in pred_suffix)
            union = true_set | pred_set
            if union:
                jaccard_sum += len(true_set & pred_set) / len(union)
            if neighbors is not None:
                hop_distances.append(
                    _bfs_hops(neighbors, pred_end, true_end, endpoint_max_hops)
                )
            if coords is not None:
                d_km = _haversine_km(coords, pred_end, true_end)
                if d_km is not None:
                    endpoint_dists_km.append(d_km)

    result = {
        "ar_accuracy_top1": correct_top1 / total if total > 0 else 0.0,
        "ar_accuracy_top3": correct_top3 / total if total > 0 else 0.0,
        "ar_accuracy_top5": correct_top5 / total if total > 0 else 0.0,
        "ar_mrr":           rr_sum        / total if total > 0 else 0.0,
        "ar_endpoint_match": (endpoint_matches / n_paths_scored
                              if n_paths_scored > 0 else 0.0),
        "ar_path_jaccard":   (jaccard_sum / n_paths_scored
                              if n_paths_scored > 0 else 0.0),
        "ar_n_tokens":      total,
        "ar_n_paths":       len(eval_paths),
        "ar_n_paths_scored": n_paths_scored,
    }
    if hop_distances:
        result["ar_endpoint_hops_mean"]   = float(np.mean(hop_distances))
        result["ar_endpoint_hops_median"] = float(np.median(hop_distances))
        result["ar_endpoint_hops_cap"]    = int(endpoint_max_hops + 1)
    if endpoint_dists_km:
        result["ar_endpoint_dist_km_mean"]   = float(np.mean(endpoint_dists_km))
        result["ar_endpoint_dist_km_median"] = float(np.median(endpoint_dists_km))
    return result


def eval_autoregressive_node_space(
    predict_proba_enc_fn,
    adapter,
    node_test_paths: List[List[int]],
    sequence_length: int,
    desc: Optional[str] = None,
    max_paths: Optional[int] = AR_MAX_PATHS,
    neighbors: Optional[Dict[int, set]] = None,
    endpoint_max_hops: int = 30,
    user_histories: Optional[List[List[int]]] = None,
    coords: Optional[np.ndarray] = None,
    accept_acc: float = 0.75,
    predictions_out: Optional[List[dict]] = None,
) -> Dict:
    """
    Auto-regressive (closed-loop) evaluation in **node-ID space**.

    Identical in spirit to :func:`eval_autoregressive`, but every model
    prediction is decoded to a raw graph node id before being scored and fed
    back into the context.  This makes the closed-loop metrics directly
    comparable across all encodings (categorical-id, de-Bruijn, graph
    structural clusters): the model always predicts *which node comes next*.

    For each test trajectory (expressed as a sequence of **node ids**, i.e. the
    categorical/identity path shared by every encoding after common-id
    filtering):

      1. Seed a raw node context with the first ``sequence_length`` nodes.
      2. ``adapter.encode_context(node_ctx)`` re-encodes the running node
         context into the model's input space (identity / clusters / k-grams).
      3. ``predict_proba_enc_fn`` returns a distribution over the model's
         encoded vocabulary, which ``adapter.decode`` maps to a distribution
         over candidate **node ids** (using the current node to restrict
         cluster predictions to graph neighbours).
      4. The argmax node is fed back into the context (closed loop).

    Parameters
    ----------
    predict_proba_enc_fn : callable(enc_ctx, user_history=None) -> np.ndarray
        Returns a score/probability vector over the model's *encoded* vocab.
    adapter : NodeSpaceAdapter
        Decoder converting encoded predictions to node-id space and
        re-encoding the running node context.
    node_test_paths : list of list of int
        Test trajectories expressed as raw node ids (categorical paths).
    sequence_length : int
        Warm-up / context length.
    neighbors : dict, optional
        Undirected node adjacency for the endpoint hop-distance metric.
    user_histories : list, optional
        Per-path STM history in the model's *encoded* input space.
    """
    if max_paths is not None and max_paths > 0:
        eval_paths = node_test_paths[:max_paths]
        eval_histories = (user_histories[:max_paths]
                          if user_histories is not None else None)
    else:
        eval_paths = node_test_paths
        eval_histories = user_histories

    correct_top1 = correct_top3 = correct_top5 = 0
    rr_sum = 0.0
    total  = 0

    n_paths_scored   = 0
    endpoint_matches = 0
    jaccard_sum      = 0.0
    hop_distances: List[int] = []
    endpoint_dists_km: List[float] = []
    # New AR diagnostics: per-path reliable-prediction horizon (how many
    # next-node steps stay correct) and relative geographic deviation of the
    # predicted path from the ground-truth path.
    reliable_horizons: List[int] = []   # leading consecutive top-1 hits
    acc_horizons: List[int] = []        # longest prefix with acc >= accept_acc
    rel_distances: List[float] = []     # sum step deviation / true path length
    pred_len_ratios: List[float] = []   # predicted path length / true length
    nll_sum = 0.0                       # closed-loop NLL: −log P(true next)

    def _path_length_km(nodes: List[int]) -> Optional[float]:
        """Great-circle length (km) along a node-id sequence (None if unknown)."""
        if coords is None or len(nodes) < 2:
            return None
        tot = 0.0
        ok = False
        for a, b in zip(nodes, nodes[1:]):
            d = _haversine_km(coords, int(a), int(b))
            if d is not None:
                tot += d
                ok = True
        return tot if ok else None

    path_iter = tqdm(eval_paths, desc=desc or "auto-regressive (node)", leave=False)
    for p_idx, path in enumerate(path_iter):
        if len(path) <= sequence_length:
            continue
        user_hist = eval_histories[p_idx] if eval_histories is not None else None
        node_ctx: List[int] = [int(x) for x in path[:sequence_length]]
        true_suffix: List[int] = [int(x) for x in path[sequence_length:]]
        pred_suffix: List[int] = []
        ranks_this_path: List[int] = []
        probs_this_path: List[float] = []   # P(true next) at each closed-loop step
        seed_last = int(path[sequence_length - 1])
        for true_next in true_suffix:
            enc_ctx = adapter.encode_context(node_ctx)
            cur = node_ctx[-1]
            if enc_ctx is None or len(enc_ctx) == 0:
                # Cannot encode the running context -> fall back to a
                # persistence guess so the metrics stay well defined.
                pred = cur
                rank = endpoint_max_hops + 10  # large -> counts as a miss
                true_p = 0.0
            else:
                enc_proba = predict_proba_enc_fn(enc_ctx, user_history=user_hist)
                cand_nodes, cand_probs = adapter.decode(enc_proba, current_node=cur)
                if cand_nodes.size == 0:
                    pred = cur
                    rank = 10 ** 9
                    true_p = 0.0
                else:
                    pred = int(cand_nodes[int(np.argmax(cand_probs))])
                    mask = cand_nodes == true_next
                    if mask.any():
                        true_p = float(cand_probs[mask][0])
                        rank = int(np.sum(cand_probs > true_p)) + 1
                    else:
                        true_p = 0.0
                        rank = int(cand_nodes.size) + 1
            if rank == 1:
                correct_top1 += 1
            if rank <= 3:
                correct_top3 += 1
            if rank <= 5:
                correct_top5 += 1
            rr_sum += 1.0 / rank
            # NLL of the realised next node (floored like the next-node metric).
            nll_sum += -np.log(max(true_p, 1e-12))
            total  += 1
            ranks_this_path.append(rank)
            probs_this_path.append(true_p)
            pred_suffix.append(pred)
            node_ctx = node_ctx + [pred]   # feed top-1 node prediction back

        if pred_suffix:
            n_paths_scored += 1
            true_end = int(true_suffix[-1])
            pred_end = int(pred_suffix[-1])
            if pred_end == true_end:
                endpoint_matches += 1
            true_set = set(int(x) for x in true_suffix)
            pred_set = set(int(x) for x in pred_suffix)
            union = true_set | pred_set
            if union:
                jaccard_sum += len(true_set & pred_set) / len(union)
            if neighbors is not None:
                hop_distances.append(
                    _bfs_hops(neighbors, pred_end, true_end, endpoint_max_hops)
                )
            if coords is not None:
                d_km = _haversine_km(coords, pred_end, true_end)
                if d_km is not None:
                    endpoint_dists_km.append(d_km)

            # --- reliable-prediction horizon ----------------------------
            # How many leading next-node predictions stay correct (rank == 1)
            # before the first miss: "how far ahead can we trust the model".
            lead = 0
            for r in ranks_this_path:
                if r == 1:
                    lead += 1
                else:
                    break
            reliable_horizons.append(lead)
            # Longest prefix whose cumulative top-1 accuracy stays at or above
            # ``accept_acc`` (a softer "acceptable accuracy" horizon).
            _cum = 0
            _best = 0
            for _i, _r in enumerate(ranks_this_path, start=1):
                if _r == 1:
                    _cum += 1
                if _cum / _i >= accept_acc:
                    _best = _i
            acc_horizons.append(_best)

            # --- relative distance (predicted vs ground-truth path) -----
            # Cumulative geographic deviation of the predicted path from the
            # true path, normalised by the true path's length (0 = perfect).
            if coords is not None:
                true_full = [seed_last] + [int(x) for x in true_suffix]
                pred_full = [seed_last] + [int(x) for x in pred_suffix]
                true_len_km = _path_length_km(true_full)
                pred_len_km = _path_length_km(pred_full)
                if true_len_km is not None and true_len_km > 1e-9:
                    if pred_len_km is not None:
                        pred_len_ratios.append(pred_len_km / true_len_km)
                    devs = [
                        _haversine_km(coords, int(p), int(t))
                        for p, t in zip(pred_suffix, true_suffix)
                    ]
                    devs = [d for d in devs if d is not None]
                    if devs:
                        rel_distances.append(sum(devs) / true_len_km)

            # --- per-path trace (for offline metric re-analysis) --------
            if predictions_out is not None:
                predictions_out.append({
                    "path_index": int(p_idx),
                    "seed": [int(x) for x in path[:sequence_length]],
                    "true": [int(x) for x in true_suffix],
                    "pred": [int(x) for x in pred_suffix],
                    "ranks": [int(r) for r in ranks_this_path],
                    "probs": [float(p) for p in probs_this_path],
                })

    result = {
        "ar_accuracy_top1": correct_top1 / total if total > 0 else 0.0,
        "ar_accuracy_top3": correct_top3 / total if total > 0 else 0.0,
        "ar_accuracy_top5": correct_top5 / total if total > 0 else 0.0,
        "ar_mrr":           rr_sum        / total if total > 0 else 0.0,
        "ar_nll":           nll_sum       / total if total > 0 else 0.0,
        "ar_endpoint_match": (endpoint_matches / n_paths_scored
                              if n_paths_scored > 0 else 0.0),
        "ar_path_jaccard":   (jaccard_sum / n_paths_scored
                              if n_paths_scored > 0 else 0.0),
        "ar_n_tokens":      total,
        "ar_n_paths":       len(eval_paths),
        "ar_n_paths_scored": n_paths_scored,
    }
    if hop_distances:
        result["ar_endpoint_hops_mean"]   = float(np.mean(hop_distances))
        result["ar_endpoint_hops_median"] = float(np.median(hop_distances))
        result["ar_endpoint_hops_cap"]    = int(endpoint_max_hops + 1)
    if endpoint_dists_km:
        result["ar_endpoint_dist_km_mean"]   = float(np.mean(endpoint_dists_km))
        result["ar_endpoint_dist_km_median"] = float(np.median(endpoint_dists_km))
    if reliable_horizons:
        result["ar_reliable_horizon_mean"]   = float(np.mean(reliable_horizons))
        result["ar_reliable_horizon_median"] = float(np.median(reliable_horizons))
        _pct = int(round(accept_acc * 100))
        result[f"ar_horizon_acc{_pct}_mean"]   = float(np.mean(acc_horizons))
        result[f"ar_horizon_acc{_pct}_median"] = float(np.median(acc_horizons))
        result["ar_horizon_accept_acc"]        = float(accept_acc)
    if rel_distances:
        result["ar_rel_distance_mean"]   = float(np.mean(rel_distances))
        result["ar_rel_distance_median"] = float(np.median(rel_distances))
    if pred_len_ratios:
        result["ar_pred_path_len_ratio_mean"]   = float(np.mean(pred_len_ratios))
        result["ar_pred_path_len_ratio_median"] = float(np.median(pred_len_ratios))
    return result


# ---------------------------------------------------------------------------
# GraphIDyOM training & evaluation
# ---------------------------------------------------------------------------

def train_eval_graphidyom(
    train_seqs:  np.ndarray,
    val_seqs:    np.ndarray,
    val_targets: np.ndarray,
    test_seqs:   np.ndarray,
    test_targets: np.ndarray,
    vocab_size:  int,
    order:       int,
    batch_size:  int = 512,
    device:      Optional[torch.device] = None,
    save_path:   Optional[str] = None,
    test_paths:  Optional[List[List[int]]] = None,
    adaptive_eval_batch_size: bool = True,
) -> Dict:
    """
    Train GraphIDyOM (pure Markov: build graph_structure from train_seqs)
    and evaluate on val + test.

    Returns a dict with accuracy_top1/3/5, mrr, val_accuracy, train_time_s.
    If test_paths is provided, also runs auto-regressive evaluation and adds
    ar_accuracy_top1 to the returned dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    print(f"    [train] Building graph structure order={order} …", end="", flush=True)

    # ---- "training": accumulate edge counts + k-gram tables ----
    seqs_t = torch.tensor(train_seqs, dtype=torch.long, device=device)
    graph_structure = build_graph_structure_from_sequences(seqs_t, vocab_size,
                                                           max_order=order)
    train_time = time.time() - t0
    print(f" ({train_time:.1f}s)")

    model = GraphIDyOMPredictor(vocab_size=vocab_size, max_order=order)
    model = model.to(device).eval()

    # Large vocabularies (e.g., debruijn) can blow up GPU memory during
    # batched (B, V) logits computation; adapt batch size conservatively.
    if adaptive_eval_batch_size:
        target_tokens_per_batch = 2_000_000
        safe_batch_size = max(8, target_tokens_per_batch // max(1, vocab_size))
        eval_batch_size = min(batch_size, safe_batch_size)
        if eval_batch_size < batch_size:
            print(f"    [eval]  reducing batch_size {batch_size} -> {eval_batch_size} for vocab={vocab_size}")
    else:
        eval_batch_size = batch_size

    # ---- evaluation helper ------------------------------------------------
    def evaluate(seqs: np.ndarray, targets: np.ndarray, desc: str) -> Dict:
        correct_top1 = correct_top3 = correct_top5 = 0
        rr_sum = 0.0
        total  = len(targets)

        batch_iter = range(0, total, eval_batch_size)
        for start in tqdm(batch_iter, desc=desc, total=(total + eval_batch_size - 1) // eval_batch_size,
                          leave=False):
            end = min(start + eval_batch_size, total)
            bseq = torch.tensor(seqs[start:end],    dtype=torch.long, device=device)
            btgt = torch.tensor(targets[start:end], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(bseq, graph_structure=graph_structure)  # (B, V) log-probs

            # top-k accuracy
            _, top5_idx = logits.topk(5, dim=1)          # (B, 5)
            btgt_exp    = btgt.unsqueeze(1)               # (B, 1)

            correct_top1 += (top5_idx[:, :1] == btgt_exp).any(dim=1).sum().item()
            correct_top3 += (top5_idx[:, :3] == btgt_exp).any(dim=1).sum().item()
            correct_top5 += (top5_idx          == btgt_exp).any(dim=1).sum().item()

            # MRR: rank of the true label in the sorted logit list
            ranks = (logits.argsort(dim=1, descending=True) == btgt_exp).nonzero()
            if ranks.numel() > 0:
                rr_sum += (1.0 / (ranks[:, 1].float() + 1)).sum().item()

        return {
            "accuracy_top1": correct_top1 / total,
            "accuracy_top3": correct_top3 / total,
            "accuracy_top5": correct_top5 / total,
            "mrr":           rr_sum         / total,
            "n_samples":     total,
        }

    # ---- val & test --------------------------------------------------------
    print(f"    [eval]  val/test …", end="", flush=True)
    eval_t0 = time.time()
    val_metrics  = evaluate(val_seqs,  val_targets, desc=f"val ord{order}")
    test_metrics = evaluate(test_seqs, test_targets, desc=f"test ord{order}")
    eval_time = time.time() - eval_t0
    print(f" done ({eval_time:.1f}s) Acc@1={test_metrics['accuracy_top1']:.4f}")

    result = {**test_metrics,
              "val_accuracy": val_metrics["accuracy_top1"],
              "train_time_s": train_time}

    # ---- auto-regressive evaluation ---------------------------------------
    if test_paths is not None and len(test_paths) > 0:
        def _gidyom_predict_proba(ctx: List[int]) -> np.ndarray:
            ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(ctx_t, graph_structure=graph_structure)
            return torch.softmax(logits[0], dim=0).cpu().numpy()

        print(f"    [eval]  auto-regressive …", end="", flush=True)
        ar_t0 = time.time()
        ar_metrics = eval_autoregressive(_gidyom_predict_proba, test_paths,
                                         sequence_length=test_seqs.shape[1],
                                         desc=f"AR ord{order}")
        print(f" done ({time.time() - ar_t0:.1f}s) "
              f"AR@1={ar_metrics['ar_accuracy_top1']:.4f}  "
              f"AR@3={ar_metrics['ar_accuracy_top3']:.4f}  "
              f"AR@5={ar_metrics['ar_accuracy_top5']:.4f}  "
              f"AR-MRR={ar_metrics['ar_mrr']:.4f}")
        result.update(ar_metrics)

    # ---- save model -------------------------------------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ew = graph_structure.get("edge_weights")
        torch.save({
            "edge_weights": ew.cpu() if isinstance(ew, torch.Tensor) else None,
            "first_order_sparse": graph_structure.get("first_order_sparse"),
            "vocab_size":   vocab_size,
            "max_order":    order,
            "smoothing":    model.smoothing,
            "test_metrics": test_metrics,
        }, save_path)

    return result


# ---------------------------------------------------------------------------
# Baseline training & evaluation
# ---------------------------------------------------------------------------

def train_eval_baseline(model_name: str,
                        train_paths: List[List[int]],
                        test_seqs:   np.ndarray,
                        test_targets: np.ndarray,
                        vocab_size:  int,
                        adjacency:   Optional[np.ndarray] = None,
                        max_order:   int = 10,
                        iohmm_states: int = 5,
                        iohmm_iter:  int = 30,
                        test_paths:  Optional[List[List[int]]] = None) -> Dict:
    """Fit a baseline on train_paths and evaluate on test (X, y) samples.

    If test_paths is provided, also runs auto-regressive evaluation and adds
    ar_accuracy_top1 to the returned dict.
    """

    model_map = {
        "akom":  lambda: AKOM(max_order=max_order),
        "cpt":   lambda: CPTPlus(max_context_length=5),
        "mogen": lambda: MOGen(max_order=max_order),
        "iohmm": lambda: IOHMM(n_states=iohmm_states, max_iter=iohmm_iter),
        "mc2":    lambda: SimpleMarkov(order=2,  blend=False),
        "mc5p":   lambda: SimpleMarkov(order=5,  blend=False),
        "mc10p":  lambda: SimpleMarkov(order=10, blend=False),
        "mc1_ge": lambda: SimpleMarkov(order=1),
        "mc2_ge": lambda: SimpleMarkov(order=2),
        "mc5_ge": lambda: SimpleMarkov(order=5),
        "mc10_ge": lambda: SimpleMarkov(order=10),
        "mc5":    lambda: SimpleMarkov(order=5),
        "mc10":   lambda: SimpleMarkov(order=10),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown baseline: {model_name}")

    model = model_map[model_name]()

    t0 = time.time()
    # Adjacency masking is disabled: trajectory edges may not match the road-
    # network JSON, causing masking to route predictions to wrong neighbours.
    # All models learn directly from trajectory statistics.
    if model_name.lower() == "iohmm":
        _iohmm_pbar = tqdm(total=model.max_iter, desc="IOHMM EM", unit="iter", leave=False)
        def _iohmm_cb(em_iter, converged):
            _iohmm_pbar.update(1)
            if converged:
                _iohmm_pbar.set_postfix({"status": f"converged @ iter {em_iter}"})
        model.fit(train_paths, adjacency=None, progress_callback=_iohmm_cb)
        _iohmm_pbar.close()
        train_time = time.time() - t0
        print(f"      [fit] IOHMM done ({train_time:.1f}s)")
    else:
        model.fit(train_paths, adjacency=None)
        train_time = time.time() - t0

    n = len(test_targets)
    correct_top1 = correct_top3 = correct_top5 = 0
    rr_sum = 0.0

    for i in range(n):
        ctx       = test_seqs[i].tolist()
        true_next = int(test_targets[i])
        proba     = model.predict_proba(ctx, vocab_size=vocab_size)
        ranked    = np.argsort(proba)[::-1]

        if ranked[0] == true_next:
            correct_top1 += 1
        if true_next in ranked[:3]:
            correct_top3 += 1
        if true_next in ranked[:5]:
            correct_top5 += 1
        pos = np.where(ranked == true_next)[0]
        if len(pos):
            rr_sum += 1.0 / (pos[0] + 1)

    result = {
        "accuracy_top1": correct_top1 / n,
        "accuracy_top3": correct_top3 / n,
        "accuracy_top5": correct_top5 / n,
        "mrr":           rr_sum / n,
        "n_samples":     n,
        "train_time_s":  train_time,
    }

    # ---- auto-regressive evaluation ---------------------------------------
    if test_paths is not None and len(test_paths) > 0:
        def _baseline_predict_proba(ctx: List[int]) -> np.ndarray:
            return np.asarray(model.predict_proba(ctx, vocab_size=vocab_size),
                              dtype=np.float64)

        print(f"    [eval]  auto-regressive …", end="", flush=True)
        ar_t0 = time.time()
        ar_metrics = eval_autoregressive(_baseline_predict_proba, test_paths,
                                         sequence_length=test_seqs.shape[1],
                                         desc=f"AR {model_name.upper()}")
        print(f" done ({time.time() - ar_t0:.1f}s) "
              f"AR@1={ar_metrics['ar_accuracy_top1']:.4f}  "
              f"AR@3={ar_metrics['ar_accuracy_top3']:.4f}  "
              f"AR@5={ar_metrics['ar_accuracy_top5']:.4f}  "
              f"AR-MRR={ar_metrics['ar_mrr']:.4f}")
        result.update(ar_metrics)

    return result


# ---------------------------------------------------------------------------
# Adjacency matrix (for topology masking in baselines)
# ---------------------------------------------------------------------------

def load_nx_graph(graph_path: str,
                  node_to_idx: Dict[int, int],
                  vocab_size: int) -> Optional[object]:
    """
    Load the road-network JSON and return a NetworkX DiGraph with nodes
    relabelled to dense integers in [0, vocab_size).  Returns None if
    networkx is unavailable or the file cannot be read.
    """
    if not _NX_AVAILABLE:
        return None
    try:
        with open(graph_path) as f:
            g_data = json.load(f)
        G = nx.DiGraph()
        G.add_nodes_from(range(vocab_size))
        edge_key = "links" if "links" in g_data else "edges"
        for edge in g_data.get(edge_key, []):
            s, t = edge.get("source"), edge.get("target")
            if s in node_to_idx and t in node_to_idx:
                G.add_edge(node_to_idx[s], node_to_idx[t])
        return G
    except Exception as exc:
        print(f"  Warning: could not load NetworkX graph: {exc}")
        return None


def load_node_coordinates(graph_path: str,
                          node_to_idx: Dict[int, int],
                          vocab_size: int) -> Optional[np.ndarray]:
    """
    Read per-node spatial coordinates from the road-network JSON.

    The WorldMove network files store nodes as ``{"id", "x", "y", ...}`` where
    ``x`` is longitude and ``y`` is latitude.  Returns a ``(vocab_size, 2)``
    array of ``[x, y]`` indexed by the dense node id, with ``NaN`` for nodes
    that have no coordinate.  Returns ``None`` if the file cannot be read or no
    coordinates are present.
    """
    try:
        with open(graph_path) as f:
            g_data = json.load(f)
    except Exception as exc:
        print(f"  Warning: could not read coordinates from {graph_path}: {exc}")
        return None

    coords = np.full((vocab_size, 2), np.nan, dtype=np.float64)
    found = False
    for nd in g_data.get("nodes", []):
        nid = nd.get("id")
        if nid in node_to_idx and ("x" in nd) and ("y" in nd):
            i = node_to_idx[nid]
            if 0 <= i < vocab_size:
                coords[i, 0] = float(nd["x"])
                coords[i, 1] = float(nd["y"])
                found = True
    return coords if found else None


def _impute_columns(features: np.ndarray) -> np.ndarray:
    """Replace non-finite entries in each column with that column's finite mean."""
    feats = np.asarray(features, dtype=np.float64).copy()
    for c in range(feats.shape[1]):
        col = feats[:, c]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = float(np.nanmean(col[~bad]))
    return feats


def _structural_features(G, vocab_size: int) -> np.ndarray:
    """Degree / in-degree / out-degree / PageRank features (legacy method)."""
    features = np.zeros((vocab_size, 4), dtype=np.float64)
    for v in range(vocab_size):
        if G.has_node(v):
            features[v, 0] = G.degree(v)
            features[v, 1] = G.in_degree(v)
            features[v, 2] = G.out_degree(v)
    try:
        pr = nx.pagerank(G, max_iter=200)
        for v, rank in pr.items():
            if 0 <= v < vocab_size:
                features[v, 3] = rank
    except Exception as exc:
        _ = exc
    return features


def _node2vec_features(G, vocab_size: int, dim: int = 64,
                       seed: int = 42) -> Optional[np.ndarray]:
    """Node2Vec embeddings as clustering features.  ``None`` if unavailable."""
    try:
        from node2vec import Node2Vec
    except ImportError:
        return None
    try:
        n2v = Node2Vec(G, dimensions=dim, walk_length=40, num_walks=10,
                       workers=4, seed=seed, quiet=True)
        model = n2v.fit(window=10, min_count=1)
    except Exception as exc:
        print(f"  Warning: Node2Vec embedding failed ({exc}).")
        return None
    emb = np.zeros((vocab_size, dim), dtype=np.float64)
    for i in range(vocab_size):
        key = str(i)
        if key in model.wv:
            emb[i] = model.wv[key]
    return emb


def _make_clusterer(n_clusters: int, n_samples: int, seed: int = 42):
    """Pick KMeans (small) or MiniBatchKMeans (large) for scalability."""
    from sklearn.cluster import KMeans, MiniBatchKMeans
    if n_clusters > 256 or n_samples > 20_000:
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                               batch_size=max(1024, 3 * n_clusters),
                               n_init=3, max_iter=200)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")


def build_graph_structural_encoding(G, vocab_size: int,
                                    n_clusters: int = GRAPH_EMBEDDING_CLUSTERS,
                                    method: str = GRAPH_EMBEDDING_METHOD,
                                    coords: Optional[np.ndarray] = None,
                                    seed: int = 42) -> Tuple[np.ndarray, int]:
    """
    Discretise nodes into clusters for the ``graph_embedding`` encoding.

    Parameters
    ----------
    G : networkx.DiGraph
        Road network (nodes relabelled to ``[0, vocab_size)``).
    vocab_size : int
        Number of distinct nodes.
    n_clusters : int
        Target number of k-means clusters (= graph_embedding vocab size).
    method : {'spatial', 'node2vec', 'structural'}
        Feature space used for clustering:
          * ``spatial``    – (x, y) node coordinates → geographically coherent
            clusters (recommended; needs ``coords``).
          * ``node2vec``   – Node2Vec graph embeddings (optional dependency).
          * ``structural`` – degree / in-degree / out-degree / PageRank (legacy).
        Each method gracefully falls back to a simpler one when its inputs are
        unavailable.
    coords : np.ndarray, optional
        ``(vocab_size, 2)`` array of node coordinates for ``method='spatial'``.

    Returns
    -------
    (node_to_cluster, actual_n_clusters)
        ``node_to_cluster`` is an ``int32`` array of shape ``(vocab_size,)``.
    """
    from sklearn.preprocessing import StandardScaler

    method = (method or "spatial").lower()
    used_method = method
    features: Optional[np.ndarray] = None

    if method == "spatial":
        if coords is not None and np.isfinite(coords).any():
            features = _impute_columns(coords)
        else:
            print("  [graph_embedding] spatial requested but no coordinates "
                  "available → falling back to structural features.")
            used_method = "structural"
    elif method == "node2vec":
        emb = _node2vec_features(G, vocab_size, seed=seed)
        if emb is not None:
            features = emb
        elif coords is not None and np.isfinite(coords).any():
            print("  [graph_embedding] node2vec unavailable → falling back to "
                  "spatial coordinates.")
            used_method = "spatial"
            features = _impute_columns(coords)
        else:
            print("  [graph_embedding] node2vec & coordinates unavailable → "
                  "falling back to structural features.")
            used_method = "structural"
    elif method != "structural":
        print(f"  [graph_embedding] unknown method '{method}' → using structural.")
        used_method = "structural"

    if features is None:
        used_method = "structural"
        features = _structural_features(G, vocab_size)

    features_scaled = StandardScaler().fit_transform(features)
    actual_k = min(n_clusters, vocab_size)
    clusterer = _make_clusterer(actual_k, len(features_scaled), seed)
    node_to_cluster = clusterer.fit_predict(features_scaled).astype(np.int32)
    print(f"  [graph_embedding] method={used_method} clusters={actual_k} "
          f"nodes={vocab_size} (~{vocab_size / max(actual_k, 1):.0f} nodes/cluster)")
    return node_to_cluster, actual_k


def apply_graph_structural_encoding(cat_paths: List[List[int]],
                                    node_to_cluster: np.ndarray) -> List[List[int]]:
    """Map already-categorical paths to graph-structural cluster IDs."""
    out = []
    for path in cat_paths:
        enc = [int(node_to_cluster[n]) for n in path]
        if enc:
            out.append(enc)
    return out


def build_adjacency(graph_path: str,
                    node_to_idx: Dict[int, int],
                    vocab_size: int) -> np.ndarray:
    adj = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    with open(graph_path) as f:
        g = json.load(f)
    for edge in g.get("links" if "links" in g else "edges", []):
        s = edge.get("source")
        t = edge.get("target")
        if s in node_to_idx and t in node_to_idx:
            adj[node_to_idx[s], node_to_idx[t]] = 1.0
    return adj


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_table(results: Dict[str, Dict]) -> None:
    w = max(len(k) for k in results) + 2
    has_ar = any("ar_accuracy_top1" in r for r in results.values())
    ar_col = f" {'AR@1':>8}" if has_ar else ""
    header = f"{'Model':<{w}} {'Acc@1':>8} {'Acc@3':>8} {'Acc@5':>8} {'MRR':>8} {'Train(s)':>9}{ar_col}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in sorted(results.items()):
        ar_val = (f" {r.get('ar_accuracy_top1', float('nan')):>8.4f}"
                  if has_ar else "")
        print(
            f"{name:<{w}} "
            f"{r.get('accuracy_top1', float('nan')):>8.4f} "
            f"{r.get('accuracy_top3', float('nan')):>8.4f} "
            f"{r.get('accuracy_top5', float('nan')):>8.4f} "
            f"{r.get('mrr',           float('nan')):>8.4f} "
            f"{r.get('train_time_s',  float('nan')):>9.1f}"
            f"{ar_val}"
        )
    print("=" * len(header))


def save_results(results: Dict[str, Dict], output_dir: str,
                 meta: Dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    payload = {"timestamp": ts, "meta": meta, "results": results}
    with open(os.path.join(output_dir, "unified_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # CSV
    rows = []
    for name, r in sorted(results.items()):
        row = {
            "Model":      name,
            "Acc@1":      r.get("accuracy_top1"),
            "Acc@3":      r.get("accuracy_top3"),
            "Acc@5":      r.get("accuracy_top5"),
            "MRR":        r.get("mrr"),
            "Train(s)":   r.get("train_time_s"),
            "N_test":     r.get("n_samples"),
        }
        if "ar_accuracy_top1" in r:
            row["AR@1"]   = r["ar_accuracy_top1"]
            row["AR@3"]   = r.get("ar_accuracy_top3")
            row["AR@5"]   = r.get("ar_accuracy_top5")
            row["AR-MRR"] = r.get("ar_mrr")
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    # LaTeX
    tex_path = os.path.join(output_dir, "comparison_table.tex")
    df_tex = df.drop(columns=["Train(s)", "N_test"], errors="ignore")
    with open(tex_path, "w") as f:
        f.write(df_tex.to_latex(index=False, float_format="%.4f",
                                caption="Next-node prediction comparison",
                                label="tab:comparison"))

    print(f"\n  Saved → {csv_path}")
    print(f"  Saved → {tex_path}")
    print(f"  Saved → {os.path.join(output_dir, 'unified_results.json')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified GraphIDyOM + baseline training & evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path",  default="data/worldmove_380_US.csv")
    p.add_argument("--graph-path", default="data/worldmove_380_US_network.json")
    p.add_argument("--output-dir", default="results_unified")
    p.add_argument("--sequence-length", type=int, default=10)
    p.add_argument("--test-ratio",  type=float, default=0.20,
                   help="Fraction of trajectories held out as test set.")
    p.add_argument("--val-ratio",   type=float, default=0.20,
                   help="Fraction of trajectories held out as validation set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION_DEFAULT,
                   help="Fraction of trajectories kept for train/val/test in each split (0,1].")
    p.add_argument("--batch-size",  type=int, default=512,
                   help="Batch size for GraphIDyOM forward passes.")
    # GraphIDyOM
    p.add_argument("--orders", type=int, nargs="+", default=GRAPHIDYOM_ORDERS,
                   help="Markov orders to train GraphIDyOM with.")
    p.add_argument("--encodings", nargs="+", default=GRAPHIDYOM_ENCODINGS,
                   choices=["categorical_id", "debruijn", "graph_embedding"],
                   help="Node encoding strategies.")
    p.add_argument("--debruijn-order", type=int, default=DEBRUIJN_KGRAM_ORDER,
                   help="k for k-gram tokenisation in DeBruijn encoding.")
    p.add_argument("--graph-embedding-clusters", type=int,
                   default=GRAPH_EMBEDDING_CLUSTERS,
                   help="Number of k-means clusters for graph_embedding encoding.")
    p.add_argument("--graph-embedding-method", default=GRAPH_EMBEDDING_METHOD,
                   choices=["spatial", "node2vec", "structural"],
                   help="Feature space for graph_embedding clustering: spatial "
                        "(x/y coordinates, locally coherent), node2vec (graph "
                        "embeddings), or structural (degree/PageRank, legacy).")
    # Baselines
    p.add_argument("--baselines", nargs="+", default=BASELINE_NAMES,
                   choices=BASELINE_NAMES,
                   help="Which baselines to train and evaluate.")
    p.add_argument("--no-adjacency", action="store_true",
                   help="Disable topology masking for baselines.")
    p.add_argument("--akom-max-order",  type=int, default=10)
    p.add_argument("--iohmm-states",    type=int, default=10)
    p.add_argument("--iohmm-iter",      type=int, default=50)
    p.add_argument("--skip-baselines",  action="store_true",
                   help="Skip baseline evaluation (faster).")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Repeated random splits for robust evaluation; generates box plots.")
    p.add_argument("--no-autoregressive", dest="autoregressive", action="store_false",
                   help="Disable auto-regressive (closed-loop) evaluation.")
    p.set_defaults(autoregressive=True)
    p.add_argument("--data-paths", nargs="+", default=[
                       "data/worldmove_380_US.csv",
                       "data/worldmove_431_DE.csv",
                       "data/worldmove_539_SG.csv",
                       "data/worldmove_609_AE.csv",
                   ],
                   help="Multiple CSV paths for multi-dataset evaluation "
                        "(overrides --data-path when given).")
    p.add_argument("--graph-paths", nargs="+", default=[
                       "data/worldmove_380_US_network.json",
                       "data/worldmove_431_DE_network.json",
                       "data/worldmove_539_SG_network.json",
                       "data/worldmove_609_AE_network.json",
                   ],
                   help="Network JSON paths corresponding to --data-paths "
                        "(same order; last entry repeated if fewer given).")
    return p.parse_args()


def _run_one_dataset(
    args,
    data_path: str,
    graph_path: str,
    device,
    all_split_results: Dict[str, List[Dict]],
    dataset_label: str,
) -> Tuple[int, int, int]:
    """Run all n_splits for one dataset, accumulating results into all_split_results.

    Returns (vocab_size_cat, vocab_size_db, vocab_size_ge).
    """
    print("\n[1/4] Loading data \u2026")
    full_paths, node_vocab = load_trajectories(data_path, args.sequence_length)
    node_to_idx    = {n: i for i, n in enumerate(node_vocab)}
    vocab_size_cat = len(node_vocab)
    n_ge_clusters  = args.graph_embedding_clusters
    print(f"  Vocabulary size (categorical): {vocab_size_cat}")
    print(f"  Running {args.n_splits} split(s) (seed base={args.seed}) \u2026")
    if args.train_fraction < 1.0:
        print(f"  Using train_fraction={args.train_fraction:.3f} on train/val/test")

    vocab_size_db = vocab_size_ge = 0

    for split_idx in tqdm(range(args.n_splits), desc=f"{dataset_label} splits", leave=False):
        split_seed = args.seed + split_idx
        sep = "=" * 60
        print(f"\n{sep}\n  Split {split_idx + 1}/{args.n_splits}  (seed={split_seed})\n{sep}")

        # ------------------------------------------------------------ Split
        train_paths, val_paths, test_paths = split_trajectories(
            full_paths, args.test_ratio, args.val_ratio, split_seed)
        if args.train_fraction < 1.0:
            n_train_full = len(train_paths)
            n_val_full = len(val_paths)
            n_test_full = len(test_paths)
            train_paths = subsample_trajectories(
                train_paths, args.train_fraction, seed=split_seed + 10_000)
            val_paths = subsample_trajectories(
                val_paths, args.train_fraction, seed=split_seed + 20_000)
            test_paths = subsample_trajectories(
                test_paths, args.train_fraction, seed=split_seed + 30_000)
            print(
                "  split subsample "
                f"train={len(train_paths)}/{n_train_full} "
                f"val={len(val_paths)}/{n_val_full} "
                f"test={len(test_paths)}/{n_test_full}"
            )
        print(f"  train={len(train_paths)}  val={len(val_paths)}  "
              f"test={len(test_paths)} trajectories")

        # Categorical encoding
        cat_train = encode_paths_categorical(train_paths, node_to_idx)
        cat_val   = encode_paths_categorical(val_paths,   node_to_idx)
        cat_test  = encode_paths_categorical(test_paths,  node_to_idx)
        X_train_cat, _          = make_samples(cat_train, args.sequence_length)
        X_val_cat,   y_val_cat  = make_samples(cat_val,   args.sequence_length)
        X_test_cat,  y_test_cat = make_samples(cat_test,  args.sequence_length)
        print(f"  Samples \u2192 train={len(X_train_cat)}  val={len(X_val_cat)}  "
              f"test={len(X_test_cat)}")

        # DeBruijn encoding (vocab built from train only)
        kgram_to_id, vocab_size_db = build_debruijn(cat_train, args.debruijn_order)
        db_train = apply_debruijn(cat_train, kgram_to_id, args.debruijn_order)
        db_val   = apply_debruijn(cat_val,   kgram_to_id, args.debruijn_order)
        db_test  = apply_debruijn(cat_test,  kgram_to_id, args.debruijn_order)
        X_train_db, _          = make_samples(db_train, args.sequence_length)
        X_val_db,   y_val_db   = make_samples(db_val,   args.sequence_length)
        X_test_db,  y_test_db  = make_samples(db_test,  args.sequence_length)

        # Graph-structural encoding
        vocab_size_ge = 0
        X_train_ge = X_val_ge = y_val_ge = X_test_ge = y_test_ge = np.empty((0,))
        active_encodings = list(args.encodings)
        if "graph_embedding" in active_encodings:
            nx_graph = load_nx_graph(graph_path, node_to_idx, vocab_size_cat)
            if nx_graph is None:
                print("  graph_embedding skipped: NetworkX graph not available.")
                active_encodings = [e for e in active_encodings if e != "graph_embedding"]
            else:
                ge_method = getattr(args, "graph_embedding_method",
                                    GRAPH_EMBEDDING_METHOD)
                ge_coords = (load_node_coordinates(graph_path, node_to_idx,
                                                   vocab_size_cat)
                             if ge_method == "spatial" else None)
                print(f"  Building graph_embedding ({n_ge_clusters} clusters, "
                      f"method={ge_method}) \u2026", flush=True)
                node_to_cluster, vocab_size_ge = build_graph_structural_encoding(
                    nx_graph, vocab_size_cat, n_ge_clusters,
                    method=ge_method, coords=ge_coords)
                ge_train = apply_graph_structural_encoding(cat_train, node_to_cluster)
                ge_val   = apply_graph_structural_encoding(cat_val,   node_to_cluster)
                ge_test  = apply_graph_structural_encoding(cat_test,  node_to_cluster)
                X_train_ge, _          = make_samples(ge_train, args.sequence_length)
                X_val_ge,   y_val_ge   = make_samples(ge_val,   args.sequence_length)
                X_test_ge,  y_test_ge  = make_samples(ge_test,  args.sequence_length)
                print(f"  graph_embedding vocab: {vocab_size_ge}  "
                      f"test samples: {len(X_test_ge)}")

        split_results: Dict[str, Dict] = {}

        # ------------------------------------------------------------ GraphIDyOM
        print("\n[2/4] GraphIDyOM configurations \u2026", flush=True)
        encoding_data = {
            "categorical_id": (X_train_cat, X_val_cat, y_val_cat,
                               X_test_cat, y_test_cat, vocab_size_cat),
            "debruijn":       (X_train_db,  X_val_db,  y_val_db,
                               X_test_db,  y_test_db,  vocab_size_db),
            "graph_embedding":(X_train_ge,  X_val_ge,  y_val_ge,
                               X_test_ge,  y_test_ge,  vocab_size_ge),
        }
        # Raw encoded paths (trajectory-level) for auto-regressive eval
        encoding_paths = {
            "categorical_id": cat_test,
            "debruijn":       db_test,
            "graph_embedding": ge_test if "graph_embedding" in active_encodings else [],
        }
        for enc_idx, enc in enumerate(active_encodings):
            if enc not in encoding_data:
                continue
            train_s, val_s, val_t, test_s, test_t, vs = encoding_data[enc]
            if len(test_s) == 0:
                print(f"  [{enc_idx+1}/{len(active_encodings)}] {enc.upper()}: No test samples – skipping.")
                continue
            print(f"  [{enc_idx+1}/{len(active_encodings)}] {enc.upper()}")
            sys.stdout.flush()
            for order_idx, order in enumerate(args.orders):
                cfg_name  = f"GraphIDyOM_{enc}_ord{order}"
                save_path = (os.path.join(args.output_dir, "models",
                                          f"{cfg_name}_s{split_idx}.pth")
                             if args.n_splits == 1 else None)
                print(f"    [{order_idx+1}/{len(args.orders)}] Order {order}")
                sys.stdout.flush()
                r = train_eval_graphidyom(
                    train_seqs=train_s, val_seqs=val_s, val_targets=val_t,
                    test_seqs=test_s,   test_targets=test_t,
                    vocab_size=vs, order=order,
                    batch_size=args.batch_size, device=device,
                    save_path=save_path,
                    test_paths=encoding_paths[enc] if args.autoregressive else None,
                    adaptive_eval_batch_size=(enc != "debruijn"),
                )
                split_results[cfg_name] = r
                print(f"      ✓ Order {order} done - Acc@1={r['accuracy_top1']:.4f}")
                sys.stdout.flush()

        # ------------------------------------------------------------ Baselines
        if not args.skip_baselines:
            print("\n[3/4] Baselines (categorical_id) …", flush=True)
            for baseline_idx, bname in enumerate(args.baselines):
                print(f"  [{baseline_idx+1}/{len(args.baselines)}] {bname.upper()}")
                sys.stdout.flush()
                try:
                    r = train_eval_baseline(
                        model_name=bname,
                        train_paths=cat_train,
                        test_seqs=X_test_cat, test_targets=y_test_cat,
                        vocab_size=vocab_size_cat,
                        adjacency=None,
                        max_order=args.akom_max_order,
                        iohmm_states=args.iohmm_states,
                        iohmm_iter=args.iohmm_iter,
                        test_paths=cat_test if args.autoregressive else None,
                    )
                    split_results[bname.upper()] = r
                    sys.stdout.flush()
                    print(f"    Acc@1={r['accuracy_top1']:.4f}  Acc@3={r['accuracy_top3']:.4f}  MRR={r['mrr']:.4f}")
                    sys.stdout.flush()
                except Exception as exc:
                    print(f"    ERROR: {exc}")
                    sys.stdout.flush()
        else:
            print("\n[3/4] Baselines skipped (--skip-baselines).")

        # Accumulate
        for model_name, metrics in split_results.items():
            all_split_results.setdefault(model_name, []).append(metrics)

    return vocab_size_cat, vocab_size_db, vocab_size_ge


def main():
    args = parse_args()
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train-fraction must be in the range (0, 1].")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve dataset list (--data-paths overrides --data-path)
    data_paths_list  = args.data_paths  if args.data_paths  else [args.data_path]
    graph_paths_list = (args.graph_paths if args.graph_paths
                        else [args.graph_path] * len(data_paths_list))
    if len(graph_paths_list) < len(data_paths_list):
        graph_paths_list += ([graph_paths_list[-1]] *
                             (len(data_paths_list) - len(graph_paths_list)))
    n_datasets = len(data_paths_list)

    all_split_results: Dict[str, List[Dict]] = {}
    vocab_size_cat = vocab_size_db = vocab_size_ge = 0

    for ds_idx, (_dp, _gp) in enumerate(zip(data_paths_list, graph_paths_list)):
        print(f"\n{'#'*60}\n  Dataset {ds_idx+1}/{n_datasets}: "
              f"{os.path.basename(_dp)}\n{'#'*60}")
        vocab_size_cat, vocab_size_db, vocab_size_ge = _run_one_dataset(
            args, _dp, _gp, device, all_split_results,
            dataset_label=f"Dataset {ds_idx + 1}/{n_datasets}")

    # ------------------------------------------------------------------ 4. Aggregate & save
    print("\n[4/4] Saving results …")

    # Mean (and optional std) across splits
    results: Dict[str, Dict] = {}
    for model_name, splits in all_split_results.items():
        float_keys = [k for k in splits[0] if isinstance(splits[0][k], (int, float))]
        results[model_name] = {
            k: float(np.mean([s[k] for s in splits])) for k in float_keys
        }
        if args.n_splits > 1:
            for k in ["accuracy_top1", "accuracy_top3", "accuracy_top5", "mrr",
                      "ar_accuracy_top1", "ar_accuracy_top3",
                      "ar_accuracy_top5", "ar_mrr"]:
                if k in results[model_name]:
                    results[model_name][f"{k}_std"] = float(
                        np.std([s[k] for s in splits]))

    meta = {
        "data_paths":      data_paths_list,
        "sequence_length": args.sequence_length,
        "n_splits":        args.n_splits,
        "n_datasets":      n_datasets,
        "test_ratio":      args.test_ratio,
        "val_ratio":       args.val_ratio,
        "train_fraction":  args.train_fraction,
        "seed":            args.seed,
        "vocab_size_cat":  vocab_size_cat,
        "vocab_size_db":   vocab_size_db,
        "vocab_size_ge":   vocab_size_ge,
        "graphidyom_encodings": list(args.encodings),
        "graphidyom_orders":    list(args.orders),
        "debruijn_kgram_order": args.debruijn_order,
        "graph_embedding_clusters": args.graph_embedding_clusters,
        "graph_embedding_method": getattr(args, "graph_embedding_method",
                                          GRAPH_EMBEDDING_METHOD),
    }

    # Save full per-split data
    all_splits_path = os.path.join(args.output_dir, "all_splits_results.json")
    with open(all_splits_path, "w") as f:
        json.dump({
            "meta": meta,
            "per_split_results": all_split_results,
            "mean_results": results,
        }, f, indent=2)
    print(f"  Saved \u2192 {all_splits_path}")

    save_results(results, args.output_dir, meta)
    plot_comparison(results, args.output_dir, all_split_results=all_split_results)
    if args.n_splits > 1:
        plot_boxplots(all_split_results, args.output_dir)
    print_table(results)

    print(f"\n\u2713  All done. Results in: {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
