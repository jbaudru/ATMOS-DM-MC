#!/usr/bin/env python3
"""
Training-only script for GraphIDyOM and all baselines.

Trains every model configuration across N random splits and saves each trained
model to disk so evaluate_models.py can load and score them without retraining.

Directory layout written to <models-dir>/
  metadata.json                      – vocab, seeds, hyperparams, split sizes
  split_0/
    node_to_idx.json                 – raw node ID (str) → dense int
    kgram_to_id.json                 – k-gram tuple (str key) → DeBruijn ID
    node_to_cluster.json             – dense node idx (str) → cluster ID  [if GE]
    test_data.npz                    – X_test / y_test arrays for each encoding
    test_paths.pkl                   – trajectory lists for AR evaluation
    graphidyom_<enc>_ord<k>.pt       – GraphIDyOM weight tensors
    baseline_<name>.pkl              – fitted baseline model
  split_1/
    ...

Usage
-----
    python train_models.py \\
        --data-path  data/worldmove_380_US.csv \\
        --graph-path data/worldmove_380_US_network.json \\
        --models-dir saved_models
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

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

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Re-use all data-preparation helpers from train_and_evaluate
from train_and_evaluate import (
    load_trajectories,
    split_trajectories,
    split_trajectories_by_user,
    subsample_trajectories,
    encode_paths_categorical,
    build_debruijn,
    apply_debruijn,
    make_samples,
    make_samples_with_users,
    load_nx_graph,
    load_node_coordinates,
    build_graph_structural_encoding,
    apply_graph_structural_encoding,
    GRAPHIDYOM_ENCODINGS,
    GRAPHIDYOM_ORDERS,
    DEBRUIJN_KGRAM_ORDER,
    GRAPH_EMBEDDING_CLUSTERS,
    GRAPH_EMBEDDING_METHOD,
    TRAIN_FRACTION_DEFAULT,
    BASELINE_NAMES,
)

from models import GraphIDyOMPredictor
from models.baselines import AKOM, CPTPlus, MOGen, IOHMM, SimpleMarkov
from models.graph_idyom import build_graph_structure_from_sequences


# ---------------------------------------------------------------------------
# Algorithm toggles (set to False to skip training)
# ---------------------------------------------------------------------------
TRAIN_ALGOS = {
    # Only the GraphIDyOM model changed (faithful-IDyOM fix), so retrain just its
    # variants and keep every previously-trained baseline on disk.
    "graphidyom_categorical_id": True,  
    "graphidyom_debruijn": False,       
    "graphidyom_graph_embedding": False,  
    
    "akom": True, 
    "cpt": True,
    "mogen": True,
    "iohmm": True,
    
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

# Fraction of each dataset split used end-to-end.
# Set to 0.2 to train / validate / test on 20% of each split.
DATASET_FRACTION = TRAIN_FRACTION_DEFAULT

# ---------------------------------------------------------------------------
# Network / dataset selector
# ---------------------------------------------------------------------------
# Global on/off switch per road network.  A dataset from --data-paths is kept
# only if its network key (matched as ``_<KEY>.`` in the CSV filename) is True
# here.  Lets you run the whole experiment on a single network (e.g. only US)
# without editing CLI arguments.  Set several to True to run a subset; if none
# match the provided paths, all datasets are run (fail-safe).
RUN_NETWORKS = {
    "US": True,
    "DE": True,
    "SG": True,
    "AE": True,
}

# ---------------------------------------------------------------------------
# Per-user history evaluation protocol
# ---------------------------------------------------------------------------
# When enabled, the test set is built user-aware: EVAL_N_USERS users are held
# out, each contributing all of a USER_TEST_RATIO fraction of their trips as
# test trajectories, plus a separate held-out history buffer of the last
# USER_HISTORY_NODES nodes.  Only GraphIDyOM consumes the per-user history
# (its STM); every model is scored on the same users / test trips.
USER_AWARE_EVAL    = True
EVAL_N_USERS       = 100
USER_TEST_RATIO    = 0.10
# Per-user STM (instance memory) buffer length, in number of *distinct* moves.
# Raised to 3000 (lever B): the STM is now built from ALL of each eval user's
# non-test trips, concatenated and truncated to the last USER_HISTORY_NODES
# nodes.  This gives recurring commuters a much longer, fully user-specific
# mobility footprint and raises the chance that the current test context is
# actually covered by the same user's own past moves (the STM's only edge over
# the population baselines).
USER_HISTORY_NODES = 3000


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_and_save_graphidyom(
    train_seqs: np.ndarray,
    vocab_size: int,
    order: int,
    save_path: str,
    device: torch.device,
) -> float:
    """
    Build GraphIDyOM corpus statistics (= training) and save the checkpoint.

    Returns the wall-clock training time in seconds.
    """
    t0 = time.time()
    print(f"    [train] GraphIDyOM order={order} vocab={vocab_size} …", end="", flush=True)
    seqs_t = torch.tensor(train_seqs, dtype=torch.long, device=device)
    graph_structure = build_graph_structure_from_sequences(seqs_t, vocab_size,
                                                           max_order=order)
    model = GraphIDyOMPredictor(vocab_size=vocab_size, max_order=order).to(device).eval()
    train_time = time.time() - t0
    print(f" done ({train_time:.1f}s)")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ew = graph_structure.get("edge_weights")
    oc = graph_structure.get("out_counts")
    torch.save({
        "edge_weights":       ew.cpu() if isinstance(ew, torch.Tensor) else None,
        "first_order_sparse": graph_structure.get("first_order_sparse"),
        "out_counts":         oc.cpu() if isinstance(oc, torch.Tensor) else None,
        # Population-level variable-order k-gram tables (orders 2..order).
        # Without these the LTM silently collapses to order-1 at eval time and
        # the whole variable-order model degenerates to a 1st-order Markov chain.
        "kgram_tables":       graph_structure.get("kgram_tables", {}),
        "vocab_size":         vocab_size,
        "max_order":          order,
        "smoothing":          model.smoothing,
        "train_time_s":       train_time,
    }, save_path)
    return train_time


def train_and_save_baseline(
    model_name: str,
    train_paths: List[List[int]],
    vocab_size: int,
    save_path: str,
    max_order: int = 10,
    iohmm_states: int = 10,
    iohmm_iter: int = 50,
) -> float:
    """
    Fit a baseline model and pickle it to *save_path*.

    Returns the wall-clock training time in seconds.
    """
    model_map = {
        "akom":  lambda: AKOM(max_order=max_order),
        "cpt":   lambda: CPTPlus(max_context_length=5),
        "mogen": lambda: MOGen(max_order=max_order),
        "iohmm": lambda: IOHMM(n_states=iohmm_states, max_iter=iohmm_iter),
        "mc2":    lambda: SimpleMarkov(order=2,  blend=False),
        "mc2p":   lambda: SimpleMarkov(order=2,  blend=False),
        "mc5p":   lambda: SimpleMarkov(order=5,  blend=False),
        "mc10p":  lambda: SimpleMarkov(order=10, blend=False),
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

    if model_name.lower() == "iohmm":
        _pbar = tqdm(total=model.max_iter, desc="IOHMM EM", unit="iter", leave=False)
        def _cb(em_iter, converged):
            _pbar.update(1)
            if converged:
                _pbar.set_postfix({"status": f"converged @ {em_iter}"})
        model.fit(train_paths, adjacency=None, progress_callback=_cb)
        _pbar.close()
    else:
        model.fit(train_paths, adjacency=None)

    train_time = time.time() - t0
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "vocab_size": vocab_size,
                     "train_time_s": train_time}, f)
    return train_time


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train all GraphIDyOM + baseline configurations and save models to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path",  default="data/worldmove_380_US.csv")
    p.add_argument("--graph-path", default="data/worldmove_380_US_network.json")
    p.add_argument("--models-dir", default="saved_models",
                   help="Root directory where trained models will be written.")
    p.add_argument("--sequence-length", type=int, default=10)
    p.add_argument("--test-ratio",  type=float, default=0.10)
    p.add_argument("--val-ratio",   type=float, default=0.10)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--train-fraction", type=float, default=DATASET_FRACTION,
                   help="Fraction of trajectories used for train/val/test in each split.")
    p.add_argument("--orders", type=int, nargs="+", default=GRAPHIDYOM_ORDERS)
    p.add_argument("--encodings", nargs="+", default=GRAPHIDYOM_ENCODINGS,
                   choices=["categorical_id", "debruijn", "graph_embedding"])
    p.add_argument("--debruijn-order", type=int, default=DEBRUIJN_KGRAM_ORDER)
    p.add_argument("--graph-embedding-clusters", type=int,
                   default=GRAPH_EMBEDDING_CLUSTERS)
    p.add_argument("--graph-embedding-method", default=GRAPH_EMBEDDING_METHOD,
                   choices=["spatial", "node2vec", "structural"],
                   help="Feature space for graph_embedding clustering: spatial "
                        "(x/y coordinates, locally coherent), node2vec (graph "
                        "embeddings), or structural (degree/PageRank, legacy).")
    p.add_argument("--baselines", nargs="+", default=BASELINE_NAMES,
                   choices=BASELINE_NAMES)
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--n-splits", type=int, default=1)
    p.add_argument("--splits", type=int, nargs="+", default=None,
                   help="Specific split indices to train (e.g. --splits 0 2). "
                        "Default: 0..n-splits-1.")
    p.add_argument("--akom-max-order",  type=int, default=10)
    p.add_argument("--iohmm-states",    type=int, default=10)
    p.add_argument("--iohmm-iter",      type=int, default=50)
    p.add_argument("--data-paths", nargs="+", default=[
                       "data/worldmove_380_US.csv",
                       "data/worldmove_431_DE.csv",
                       "data/worldmove_539_SG.csv",
                       "data/worldmove_609_AE.csv",
                   ],
                   help="Multiple CSVs; overrides --data-path when given.")
    p.add_argument("--graph-paths", nargs="+", default=[
                       "data/worldmove_380_US_network.json",
                       "data/worldmove_431_DE_network.json",
                       "data/worldmove_539_SG_network.json",
                       "data/worldmove_609_AE_network.json",
                   ],
                   help="Network JSONs matching --data-paths (same order).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train-fraction must be in (0, 1].")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.models_dir, exist_ok=True)

    # Which split indices to train (explicit subset, or 0..n-splits-1).
    split_indices_global = (sorted(set(args.splits)) if args.splits is not None
                            else list(range(args.n_splits)))

    data_paths_list  = args.data_paths  if args.data_paths  else [args.data_path]
    graph_paths_list = args.graph_paths if args.graph_paths else [args.graph_path] * len(data_paths_list)
    if len(graph_paths_list) < len(data_paths_list):
        graph_paths_list += [graph_paths_list[-1]] * (len(data_paths_list) - len(graph_paths_list))

    # ---- apply the RUN_NETWORKS global selector -----------------------------
    # Keep only datasets whose network key is enabled in RUN_NETWORKS.  The key
    # is matched as ``_<KEY>.`` in the CSV basename (e.g. worldmove_380_US.csv
    # -> "_US.").  If nothing matches, fall back to running every dataset.
    _enabled_keys = [k for k, v in RUN_NETWORKS.items() if v]
    _selected = [(d, g) for d, g in zip(data_paths_list, graph_paths_list)
                 if any(f"_{k}." in os.path.basename(d) for k in _enabled_keys)]
    if _selected:
        data_paths_list, graph_paths_list = (list(t) for t in zip(*_selected))
        print(f"[network-select] RUN_NETWORKS -> "
              f"{[os.path.basename(d) for d in data_paths_list]}")
    else:
        print("[network-select] RUN_NETWORKS matched no dataset "
              "-> running all provided networks.")

    # ---- master metadata accumulator ----------------------------------------
    all_meta = {
        "data_paths":      data_paths_list,
        "sequence_length": args.sequence_length,
        "n_splits":        len(split_indices_global),
        "split_indices":   split_indices_global,
        "test_ratio":      args.test_ratio,
        "val_ratio":       args.val_ratio,
        "train_fraction":  args.train_fraction,
        "seed":            args.seed,
        "encodings":       args.encodings,
        "orders":          args.orders,
        "debruijn_order":  args.debruijn_order,
        "graph_embedding_clusters": args.graph_embedding_clusters,
        "graph_embedding_method":   args.graph_embedding_method,
        "baselines":       args.baselines if not args.skip_baselines else [],
        "datasets":        [],   # list of per-dataset metadata blocks
        "splits":          [],   # flat list (for backward compatibility)
    }

    for ds_idx, (data_path, graph_path) in enumerate(zip(data_paths_list, graph_paths_list)):
        print(f"\n{'#'*60}\n  Dataset {ds_idx+1}/{len(data_paths_list)}: "
              f"{os.path.basename(data_path)}\n{'#'*60}")

        print("\n[1] Loading trajectories …")
        full_paths, node_vocab, user_ids = load_trajectories(
            data_path, args.sequence_length, return_users=True)
        node_to_idx    = {n: i for i, n in enumerate(node_vocab)}
        vocab_size_cat = len(node_vocab)
        n_ge_clusters  = args.graph_embedding_clusters
        print(f"  Vocabulary (categorical): {vocab_size_cat}")

        ds_label   = os.path.splitext(os.path.basename(data_path))[0]
        ds_dir     = os.path.join(args.models_dir, f"ds{ds_idx}_{ds_label}")
        ds_meta    = {
            "ds_idx":    ds_idx,
            "data_path": data_path,
            "graph_path": graph_path,
            "label":     ds_label,
            "ds_dir":    ds_dir,
            "vocab_size_cat": vocab_size_cat,
            "splits":    [],
        }
        os.makedirs(ds_dir, exist_ok=True)

        split_indices = split_indices_global
        for split_idx in tqdm(split_indices, desc=f"ds{ds_idx} splits"):
            split_seed = args.seed + split_idx
            split_dir  = os.path.join(ds_dir, f"split_{split_idx}")
            os.makedirs(split_dir, exist_ok=True)

            sep = "=" * 60
            print(f"\n{sep}\n  Split {split_idx} ({split_indices.index(split_idx)+1}/{len(split_indices)})  seed={split_seed}\n{sep}")

            # ---- split --------------------------------------------------------
            if USER_AWARE_EVAL:
                usplit = split_trajectories_by_user(
                    full_paths, user_ids,
                    eval_n_users=EVAL_N_USERS,
                    user_test_ratio=USER_TEST_RATIO,
                    user_history_nodes=USER_HISTORY_NODES,
                    val_ratio=args.val_ratio,
                    seed=split_seed,
                )
                train_paths     = usplit["train_paths"]
                val_paths       = usplit["val_paths"]
                test_paths      = usplit["test_paths"]
                test_user_ids   = usplit["test_user_ids"]
                history_by_user = usplit["history_by_user"]
                if args.train_fraction < 1.0:
                    n_tr = len(train_paths)
                    train_paths = subsample_trajectories(train_paths, args.train_fraction,
                                                         seed=split_seed + 10_000)
                    val_paths   = subsample_trajectories(val_paths,   args.train_fraction,
                                                         seed=split_seed + 20_000)
                    print(f"  subsample train {len(train_paths)}/{n_tr}")
                print(f"  [user-aware] eval_users={len(usplit['eval_user_ids'])}  "
                      f"train={len(train_paths)}  val={len(val_paths)}  "
                      f"test_trips={len(test_paths)}  hist_users={len(history_by_user)}")
            else:
                train_paths, val_paths, test_paths = split_trajectories(
                    full_paths, args.test_ratio, args.val_ratio, split_seed)
                if args.train_fraction < 1.0:
                    n_tr = len(train_paths)
                    train_paths = subsample_trajectories(train_paths, args.train_fraction,
                                                         seed=split_seed + 10_000)
                    val_paths   = subsample_trajectories(val_paths,   args.train_fraction,
                                                         seed=split_seed + 20_000)
                    test_paths  = subsample_trajectories(test_paths,  args.train_fraction,
                                                         seed=split_seed + 30_000)
                    print(f"  subsample train {len(train_paths)}/{n_tr}")
                # Each test trajectory is treated as its own user (no history).
                test_user_ids   = list(range(len(test_paths)))
                history_by_user = {}
                print(f"  train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")

            # ---- categorical encoding ----------------------------------------
            cat_train = encode_paths_categorical(train_paths, node_to_idx)
            cat_val   = encode_paths_categorical(val_paths,   node_to_idx)
            X_train_cat, _          = make_samples(cat_train, args.sequence_length)
            X_val_cat,   y_val_cat  = make_samples(cat_val,   args.sequence_length)
            # Test paths are encoded with their owning user id preserved so the
            # per-user STM history can be attached at evaluation time.
            cat_test, cat_test_users, cat_test_path_ids = [], [], []
            for _pid, (_p, _u) in enumerate(zip(test_paths, test_user_ids)):
                _enc = [node_to_idx[n] for n in _p if n in node_to_idx]
                if _enc:
                    cat_test.append(_enc)
                    cat_test_users.append(_u)
                    cat_test_path_ids.append(int(_pid))
            X_test_cat, y_test_cat, user_test_cat = make_samples_with_users(
                cat_test, cat_test_users, args.sequence_length)

            # ---- DeBruijn encoding -------------------------------------------
            kgram_to_id, vocab_size_db = build_debruijn(cat_train, args.debruijn_order)
            db_train = apply_debruijn(cat_train, kgram_to_id, args.debruijn_order)
            db_val   = apply_debruijn(cat_val,   kgram_to_id, args.debruijn_order)
            X_train_db, _         = make_samples(db_train, args.sequence_length)
            X_val_db,  y_val_db   = make_samples(db_val,   args.sequence_length)
            db_test, db_test_users, db_test_path_ids = [], [], []
            for _enc, _u, _pid in zip(cat_test, cat_test_users, cat_test_path_ids):
                _d = apply_debruijn([_enc], kgram_to_id, args.debruijn_order)
                if _d and _d[0]:
                    db_test.append(_d[0])
                    db_test_users.append(_u)
                    db_test_path_ids.append(int(_pid))
            X_test_db, y_test_db, user_test_db = make_samples_with_users(
                db_test, db_test_users, args.sequence_length)

            # ---- graph-embedding encoding ------------------------------------
            vocab_size_ge   = 0
            node_to_cluster = None
            ge_test: List[List[int]] = []
            ge_test_users: List[int] = []
            ge_test_path_ids: List[int] = []
            user_test_ge = np.empty(0, dtype=np.int64)
            X_train_ge = X_val_ge = y_val_ge = X_test_ge = y_test_ge = np.empty((0,))
            active_encodings = list(args.encodings)
            if "graph_embedding" in active_encodings:
                nx_graph = load_nx_graph(graph_path, node_to_idx, vocab_size_cat)
                if nx_graph is None:
                    print("  graph_embedding skipped (NetworkX graph unavailable).")
                    active_encodings = [e for e in active_encodings if e != "graph_embedding"]
                else:
                    ge_coords = (load_node_coordinates(graph_path, node_to_idx,
                                                       vocab_size_cat)
                                 if args.graph_embedding_method == "spatial"
                                 else None)
                    print(f"  Building graph_embedding ({n_ge_clusters} clusters, "
                          f"method={args.graph_embedding_method}) …", flush=True)
                    node_to_cluster, vocab_size_ge = build_graph_structural_encoding(
                        nx_graph, vocab_size_cat, n_ge_clusters,
                        method=args.graph_embedding_method, coords=ge_coords)
                    ge_train = apply_graph_structural_encoding(cat_train, node_to_cluster)
                    ge_val   = apply_graph_structural_encoding(cat_val,   node_to_cluster)
                    X_train_ge, _          = make_samples(ge_train, args.sequence_length)
                    X_val_ge,   y_val_ge   = make_samples(ge_val,   args.sequence_length)
                    for _enc, _u, _pid in zip(cat_test, cat_test_users, cat_test_path_ids):
                        _g = apply_graph_structural_encoding([_enc], node_to_cluster)
                        if _g and _g[0]:
                            ge_test.append(_g[0])
                            ge_test_users.append(_u)
                            ge_test_path_ids.append(int(_pid))
                    X_test_ge, y_test_ge, user_test_ge = make_samples_with_users(
                        ge_test, ge_test_users, args.sequence_length)
                    print(f"  graph_embedding vocab={vocab_size_ge}  "
                          f"test_samples={len(X_test_ge)}")

            # ---- per-user STM history buffers (encoded per encoding) ----------
            history_cat: Dict[int, List[int]] = {}
            history_db:  Dict[int, List[int]] = {}
            history_ge:  Dict[int, List[int]] = {}
            for _u, _nodes in history_by_user.items():
                _enc = [node_to_idx[n] for n in _nodes if n in node_to_idx]
                history_cat[_u] = _enc
                _d = apply_debruijn([_enc], kgram_to_id, args.debruijn_order) if _enc else []
                history_db[_u] = _d[0] if _d and _d[0] else []
                if node_to_cluster is not None and _enc:
                    _g = apply_graph_structural_encoding([_enc], node_to_cluster)
                    history_ge[_u] = _g[0] if _g and _g[0] else []
                else:
                    history_ge[_u] = []

            # ---- save vocab / split metadata ---------------------------------
            with open(os.path.join(split_dir, "node_to_idx.json"), "w") as f:
                json.dump({str(k): v for k, v in node_to_idx.items()}, f)

            # kgram_to_id keys are tuples → serialise as "(a, b)" strings
            with open(os.path.join(split_dir, "kgram_to_id.json"), "w") as f:
                json.dump({str(k): v for k, v in kgram_to_id.items()}, f)

            if node_to_cluster is not None:
                with open(os.path.join(split_dir, "node_to_cluster.json"), "w") as f:
                    json.dump({str(i): int(c) for i, c in enumerate(node_to_cluster)}, f)

            # Save test arrays (used by evaluate_models.py).  user_test_* gives
            # the owning user id of every teacher-forced window so the matching
            # per-user STM history can be attached during evaluation.
            np.savez(os.path.join(split_dir, "test_data.npz"),
                     X_test_cat=X_test_cat, y_test_cat=y_test_cat,
                     X_test_db=X_test_db,   y_test_db=y_test_db,
                     X_test_ge=X_test_ge,   y_test_ge=y_test_ge,
                     user_test_cat=user_test_cat,
                     user_test_db=user_test_db,
                     user_test_ge=user_test_ge)

            # Node visit-frequency prior (population popularity) over the
            # training trajectories.  Used at eval time to decode a
            # graph_embedding cluster prediction back to a specific node via
            # the within-cluster node distribution P(node | cluster).
            node_prior = np.zeros(vocab_size_cat, dtype=np.float64)
            for _seq in cat_train:
                if len(_seq):
                    np.add.at(node_prior, np.asarray(_seq, dtype=np.int64), 1.0)

            # First-order successor counts u -> v over the training
            # trajectories.  This is the data-driven adjacency used to generate
            # node candidates when decoding a graph_embedding cluster
            # prediction (the supplied road network often does not match the
            # observed trajectory transitions).
            node_transitions: Dict[int, Dict[int, int]] = {}
            for _seq in cat_train:
                for _a, _b in zip(_seq[:-1], _seq[1:]):
                    _d = node_transitions.setdefault(int(_a), {})
                    _d[int(_b)] = _d.get(int(_b), 0) + 1

            # Second-order (bigram) successor counts (u,v) -> w.  Used by the
            # node-space decoder to generate highly specific candidates:
            # given the last two context nodes (prev=u, current=v), the next
            # node is almost always one of the few w values seen in training
            # for that exact bigram, which compensates for the coarser cluster
            # predictions made by low-order GE Markov chains (MC1_GE, ord2).
            node_transitions2: Dict[tuple, Dict[int, int]] = {}
            for _seq in cat_train:
                for _a, _b, _c in zip(_seq[:-2], _seq[1:-1], _seq[2:]):
                    _key = (int(_a), int(_b))
                    _d2 = node_transitions2.setdefault(_key, {})
                    _d2[int(_c)] = _d2.get(int(_c), 0) + 1

            with open(os.path.join(split_dir, "test_paths.pkl"), "wb") as f:
                pickle.dump({"cat_test": cat_test,
                             "db_test":  db_test,
                             "ge_test":  ge_test,
                             "cat_test_users": cat_test_users,
                             "db_test_users":  db_test_users,
                             "ge_test_users":  ge_test_users,
                             "cat_test_path_ids": cat_test_path_ids,
                             "db_test_path_ids":  db_test_path_ids,
                             "ge_test_path_ids":  ge_test_path_ids,
                             "history_cat": history_cat,
                             "history_db":  history_db,
                             "history_ge":  history_ge,
                             "node_prior":  node_prior,
                             "node_transitions": node_transitions,
                             "node_transitions2": node_transitions2,
                             "user_history_nodes": USER_HISTORY_NODES}, f)

            split_info = {
                "split_idx":      split_idx,
                "split_seed":     split_seed,
                "split_dir":      split_dir,
                "ds_idx":         ds_idx,
                "ds_label":       ds_label,
                "n_train":        len(train_paths),
                "n_val":          len(val_paths),
                "n_test":         len(test_paths),
                "vocab_size_cat": vocab_size_cat,
                "vocab_size_db":  int(vocab_size_db),
                "vocab_size_ge":  int(vocab_size_ge),
                "active_encodings": active_encodings,
                "user_aware_eval":    USER_AWARE_EVAL,
                "user_history_nodes": USER_HISTORY_NODES,
            }
            ds_meta["splits"].append(split_info)
            all_meta["splits"].append(split_info)

            # ---- GraphIDyOM models -------------------------------------------
            print("\n[2] Training GraphIDyOM …")
            encoding_train_data = {
                "categorical_id": (X_train_cat, vocab_size_cat),
                "debruijn":       (X_train_db,  int(vocab_size_db)),
                "graph_embedding":(X_train_ge,  int(vocab_size_ge)),
            }
            for enc in active_encodings:
                toggle_key = f"graphidyom_{enc}"
                if not TRAIN_ALGOS.get(toggle_key, True):
                    print(f"  graphidyom_{enc}: disabled by TRAIN_ALGOS.")
                    continue
                if enc not in encoding_train_data:
                    continue
                X_tr, vs = encoding_train_data[enc]
                if len(X_tr) == 0 or vs == 0:
                    print(f"  {enc}: no training samples – skipped.")
                    continue
                for order in args.orders:
                    cfg_name  = f"graphidyom_{enc}_ord{order}"
                    save_path = os.path.join(split_dir, f"{cfg_name}.pt")
                    try:
                        train_and_save_graphidyom(X_tr, vs, order, save_path, device)
                    except Exception as exc:
                        print(f"  ERROR training {cfg_name}: {exc}")

            # ---- Baseline models --------------------------------------------
            if not args.skip_baselines:
                print("\n[3] Training baselines (categorical_id) …")
                # Encoded training sequences per baseline. By default a baseline
                # consumes categorical-id (raw node) sequences; ``mc1_ge`` is the
                # only one trained on graph-embedding cluster sequences so we can
                # tease apart "model gain" from "encoding gain".
                ge_available = (vocab_size_ge > 0 and len(X_train_ge) > 0)
                for bname in args.baselines:
                    if not TRAIN_ALGOS.get(bname, True):
                        print(f"  {bname.upper()} … skipped (disabled by TRAIN_ALGOS)")
                        continue
                    if bname.endswith("_ge"):
                        if not ge_available:
                            print(f"  {bname.upper()} … skipped (graph_embedding unavailable for this split)")
                            continue
                        train_seqs_b = ge_train
                        vs_b         = int(vocab_size_ge)
                    else:
                        train_seqs_b = cat_train
                        vs_b         = vocab_size_cat
                    print(f"  {bname.upper()} …", end="", flush=True)
                    save_path = os.path.join(split_dir, f"baseline_{bname}.pkl")
                    try:
                        t = train_and_save_baseline(
                            bname, train_seqs_b, vs_b, save_path,
                            max_order=args.akom_max_order,
                            iohmm_states=args.iohmm_states,
                            iohmm_iter=args.iohmm_iter,
                        )
                        print(f" done ({t:.1f}s)")
                    except Exception as exc:
                        print(f" ERROR: {exc}")
            else:
                print("\n[3] Baselines skipped.")

        all_meta["datasets"].append(ds_meta)

    # ---- write master metadata ----------------------------------------------
    meta_path = os.path.join(args.models_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\n✓  Training complete.  Metadata → {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
