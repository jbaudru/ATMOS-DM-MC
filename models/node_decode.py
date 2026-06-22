"""
Node-ID decoding layer.

Research goal: *every* model must predict a raw NODE ID given a sequence of
nodes, and be scored in node-ID space — regardless of which encoding it was
trained on.

* ``categorical_id`` already predicts node IDs (model vocab == node vocab), so
  it needs no decoding and is handled on the fast existing path.
* ``debruijn`` predicts a *k-gram* ID.  A k-gram deterministically ends in a
  single node, so the predicted next node is the last element of the predicted
  k-gram.  We collapse the k-gram distribution onto node IDs by summing the
  probability mass of every k-gram that shares the same last node.
* ``graph_embedding`` predicts one of a small number of structural *clusters*.
  A cluster maps to many nodes, so the cluster prediction is resolved to a node
  by restricting to the graph-neighbours of the current node (the next node in
  a road trajectory is always graph-adjacent) and splitting each predicted
  cluster's mass equally among the current node's neighbours that fall in that
  cluster.

The decoder returns a *sparse* node distribution ``(cand_nodes, cand_probs)``
(only the nodes that can carry mass), which is enough to compute every ranking
/ calibration metric without materialising a length-``vocab_cat`` vector.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

import numpy as np


class NodeSpaceAdapter:
    """Convert a model's encoded-space prediction into a node-ID prediction.

    Parameters
    ----------
    encoding : str
        One of ``'categorical_id'``, ``'debruijn'``, ``'graph_embedding'``.
    vocab_cat : int
        Number of distinct nodes (categorical vocabulary size).
    seq_len : int
        Teacher-forcing / warm-up context length (model input window length).
    kgram_to_id : dict, optional
        ``{kgram_tuple: id}`` mapping (debruijn only).
    debruijn_order : int
        k for the de Bruijn k-gram encoding.
    node_to_cluster : np.ndarray, optional
        ``(vocab_cat,)`` array mapping node id -> structural cluster id
        (graph_embedding only).
    neighbors : dict, optional
        ``{node_id: set(neighbor_node_ids)}`` undirected road adjacency
        (graph_embedding only; used to resolve a cluster to a node).
    node_prior : np.ndarray, optional
        ``(vocab_cat,)`` training visit-frequency counts per node
        (graph_embedding only).  Used to learn the within-cluster node
        distribution ``P(node | cluster)`` so a predicted cluster is resolved
        to the more frequently visited node inside it.  When absent the
        within-cluster distribution is uniform.
    transitions : dict, optional
        ``{node_id: {next_node_id: count}}`` first-order successor counts
        observed in the *training* trajectories (graph_embedding only).  This
        is the primary candidate generator for resolving a cluster prediction
        to a node: the next node is drawn from the nodes that actually
        followed the current node during training, reranked by the predicted
        cluster.  It is used in preference to ``neighbors`` because a supplied
        road-network graph often does not match the observed trajectory
        transitions (e.g. sparsely sampled GPS traces skip intermediate
        intersections).
    transitions2 : dict, optional
        ``{(prev_node_id, node_id): {next_node_id: count}}`` second-order
        (bigram) successor counts from training trajectories.  When available
        and a previous node is supplied at decode time, bigram transitions are
        tried first, providing a much smaller and more specific candidate set
        that compensates for the coarse cluster predictions of low-order
        Markov models (MC1_GE, GE_ord2).  Falls back to first-order
        transitions when the bigram has not been seen in training.
    """

    def __init__(self,
                 encoding: str,
                 vocab_cat: int,
                 seq_len: int,
                 *,
                 kgram_to_id: Optional[Dict[tuple, int]] = None,
                 debruijn_order: int = 1,
                 node_to_cluster: Optional[np.ndarray] = None,
                 neighbors: Optional[Dict[int, set]] = None,
                 node_prior: Optional[np.ndarray] = None,
                 transitions: Optional[Dict[int, Dict[int, float]]] = None,
                 transitions2: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None):
        self.encoding = encoding
        self.vocab_cat = int(vocab_cat)
        self.seq_len = int(seq_len)
        self.debruijn_order = int(debruijn_order)
        self.neighbors = neighbors
        self._transitions = transitions
        self._transitions2 = transitions2

        # ---- debruijn lookup tables --------------------------------------
        self._id_to_lastnode: Optional[np.ndarray] = None
        self._kgram_to_id: Optional[Dict[tuple, int]] = None
        if encoding == "debruijn" and kgram_to_id:
            n_kg = max(kgram_to_id.values()) + 1
            last = np.zeros(n_kg, dtype=np.int64)
            for kg, kid in kgram_to_id.items():
                last[kid] = int(kg[-1])
            self._id_to_lastnode = last
            self._kgram_to_id = dict(kgram_to_id)

        # ---- graph-embedding lookup tables -------------------------------
        self._node_to_cluster: Optional[np.ndarray] = None
        self._within_cluster: Optional[np.ndarray] = None
        if encoding == "graph_embedding" and node_to_cluster is not None:
            n2c = np.asarray(node_to_cluster, dtype=np.int64)
            self._node_to_cluster = n2c
            n_clusters = int(n2c.max()) + 1 if n2c.size else 1
            # Learn P(node | its cluster) from training visit frequencies (or
            # uniform when no prior is supplied).  This is the maximum-
            # likelihood decode of a cluster prediction back to a node and
            # replaces the previous "split a cluster's mass equally among the
            # current node's neighbours" rule -- that rule divided by a *local*
            # neighbour-cluster count and systematically buried the true next
            # node whenever it shared a cluster with sibling neighbours,
            # yielding worse-than-random rankings.
            if node_prior is not None and len(node_prior) >= self.vocab_cat:
                counts = np.asarray(node_prior, dtype=np.float64)[:self.vocab_cat].copy()
            else:
                counts = np.ones(self.vocab_cat, dtype=np.float64)
            counts += 1.0  # Laplace smoothing so every node stays reachable
            cluster_total = np.zeros(max(n_clusters, 1), dtype=np.float64)
            np.add.at(cluster_total, n2c, counts)
            self._within_cluster = counts / np.maximum(cluster_total[n2c], 1e-12)

    # ------------------------------------------------------------------
    # Encoded prediction  ->  sparse node distribution
    # ------------------------------------------------------------------
    def decode(self, enc_proba: np.ndarray,
               current_node: Optional[int] = None,
               prev_node: Optional[int] = None,
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(cand_nodes, cand_probs)`` over node IDs (probs sum to 1).

        ``current_node`` is required for ``graph_embedding`` (neighbour
        restriction) and ignored otherwise.  ``prev_node`` is the node
        immediately before ``current_node`` in the context; when provided and
        ``transitions2`` was supplied at construction time, bigram transitions
        are used for candidate generation (more specific candidates).
        """
        if self.encoding == "categorical_id":
            nz = np.nonzero(enc_proba)[0]
            p = enc_proba[nz]
            s = p.sum()
            return nz.astype(np.int64), (p / s if s > 0 else p)

        if self.encoding == "debruijn":
            return self._decode_debruijn(enc_proba)

        if self.encoding == "graph_embedding":
            return self._decode_graph_embedding(enc_proba, current_node, prev_node)

        raise ValueError(f"Unknown encoding '{self.encoding}'.")

    def _decode_debruijn(self, enc_proba: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        last = self._id_to_lastnode
        if last is None:
            nz = np.nonzero(enc_proba)[0]
            return nz.astype(np.int64), enc_proba[nz]
        m = min(len(enc_proba), len(last))
        node_mass = np.zeros(self.vocab_cat, dtype=np.float64)
        np.add.at(node_mass, last[:m], enc_proba[:m])
        cand = np.nonzero(node_mass)[0]
        p = node_mass[cand]
        s = p.sum()
        return cand.astype(np.int64), (p / s if s > 0 else p)

    def _decode_graph_embedding(self, cluster_proba: np.ndarray,
                                current_node: Optional[int],
                                prev_node: Optional[int] = None,
                                ) -> Tuple[np.ndarray, np.ndarray]:
        n2c = self._node_to_cluster
        if n2c is None:
            nz = np.nonzero(cluster_proba)[0]
            return nz.astype(np.int64), cluster_proba[nz]

        u = int(current_node) if current_node is not None else None
        prev = int(prev_node) if prev_node is not None else None
        cand, edge_w = self._candidates(u, prev)
        if cand is None or cand.size == 0:
            # No candidate generator available: spread over all cluster members.
            return self._decode_ge_fallback(cluster_proba)

        cl = n2c[cand]
        # P(next = v) proportional to P(cluster(v)) * P(v | context), where
        # P(v | context) is the normalised training transition probability.
        # Both signals are normalised over the candidate set so neither the
        # absolute scale of counts nor the total cluster probability mass can
        # dominate: cluster and transition evidence are weighted equally.
        cl_p = cluster_proba[cl].astype(np.float64)
        cl_s = cl_p.sum()
        if cl_s > 0:
            cl_p = cl_p / cl_s   # P(cluster | ctx) re-normalised over candidates
        score = cl_p * edge_w    # edge_w is already P(v|u) from _candidates
        s = score.sum()
        if s <= 0:
            # Predicted clusters carry no mass over the candidates: fall back to
            # the transition prior alone so ranking stays informative.
            score = edge_w
            s = score.sum()
        if s <= 0:
            score = np.ones(cand.size, dtype=np.float64)
            s = float(cand.size)
        return cand, score / s

    def _candidates(self, u: Optional[int], prev: Optional[int] = None):
        """Return ``(cand_nodes, weights)`` candidate generator for a node ``u``.

        Preference order:
        1. Bigram (prev, u) training transitions  -- most specific
        2. Unigram (u) training transitions       -- fallback
        3. Road-graph neighbours of u             -- last resort
        4. (None, None)                           -- no candidates

        Returned weights are normalised to sum to 1 (i.e. P(v | context)) so
        they can be combined with the cluster-probability signal on equal
        footing without raw-count scale dominating.
        """
        def _norm(arr: np.ndarray) -> np.ndarray:
            s = arr.sum()
            return arr / s if s > 0 else arr

        # --- bigram transitions (prev, u) → {next: count} ---
        tr2 = self._transitions2
        if tr2 is not None and u is not None and prev is not None:
            key = (int(prev), int(u))
            d2 = tr2.get(key)
            if d2:
                cand = np.fromiter((v for v in d2 if 0 <= v < self.vocab_cat),
                                   dtype=np.int64)
                if cand.size:
                    w = np.fromiter((d2[int(v)] for v in cand),
                                    dtype=np.float64)
                    return cand, _norm(w)

        # --- first-order transitions u → {next: count} ---
        tr = self._transitions
        if tr is not None and u is not None:
            d = tr.get(u)
            if d:
                cand = np.fromiter((v for v in d if 0 <= v < self.vocab_cat),
                                   dtype=np.int64)
                if cand.size:
                    w = np.fromiter((d[int(v)] for v in cand),
                                    dtype=np.float64)
                    return cand, _norm(w)

        # --- road-graph neighbours ---
        if self.neighbors is not None and u is not None:
            nb = self.neighbors.get(u)
            if nb:
                cand = np.fromiter((m for m in nb if 0 <= m < self.vocab_cat),
                                   dtype=np.int64)
                if cand.size:
                    w = (self._within_cluster[cand]
                         if self._within_cluster is not None
                         else np.ones(cand.size, dtype=np.float64))
                    return cand, _norm(w)
        return None, None

    def _decode_ge_fallback(self, cluster_proba: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """No-adjacency fallback: spread each cluster's mass over its member
        nodes weighted by the within-cluster popularity prior (dense, only used
        when the road graph is unavailable)."""
        n2c = self._node_to_cluster
        within = (self._within_cluster
                  if self._within_cluster is not None
                  else np.ones(self.vocab_cat, dtype=np.float64))
        per_node = cluster_proba[n2c].astype(np.float64) * within
        cand = np.nonzero(per_node)[0]
        p = per_node[cand]
        s = p.sum()
        return cand.astype(np.int64), (p / s if s > 0 else p)

    # ------------------------------------------------------------------
    # Node context  ->  encoded model-input window  (auto-regressive)
    # ------------------------------------------------------------------
    def encode_context(self, node_ctx: List[int]) -> Optional[List[int]]:
        """Encode a raw NODE context into the model's input symbol space.

        Returns the (up to ``seq_len``) trailing encoded symbols, or ``None``
        when the context cannot be encoded (e.g. only unseen debruijn k-grams).
        """
        if self.encoding == "categorical_id":
            return list(node_ctx[-self.seq_len:])

        if self.encoding == "graph_embedding":
            n2c = self._node_to_cluster
            if n2c is None:
                return list(node_ctx[-self.seq_len:])
            enc = [int(n2c[n]) for n in node_ctx if 0 <= n < self.vocab_cat]
            return enc[-self.seq_len:] if enc else None

        if self.encoding == "debruijn":
            k2i = self._kgram_to_id
            o = self.debruijn_order
            if not k2i or len(node_ctx) < o:
                return None
            enc = []
            for i in range(len(node_ctx) - o + 1):
                kid = k2i.get(tuple(node_ctx[i: i + o]))
                if kid is not None:
                    enc.append(kid)
            return enc[-self.seq_len:] if enc else None

        raise ValueError(f"Unknown encoding '{self.encoding}'.")


# ---------------------------------------------------------------------------
# Node-space metric accumulation (shared by teacher-forced + auto-regressive)
# ---------------------------------------------------------------------------

def _haversine_km(coords: np.ndarray, a: int, b: int) -> Optional[float]:
    """Great-circle distance (km) between nodes ``a`` and ``b``.

    ``coords`` is the ``(vocab, 2)`` ``[x=lon, y=lat]`` array (degrees).
    Returns ``None`` when either id is out of range or lacks coordinates.
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
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    h = (np.sin(dphi / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2)
    return float(2.0 * R * np.arcsin(np.sqrt(min(1.0, h))))


class NodeMetricAccumulator:
    """Accumulate node-ID ranking + calibration metrics over predictions.

    Ranking convention matches the categorical batched path: the rank of the
    true node is ``(# nodes with strictly greater probability) + 1`` (ties are
    resolved optimistically), so all encodings are scored identically.
    """

    def __init__(self, coords: Optional[np.ndarray] = None,
                 predictions_out: Optional[dict] = None) -> None:
        self.n = 0
        self.c1 = self.c3 = self.c5 = 0
        self.rr = 0.0
        self.nll = 0.0
        self.brier = 0.0
        self.ndcg5 = 0.0
        self._coords = coords
        self.dist_km_sum = 0.0
        self.dist_km_n = 0
        # Optional columnar capture of per-window ground truth / prediction /
        # rank so the next-node results can be re-analysed offline without
        # re-running evaluation. Caller supplies a dict with list values.
        self._pred_out = predictions_out

    def add(self, cand_nodes: np.ndarray, cand_probs: np.ndarray,
            true_node: int) -> int:
        """Score one prediction; returns the predicted (argmax) node id."""
        self.n += 1
        if cand_nodes.size == 0:
            # Degenerate: no mass anywhere -> miss, uniform-ish penalties.
            self.rr += 0.0
            self.nll += -np.log(1e-300)
            self.brier += 1.0
            if self._pred_out is not None:
                self._pred_out["true"].append(int(true_node))
                self._pred_out["pred"].append(-1)
                self._pred_out["rank"].append(-1)
            return -1

        order = np.argsort(cand_probs)[::-1]
        cand_sorted = cand_nodes[order]
        probs_sorted = cand_probs[order]
        pred = int(cand_sorted[0])

        hit = np.where(cand_nodes == true_node)[0]
        if hit.size:
            p_true = float(cand_probs[hit[0]])
            rank = int(np.sum(cand_probs > p_true)) + 1
        else:
            p_true = 0.0
            rank = int(cand_nodes.size) + 1   # below every candidate

        if rank == 1:
            self.c1 += 1
        if rank <= 3:
            self.c3 += 1
        if rank <= 5:
            self.c5 += 1
        self.rr += 1.0 / rank

        self.nll += -np.log(max(p_true, 1e-300))
        self.brier += float(np.sum(cand_probs ** 2)) - 2.0 * p_true + 1.0
        if rank <= 5:
            self.ndcg5 += 1.0 / np.log2(rank + 1)
        if self._coords is not None:
            d_km = _haversine_km(self._coords, pred, int(true_node))
            if d_km is not None:
                self.dist_km_sum += d_km
                self.dist_km_n += 1
        if self._pred_out is not None:
            self._pred_out["true"].append(int(true_node))
            self._pred_out["pred"].append(pred)
            self._pred_out["rank"].append(int(rank))
        return pred

    def result(self) -> Dict[str, float]:
        n = max(self.n, 1)
        out = {
            "accuracy_top1": self.c1 / n,
            "accuracy_top3": self.c3 / n,
            "accuracy_top5": self.c5 / n,
            "mrr":           self.rr / n,
            "nll":           self.nll / n,
            "brier":         self.brier / n,
            "ndcg5":         self.ndcg5 / n,
            "n_samples":     self.n,
        }
        if self.dist_km_n > 0:
            out["dist_km"] = self.dist_km_sum / self.dist_km_n
        return out


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------

def load_kgram_to_id(path: str) -> Dict[tuple, int]:
    """Load ``kgram_to_id.json`` whose keys are ``"(a, b)"`` tuple strings."""
    import json
    with open(path) as f:
        raw = json.load(f)
    out: Dict[tuple, int] = {}
    for k, v in raw.items():
        out[tuple(ast.literal_eval(k))] = int(v)
    return out


def load_node_to_cluster(path: str, vocab_cat: int) -> np.ndarray:
    """Load ``node_to_cluster.json`` (``{str(node_idx): cluster}``)."""
    import json
    with open(path) as f:
        raw = json.load(f)
    arr = np.zeros(vocab_cat, dtype=np.int64)
    for k, v in raw.items():
        i = int(k)
        if 0 <= i < vocab_cat:
            arr[i] = int(v)
    return arr
