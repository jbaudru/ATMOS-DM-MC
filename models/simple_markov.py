"""
SimpleMarkov – Multi-order Markov chain with entropy-weighted order merging.

Inspired by IDyOM's combination scheme: at prediction time, compute the
empirical next-node distribution at every order k = 1 .. order whose context
was actually observed during training, then merge them with weights
inversely proportional to each distribution's Shannon entropy.

Lower-entropy (more confident) orders therefore dominate the mixture, while
high-entropy orders contribute little.  When no order matches, the model
falls back to the unigram distribution and finally to a uniform.

Compared with a fixed-order MC:
- MC(k) with no back-off goes uniform on every unseen k-gram (V^k explosion).
- This implementation uses *all* orders that have evidence, weighted by their
  certainty.  This makes it a strong, IDyOM-flavoured baseline that lets us
  measure how much extra value the full DM-MC / GraphIDyOM machinery
  contributes on top of plain entropy-weighted order merging.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ._base import _BaselineModel


class SimpleMarkov(_BaselineModel):
    """
    Multi-order Markov chain with entropy-weighted merging.

    At fit time, transition counts are stored for every order k = 1..order
    (plus the unigram).  At prediction time:

        1. For each k, look up the empirical next-token distribution
           p_k(.| ctx_{-k:}).  Skip orders whose context has zero training
           support.
        2. Compute the Shannon entropy H_k of every retained p_k.
        3. Combine:  p = sum_k w_k * p_k,
                     w_k proportional to 1 / (H_k + eps),  sum_k w_k = 1.
        4. Fall back to unigram, then uniform, if no order matched.

    Parameters
    ----------
    order : int
        Maximum context length stored / used.  Setting ``order=10`` therefore
        merges up to 10 distributions (orders 1..10) at prediction time.
    count_discount : float
        Exponent ``lambda >= 0`` of the evidence-confidence discount.
    count_tau : float
        Saturation scale ``tau > 0`` for support confidence. We use
        ``conf_k = (N_k / (N_k + tau))^lambda`` and
        ``w_k ~ conf_k / (H_k + eps)``. This strongly discounts tiny-support
        orders while preventing huge counts from fully overwhelming other
        orders. ``count_discount=0`` recovers the original inverse-entropy rule.
    """

    _EPS = 1e-6  # numerical floor on entropy weights

    def __init__(self, order: int = 1,
                 count_discount: float = 1.0,
                 count_tau: float = 8.0,
                 blend: bool = True):
        super().__init__()
        if order < 1:
            raise ValueError("order must be >= 1")
        if count_discount < 0:
            raise ValueError("count_discount must be >= 0")
        if count_tau <= 0:
            raise ValueError("count_tau must be > 0")
        self.order = order
        self._count_discount = float(count_discount)
        self._count_tau = float(count_tau)
        self.blend = blend
        # _counts[k] : Dict[ k-gram tuple, Dict[next_node, count] ]
        self._counts: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {}
        # Unigram distribution (k = 0)
        self._unigram: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def fit(self, sequences: List[List[int]],
            adjacency: Optional[np.ndarray] = None) -> "SimpleMarkov":
        self.adjacency = None   # topology masking disabled by design
        self.vocab_size = self._resolve_vocab(sequences)

        self._counts = {k: {} for k in range(1, self.order + 1)}
        self._unigram = {}

        for seq in sequences:
            n = len(seq)
            for i in range(1, n):
                tgt = seq[i]
                self._unigram[tgt] = self._unigram.get(tgt, 0) + 1
                for k in range(1, self.order + 1):
                    if i - k < 0:
                        break
                    ctx = tuple(seq[i - k: i])
                    table = self._counts[k]
                    row = table.get(ctx)
                    if row is None:
                        row = {}
                        table[ctx] = row
                    row[tgt] = row.get(tgt, 0) + 1
        return self

    # ------------------------------------------------------------------
    def _proba_from_counts(self, counts: Dict[int, int], V: int) -> np.ndarray:
        proba = np.zeros(V, dtype=np.float64)
        for node, cnt in counts.items():
            if 0 <= node < V:
                proba[node] = cnt
        s = proba.sum()
        return proba / s if s > 0 else proba

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        nz = p > 0
        if not np.any(nz):
            return 0.0
        return float(-np.sum(p[nz] * np.log2(p[nz])))

    # ------------------------------------------------------------------
    def predict_proba(self, context: List[int],
                      vocab_size: Optional[int] = None) -> np.ndarray:
        V = vocab_size or self.vocab_size
        ctx_full = list(context)
        max_k = min(self.order, len(ctx_full))

        if not self.blend:
            # Pure greedy backoff: return the distribution from the highest
            # matching order without mixing lower orders.  Falls back through
            # k-1, k-2, … 1, then unigram, then uniform.
            for k in range(max_k, 0, -1):
                ctx = tuple(ctx_full[-k:])
                row = self._counts.get(k, {}).get(ctx)
                if row:
                    p = self._proba_from_counts(row, V)
                    if p.sum() > 0:
                        return p
            if self._unigram:
                return self._proba_from_counts(self._unigram, V)
            return np.ones(V, dtype=np.float64) / V

        # Collect (distribution, entropy, support count) for every order whose
        # context was seen during training.
        per_order: List[Tuple[np.ndarray, float, float]] = []
        for k in range(1, max_k + 1):
            ctx = tuple(ctx_full[-k:])
            row = self._counts.get(k, {}).get(ctx)
            if row:
                p_k = self._proba_from_counts(row, V)
                if p_k.sum() > 0:
                    n_k = float(sum(row.values()))
                    per_order.append((p_k, self._entropy(p_k), n_k))

        if per_order:
            # Count-discounted entropy-weighted mixture:
            #   conf_k = (N_k / (N_k + tau))^lambda
            #   w_k    ~ conf_k / (H_k + eps)
            weights = np.array(
                [((n / (n + self._count_tau)) ** self._count_discount) / (h + self._EPS)
                 for _, h, n in per_order],
                dtype=np.float64,
            )
            wsum = weights.sum()
            if wsum > 0:
                weights /= wsum
                mix = np.zeros(V, dtype=np.float64)
                for w, (p_k, _, _) in zip(weights, per_order):
                    mix += w * p_k
                s = mix.sum()
                if s > 0:
                    return mix / s

        # Unigram fallback
        if self._unigram:
            return self._proba_from_counts(self._unigram, V)

        # Final fallback: uniform
        return np.ones(V, dtype=np.float64) / V
