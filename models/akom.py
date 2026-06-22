"""
AKOM – All K-th Order Markov model.

At training time, counts n-gram transitions for every order k in [1, max_order].
At prediction time, uses *weighted specificity*: every context suffix of length k
that was seen during training contributes its conditional distribution, weighted
by k.  The final distribution is the normalised weighted sum across all matching
orders.  This matches the prediction rule described in Pitkow & Pirolli (1999)
Section 6.2 and improves Acc@N / MRR by leveraging shorter-context signal when
the longest context is sparse.

Reference
---------
Pitkow & Pirolli (1999). Mining longest repeated subsequences to predict
world wide web surfing. USENIX USITS'99.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ._base import _BaselineModel


class AKOM(_BaselineModel):
    """
    All K-th Order Markov model with weighted-specificity prediction.

    Parameters
    ----------
    max_order : int
        Maximum Markov order (context length) to store.
    smoothing : float
        Additive (Laplace-like) smoothing applied to observed transition counts
        before normalisation.

    Notes
    -----
    On road networks the dominant signal is almost always first-order (the
    next node is constrained by the current node's graph neighbours).  Higher
    orders give marginal gains at the cost of data sparsity: a k=10 context
    over a vocabulary of 1 000+ nodes is rarely repeated.  Weighted specificity
    handles this gracefully: sparse high-order distributions are diluted by
    denser low-order ones, yielding better-calibrated probabilities and higher
    Acc@N / MRR than a hard longest-match back-off.
    """

    def __init__(self, max_order: int = 10, smoothing: float = 1e-3,
                 support_tau: float = 8.0, min_context_count: int = 3,
                 topo_beta: float = 0.0,
                 lambda_mc1: float = 0.4):
        super().__init__()
        self.max_order = max_order
        self.smoothing = smoothing
        self.support_tau = support_tau
        self.min_context_count = min_context_count
        self.topo_beta = topo_beta
        self.lambda_mc1 = lambda_mc1
        # {context_tuple: {next_node: raw_count}}
        self._counts: Dict[Tuple, Dict[int, int]] = {}

    # ------------------------------------------------------------------
    def fit(self, sequences: List[List[int]],
            adjacency: Optional[np.ndarray] = None) -> "AKOM":
        self.adjacency = adjacency
        self.vocab_size = self._resolve_vocab(sequences)
        self._build_mc1_table(sequences)
        self._counts = {}

        for seq in sequences:
            n = len(seq)
            for i in range(1, n):
                target = seq[i]
                for k in range(1, min(self.max_order + 1, i + 1)):
                    ctx = tuple(seq[i - k: i])
                    if ctx not in self._counts:
                        self._counts[ctx] = {}
                    self._counts[ctx][target] = self._counts[ctx].get(target, 0) + 1

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, context: List[int],
                      vocab_size: Optional[int] = None) -> np.ndarray:
        V = vocab_size or self.vocab_size
        if V == 0:
            raise RuntimeError("Model not fitted yet (vocab_size == 0).")

        context = list(context)

        # Pitkow & Pirolli (1999) weighted-specificity rule with confidence
        # gate: each matching context suffix of length k contributes its
        # conditional distribution weighted linearly by k, but only if its
        # support exceeds `min_context_count` (sparse contexts produce noisy,
        # near-degenerate distributions that hurt the weighted average).
        scores = np.zeros(V)
        total_weight = 0.0

        for k in range(min(self.max_order, len(context)), 0, -1):
            ctx = tuple(context[-k:])
            if ctx not in self._counts:
                continue
            counts = self._counts[ctx]
            support = float(sum(counts.values()))
            if support < self.min_context_count:
                continue

            proba_k = np.full(V, self.smoothing, dtype=float)
            for node, cnt in counts.items():
                if 0 <= node < V:
                    proba_k[node] += cnt
            s = float(proba_k.sum())
            if s > 0:
                proba_k /= s
                weight = float(k)
                scores += weight * proba_k
                total_weight += weight

        if total_weight > 0:
            s = scores.sum()
            if s > 0:
                scores /= s
            scores = self._apply_topo_prior(scores, context, self.topo_beta)
            return self._mc1_interpolate(scores, context, self.lambda_mc1)

        # No high-order match -> fall back to pure MC1.
        if context:
            return self._mc1_proba(context[-1], V)
        return np.ones(V) / V
