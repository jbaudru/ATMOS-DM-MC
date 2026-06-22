"""
CPT+ – Compact Prediction Tree Plus.

Stores all training sequences and an inverted index mapping each node to the
set of training sequences containing it.  At prediction time it selects
candidate sequences that share the current context, finds the
immediately-following element in each candidate, and aggregates into a
probability distribution.  Falls back to shorter contexts on a miss.

Reference
---------
Gueniche, Fournier-Viger & Tseng (2013). Compact prediction tree: A lossless
model for accurate sequence prediction. ADMA 2013.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ._base import _BaselineModel


class CPTPlus(_BaselineModel):
    """
    Compact Prediction Tree Plus (CPT+).

    Parameters
    ----------
    max_context_length : int
        Maximum history length to use when looking for similar sequences.
    noise_threshold : int
        Minimum occurrence count for a consequent to be included in the
        prediction (CPT+ noise-reduction heuristic).

    Notes
    -----
    The "+"-improvements over plain CPT are:
      * decreasing-context fallback (try length L, L-1, … until a match)
      * noise-reduction threshold to filter rare consequents

    On dense, short trajectories (e.g. road network paths of ~10 nodes) the
    model effectively reduces to a 1st-order lookup because longer contexts
    are rarely shared across training sequences.
    """

    def __init__(self, max_context_length: int = 5,
                 noise_threshold: int = 2,
                 topo_beta: float = 0.0,
                 lambda_mc1: float = 0.5):
        super().__init__()
        self.max_context_length = max_context_length
        self.noise_threshold = noise_threshold
        self.topo_beta = topo_beta
        self.lambda_mc1 = lambda_mc1
        self._sequences: List[List[int]] = []
        # {node: set of sequence indices that contain that node}
        self._inv_index: Dict[int, set] = {}

    # ------------------------------------------------------------------
    def fit(self, sequences: List[List[int]],
            adjacency: Optional[np.ndarray] = None) -> "CPTPlus":
        self.adjacency = adjacency
        self.vocab_size = self._resolve_vocab(sequences)
        self._build_mc1_table(sequences)
        self._sequences = [list(s) for s in sequences]
        self._inv_index = {}

        for idx, seq in enumerate(self._sequences):
            for node in set(seq):
                if node not in self._inv_index:
                    self._inv_index[node] = set()
                self._inv_index[node].add(idx)

        return self

    # ------------------------------------------------------------------
    def _consequents(self, context: List[int]) -> Dict[int, float]:
        """
        Contiguous suffix-match vote with decreasing-context fallback
        (Gueniche, Fournier-Viger & Tseng 2013, simplified prediction tree).

        For each candidate sequence containing the last context item, count
        every contiguous occurrence of the queried context and add the
        immediately-following element to the score table.  Decreasing context
        is handled by the caller (predict_proba) via the recursive divider
        loop.
        """
        if not context:
            return {}

        last = context[-1]
        candidates = self._inv_index.get(last)
        if candidates is None:
            return {}

        L = len(context)
        count_table: Dict[int, int] = {}

        for seq_idx in candidates:
            seq = self._sequences[seq_idx]
            if len(seq) <= L:
                continue
            for end in range(L - 1, len(seq) - 1):
                start = end - L + 1
                if seq[start:end + 1] == context:
                    next_elem = seq[end + 1]
                    count_table[next_elem] = count_table.get(next_elem, 0) + 1

        if self.noise_threshold > 1:
            count_table = {k: v for k, v in count_table.items()
                           if v >= self.noise_threshold}

        return count_table

    # ------------------------------------------------------------------
    def predict_proba(self, context: List[int],
                      vocab_size: Optional[int] = None) -> np.ndarray:
        V = vocab_size or self.vocab_size
        if V == 0:
            raise RuntimeError("Model not fitted yet (vocab_size == 0).")

        context = list(context[-self.max_context_length:])

        for ctx_len in range(len(context), 0, -1):
            ctx = context[-ctx_len:]
            counts = self._consequents(ctx)
            if counts:
                total = sum(counts.values())
                proba = np.zeros(V)
                for node, cnt in counts.items():
                    if 0 <= node < V:
                        proba[node] = cnt / total
                proba = self._apply_topo_prior(proba, context, self.topo_beta)
                return self._mc1_interpolate(proba, context, self.lambda_mc1)

        # No context match found at any length -> pure MC1 fallback.
        if context:
            return self._mc1_proba(context[-1], V)
        return np.ones(V) / V
