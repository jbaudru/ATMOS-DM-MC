"""
Abstract base class shared by all baseline models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class _BaselineModel(ABC):
    """Abstract base – shared helpers and API contract."""

    def __init__(self):
        self.vocab_size: int = 0
        self.adjacency: Optional[np.ndarray] = None
        # First-order empirical transition counts, populated by
        # `_build_mc1_table()` (called from each subclass `fit`).  Used as a
        # Jelinek-Mercer back-off floor in `_mc1_interpolate`.
        self._mc1_counts: Dict[int, Dict[int, int]] = {}
        self._mc1_totals: Dict[int, int] = {}

    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, sequences: List[List[int]],
            adjacency: Optional[np.ndarray] = None) -> "_BaselineModel":
        ...

    @abstractmethod
    def predict_proba(self, context: List[int],
                      vocab_size: Optional[int] = None) -> np.ndarray:
        ...

    def predict(self, context: List[int],
                vocab_size: Optional[int] = None) -> int:
        return int(np.argmax(self.predict_proba(context, vocab_size)))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _resolve_vocab(self, sequences: List[List[int]]) -> int:
        nodes = {n for seq in sequences for n in seq}
        return (max(nodes) + 1) if nodes else 0

    @staticmethod
    def _align_mask(mask: np.ndarray, V: int) -> np.ndarray:
        """Clip or zero-pad adjacency row to length V."""
        if len(mask) == V:
            return mask
        aligned = np.zeros(V)
        aligned[:min(len(mask), V)] = mask[:V]
        return aligned

    def _apply_adjacency(self, proba: np.ndarray,
                         context: List[int]) -> np.ndarray:
        """Zero-out topologically invalid transitions and renormalise."""
        if self.adjacency is not None and context:
            curr = context[-1]
            if 0 <= curr < len(self.adjacency):
                mask = self._align_mask(self.adjacency[curr].astype(float), len(proba))
                masked = proba * mask
                s = masked.sum()
                if s > 0:
                    return masked / s
        return proba

    def _neighbors_uniform(self, V: int, context: List[int]) -> np.ndarray:
        """Uniform distribution restricted to topological neighbours, or
        global uniform when no adjacency is available."""
        if self.adjacency is not None and context:
            curr = context[-1]
            if 0 <= curr < len(self.adjacency):
                mask = self.adjacency[curr].astype(float)
                s = mask.sum()
                if s > 0:
                    out = np.zeros(V)
                    out[:len(mask)] = mask / s
                    return out
        return np.ones(V) / V

    def _apply_topo_prior(self, proba: np.ndarray,
                          context: List[int],
                          beta: float) -> np.ndarray:
        """Soft topology prior: multiply proba by  beta*mask + (1-beta)*uniform.

        Unlike `_apply_adjacency` (hard gate, zeros off-graph nodes), this
        downweights but never eliminates them, so unseen edges remain
        recoverable.  beta=1 -> hard gate, beta=0 -> no prior.
        """
        if beta <= 0.0 or self.adjacency is None or not context:
            return proba
        curr = context[-1]
        if not (0 <= curr < len(self.adjacency)):
            return proba
        mask = self._align_mask(self.adjacency[curr].astype(float), len(proba))
        s = mask.sum()
        if s <= 0:
            return proba
        mask = mask / s
        uni  = np.full_like(proba, 1.0 / len(proba))
        prior = beta * mask + (1.0 - beta) * uni
        out = proba * prior
        z = out.sum()
        return out / z if z > 0 else proba

    # ------------------------------------------------------------------
    # MC1 floor (Jelinek-Mercer back-off)
    # ------------------------------------------------------------------
    def _build_mc1_table(self, sequences: List[List[int]]) -> None:
        """Populate `_mc1_counts[curr][next] = count` from training data.

        Vectorised: collect all (curr, next) pairs with numpy per sequence,
        then collapse to unique pairs with counts in one C-level pass.  The
        Python loop runs over distinct observed transitions (≪ total
        transitions on repetitive routed data), not every transition.  The
        resulting tables are identical to the per-transition build.
        """
        counts: Dict[int, Dict[int, int]] = {}
        totals: Dict[int, int] = {}
        _u_parts, _v_parts = [], []
        for seq in sequences:
            arr = np.asarray(seq, dtype=np.intp)
            if arr.size < 2:
                continue
            _u_parts.append(arr[:-1])
            _v_parts.append(arr[1:])
        if _u_parts:
            pairs = np.stack([np.concatenate(_u_parts),
                              np.concatenate(_v_parts)], axis=1)
            uniq, cnts = np.unique(pairs, axis=0, return_counts=True)
            for (u, v), c in zip(uniq, cnts):
                u, v, c = int(u), int(v), int(c)
                row = counts.get(u)
                if row is None:
                    row = {}
                    counts[u] = row
                row[v] = c
                totals[u] = totals.get(u, 0) + c
        self._mc1_counts = counts
        self._mc1_totals = totals

    def _mc1_proba(self, curr: int, V: int) -> np.ndarray:
        """Return MC1 next-node distribution for `curr`, uniform if unseen."""
        row = self._mc1_counts.get(curr)
        tot = self._mc1_totals.get(curr, 0)
        if not row or tot <= 0:
            return np.ones(V) / V
        out = np.zeros(V)
        for v, c in row.items():
            if 0 <= v < V:
                out[v] = c
        s = out.sum()
        return out / s if s > 0 else np.ones(V) / V

    def _mc1_interpolate(self, proba: np.ndarray,
                         context: List[int],
                         lam: float) -> np.ndarray:
        """Linear (Jelinek-Mercer) interpolation:
            (1 - lam) * MC1(curr) + lam * proba
        Guarantees a hard MC1 floor: high-order signal can only add a small
        bonus on top of empirical first-order transitions.
        """
        if lam >= 1.0 or not context:
            return proba
        curr = context[-1]
        mc1 = self._mc1_proba(curr, len(proba))
        out = (1.0 - lam) * mc1 + lam * proba
        s = out.sum()
        return out / s if s > 0 else proba
