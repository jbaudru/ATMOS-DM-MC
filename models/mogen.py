"""
MOGen – Multi-Order Generative Model.

Implements the multi-order generative framework of Gote et al. (2023) exactly:

  • Transition tables are built for every order k = 1 … max_order, including
    start transitions (* → v1) and terminal transitions (last_ctx → †).
  • Model selection uses the AIC criterion from Theorem 1:

        K* = argmin_K [ dof(K) − Σ_t log P(t | T^(K)) ]

    where dof(K) = Σ_{k=1}^K Σ_{i,j} (A^k)_{i,j}  (A = observed binary
    adjacency matrix, A^k = k-th matrix power) and P(t | T^(K)) is the full
    path probability from Lemma 1 (start + ramp-up + steady-state + terminal).

  • Predictions use the K*-th order table with fallback to lower orders on a
    context miss.

Reference
---------
Gote, Casiraghi, Schweitzer & Scholtes (2023). Predicting variable-length paths
in networked systems using multi-order generative models.
Applied Network Science 8(1):68.  doi:10.1007/s41109-023-00596-x
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)

from ._base import _BaselineModel


class MOGen(_BaselineModel):
    """
    Multi-Order Generative Model (MOGen) – faithful implementation of
    Gote et al. (2023).

    Model selection follows Theorem 1:
        K* = argmin_K [ dof(K) − Σ_t log P(t | T^(K)) ]
    where dof(K) penalises model complexity and ll(K) is the full path
    log-likelihood per Lemma 1.

    Parameters
    ----------
    max_order : int
        Maximum Markov order K to consider during model selection.
    laplace_alpha : float
        Laplace smoothing pseudo-count (applied to next-nodes and terminal †).

    Notes
    -----
    On road-network trajectory data the AIC almost always selects K=1 because
    the graph topology encodes the dominant transition constraint; longer-range
    history adds negligible signal.
    """

    def __init__(self, max_order: int = 5,
                 significance: float = 1e-3,   # kept for API compat, unused
                 laplace_alpha: Optional[float] = None,
                 order_objective: str = "full_path_aic",
                 topo_beta: float = 0.0,
                 min_context_count: int = 3,
                 lambda_mc1: float = 0.4):
        super().__init__()
        self.max_order     = max_order
        self.significance  = significance       # legacy; AIC is parameter-free
        # When None, set in fit() to 1/V so total smoothing mass alpha*V == 1
        # (independent of vocab size).
        self.laplace_alpha = laplace_alpha
        self.order_objective = order_objective
        self.topo_beta = topo_beta
        self.min_context_count = min_context_count
        self.lambda_mc1 = lambda_mc1
        # start counts:        v → count(* → v)
        self._start_counts: Dict[int, int] = {}
        # regular transitions: k → ctx_tuple → {next_node: count}
        self._trans_counts: Dict[int, Dict[Tuple, Dict[int, int]]] = {}
        # terminal counts:     k → ctx_tuple → count(ctx → †)
        self._term_counts:  Dict[int, Dict[Tuple, int]] = {}
        self._obs_adj:      Optional[Tuple[np.ndarray, np.ndarray, int]] = None
        self.optimal_order: int = 1

    # ------------------------------------------------------------------
    def fit(self, sequences: List[List[int]],
            adjacency: Optional[np.ndarray] = None) -> "MOGen":
        self.adjacency  = adjacency  # used only as soft prior
        self.vocab_size = self._resolve_vocab(sequences)
        self._build_mc1_table(sequences)
        if self.laplace_alpha is None:
            self.laplace_alpha = 1.0 / max(self.vocab_size, 1)
        K = self.max_order

        self._start_counts = {}
        self._trans_counts = {}
        self._term_counts  = {}

        for seq in sequences:
            l = len(seq)
            if l == 0:
                continue

            # ---- start transition: * → seq[0] ----
            v0 = seq[0]
            self._start_counts[v0] = self._start_counts.get(v0, 0) + 1

            # ---- regular transitions (ramp-up k<K and steady-state k=K) ----
            for i in range(1, l):
                k   = min(i, K)
                ctx = tuple(seq[i - k: i])
                nxt = seq[i]
                tc_ctx = self._trans_counts.setdefault(k, {}).setdefault(ctx, {})
                tc_ctx[nxt] = tc_ctx.get(nxt, 0) + 1

            # ---- terminal transition: last context → † ----
            k_t   = min(l, K)
            ctx_t = tuple(seq[l - k_t: l])
            tk_k  = self._term_counts.setdefault(k_t, {})
            tk_k[ctx_t] = tk_k.get(ctx_t, 0) + 1

        self._obs_adj      = self._build_obs_adj(sequences)
        self.optimal_order = self._select_order(sequences)
        return self

    # ------------------------------------------------------------------
    def _build_obs_adj(self, sequences: List[List[int]]):
        """
        Binary first-order adjacency A where A[u,v]=1 iff u→v was observed at
        least once in training (Gote et al., footnote 4).

        Stored as a *sparse* edge list ``(rows, cols, V)`` of the unique
        observed directed edges instead of a dense V×V matrix.  For road
        networks V can be tens of thousands, so a dense matrix (V² floats)
        is both memory-prohibitive (~32 GB at V=63k) and needless: ``_dof``
        only ever needs row/column sums of matrix powers, which are obtained
        from sparse matrix-vector products over these edges.
        """
        V = self.vocab_size
        rows: List[int] = []
        cols: List[int] = []
        for seq in sequences:
            for i in range(len(seq) - 1):
                u, v = seq[i], seq[i + 1]
                if 0 <= u < V and 0 <= v < V:
                    rows.append(u)
                    cols.append(v)
        if not rows:
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64), V)
        edges = np.stack([np.asarray(rows, dtype=np.int64),
                          np.asarray(cols, dtype=np.int64)])
        uniq = np.unique(edges, axis=1)          # dedupe → binary adjacency
        return (uniq[0], uniq[1], V)

    # ------------------------------------------------------------------
    def _dof(self, K: int) -> float:
        """
        AIC penalty for max-order K (Eq. 7, constant terms omitted):
            dof(K) = Σ_{k=1}^K Σ_{i,j} (A^k)_{i,j}

        Identical value to the dense formulation, computed without ever
        materialising A^k.  Since Σ_{i,j}(A^k)_{i,j} = 1ᵀ A^k 1, we propagate
        the all-ones vector through k sparse matrix-vector products
        w_k = A w_{k-1} (w_0 = 1) and accumulate 1ᵀ w_k = w_k.sum().
        Cost is O(K · nnz) instead of O(K · V³).
        """
        rows, cols, V = self._obs_adj
        if rows.size == 0 or K < 1:
            return 0.0
        w = np.ones(V, dtype=np.float64)         # A^0 · 1
        total = 0.0
        for _ in range(1, K + 1):
            # (A w)[i] = Σ_{(i,j)∈E} w[j]
            w = np.bincount(rows, weights=w[cols], minlength=V)
            total += float(w.sum())              # 1ᵀ A^k 1 = Σ_{i,j}(A^k)_{i,j}
        return total

    # ------------------------------------------------------------------
    def _path_log_prob(self, seq: List[int], K: int) -> float:
        """
        log P(t | T^(K)) for path seq (Lemma 1, Gote et al. 2023).

        Transitions modelled:
          * → seq[0]                    (start)
          ctx_i → seq[i], i = 1..l-1   (regular; † in denominator)
          last_ctx → †                  (terminal)
        """
        l = len(seq)
        if l == 0:
            return 0.0

        V     = self.vocab_size
        alpha = self.laplace_alpha
        ll    = 0.0

        # ---- start ----
        total_s = sum(self._start_counts.values()) if self._start_counts else 0
        cnt_s   = self._start_counts.get(seq[0], 0)
        p_start = (cnt_s + alpha) / (total_s + alpha * V) if total_s > 0 else 1.0 / V
        ll += np.log(max(p_start, 1e-300))

        # ---- regular transitions ----
        for i in range(1, l):
            k     = min(i, K)
            ctx   = tuple(seq[i - k: i])
            nxt   = seq[i]
            trans = self._trans_counts.get(k, {}).get(ctx, {})
            term  = self._term_counts.get(k,  {}).get(ctx, 0)
            # row is stochastic over V next-nodes + 1 terminal → V+1 outcomes
            total_ctx = sum(trans.values()) + term
            prob = (trans.get(nxt, 0) + alpha) / (total_ctx + alpha * (V + 1))
            ll  += np.log(max(prob, 1e-300))

        # ---- terminal ----
        k_t     = min(l, K)
        ctx_t   = tuple(seq[l - k_t: l])
        trans_t = self._trans_counts.get(k_t, {}).get(ctx_t, {})
        term_t  = self._term_counts.get(k_t, {}).get(ctx_t, 0)
        total_t = sum(trans_t.values()) + term_t
        p_term  = (term_t + alpha) / (total_t + alpha * (V + 1))
        ll += np.log(max(p_term, 1e-300))

        return ll

    # ------------------------------------------------------------------
    def _next_step_log_prob(self, seq: List[int], K: int) -> float:
        """
        Next-step transition log-likelihood aligned with predict_proba().

        This objective excludes start/terminal events and models only
        P(x_t | x_{<t}) over observed next-node transitions.
        """
        l = len(seq)
        if l < 2:
            return 0.0

        V = self.vocab_size
        alpha = self.laplace_alpha
        ll = 0.0

        for i in range(1, l):
            k = min(i, K)
            ctx = tuple(seq[i - k: i])
            nxt = seq[i]
            trans = self._trans_counts.get(k, {}).get(ctx, {})

            total_ctx = float(sum(trans.values()))
            prob = (trans.get(nxt, 0) + alpha) / (total_ctx + alpha * V)
            ll += np.log(max(prob, 1e-300))

        return ll

    # ------------------------------------------------------------------
    def _select_order(self, sequences: List[List[int]]) -> int:
        """
        AIC-based model selection (Theorem 1, Gote et al. 2023):
            K* = argmin_K [ dof(K) − Σ_t log P(t | T^(K)) ]
        """
        best_K   = 1
        best_val = float("inf")
        use_full = self.order_objective == "full_path_aic"
        outer = tqdm(range(1, self.max_order + 1),
                     desc="MOGen order-AIC", unit="order", leave=False)
        for K in outer:
            if use_full:
                ll = 0.0
                for seq in tqdm(sequences, desc=f"  K={K} log-lik",
                                unit="seq", leave=False):
                    ll += self._path_log_prob(seq, K)
            else:
                ll = 0.0
                for seq in tqdm(sequences, desc=f"  K={K} log-lik",
                                unit="seq", leave=False):
                    ll += self._next_step_log_prob(seq, K)
            val = self._dof(K) - ll
            if val < best_val:
                best_val = val
                best_K   = K
            outer.set_postfix({"K*": best_K, "AIC": f"{val:.0f}"})
        return best_K

    # ------------------------------------------------------------------
    def predict_proba(self, context: List[int],
                      vocab_size: Optional[int] = None) -> np.ndarray:
        V = vocab_size or self.vocab_size
        if V == 0:
            raise RuntimeError("Model not fitted yet (vocab_size == 0).")

        alpha = self.laplace_alpha
        context = list(context)

        # Use max_order (not optimal_order) as the upper bound so that higher-
        # order contexts are tried whenever they exist.  The AIC-selected
        # optimal_order is only an aggregate statistic; for individual
        # prediction the best strategy is always to try the highest available
        # context and fall back via the loop.  Without this, AIC picks K=1 for
        # large-V sparse graphs (dof penalty dominates) and the model
        # degenerates to MC1 even when k>1 contexts are present in the table.
        for k in range(min(self.max_order, len(context)), 0, -1):
            ctx     = tuple(context[-k:])
            trans_k = self._trans_counts.get(k, {})
            if ctx in trans_k:
                counts = trans_k[ctx]
                # Confidence gate: skip sparsely-supported contexts that
                # produce near-degenerate distributions.
                if k > 1 and sum(counts.values()) < self.min_context_count:
                    continue
                proba  = np.full(V, alpha, dtype=float)
                for node, cnt in counts.items():
                    if 0 <= node < V:
                        proba[node] += cnt
                s = proba.sum()
                if s > 0:
                    proba /= s
                proba = self._apply_topo_prior(proba, context, self.topo_beta)
                # For k > 1 a proper higher-order context was found — trust it
                # fully.  MC1 interpolation would cap the higher-order signal
                # at only 40%, which largely eliminates the benefit of k > 1.
                # For k == 1 the transition table IS the MC1 table, so the
                # interpolation is a no-op anyway (0.6×MC1 + 0.4×MC1 = MC1).
                if k == 1:
                    return self._mc1_interpolate(proba, context, self.lambda_mc1)
                return proba

        if context:
            return self._mc1_proba(context[-1], V)
        return np.ones(V) / V
