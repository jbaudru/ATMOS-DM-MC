"""
IOHMM – Input-Output Hidden Markov Model.

Hidden states represent latent activity patterns (e.g. commuter vs. leisure
traveller).  Trained with the Baum-Welch EM algorithm on trajectory data.
Prediction combines the inferred state distribution with per-state emission
preferences, scaled by empirical first-order transition counts so that the
model remains position-aware.

Reference
---------
Mo, Zhao, Koutsopoulos & Zhao (2022). Individual mobility prediction in mass
transit systems using smart card data.
IEEE Transactions on ITS 23(8):12014-12026.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, diags

from ._base import _BaselineModel

# Use Mogeng/IOHMM's reference forward-backward when the package is installed
# (pip install IOHMM).  Falls back to the built-in implementation otherwise.
# The emission model and M-step remain custom: Mogeng's GLM/regression emissions
# are incompatible with the factored B[s,next]×adj[curr,next] parameterisation.
try:
    from IOHMM.forward_backward import (
        forward_backward as _mogeng_forward_backward,
        forward         as _mogeng_forward,
    )
    _MOGENG_AVAILABLE: bool = True
except ImportError:
    _MOGENG_AVAILABLE = False

# GPU acceleration via PyTorch when a CUDA device is available.
# Falls back to the numpy CPU path silently if torch is not installed.
try:
    import torch as _torch
    _TORCH_CUDA: bool = _torch.cuda.is_available()
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_CUDA = False

# Maximum memory (bytes) allowed for the full (V, S, V) emission table.
# Above this the E-step falls back to a per-unique-node sparse cache.
_EM_ALL_MEMORY_LIMIT: int = 256 * 1024 * 1024   # 256 MB

# Number of sequences processed per mini-batch in the E-step.
# GPU path uses a larger batch: the inner kernel is tiny (Nb, S, S) so
# larger batches improve GPU utilisation without exceeding VRAM.
# At Nb=20000, T=47, S=10: peak tensor is (20000,46,10,10)×4B ≈ 368MB ≪ 8GB.
_GPU_E_STEP_BATCH_SIZE: int = 20_000
# At S=5, T=47: batch=5000 → ~9MB per array (forward/backward/obs each).
_E_STEP_BATCH_SIZE: int = 5_000

# ---------------------------------------------------------------------------
# torch.compile wrappers for forward / backward passes.
# Compiled once on first GPU E-step call; subsequent calls reuse the kernel.
# The for-t loop is fused by the Inductor/Triton backend into a single CUDA
# kernel, eliminating Python overhead between time steps.
# ---------------------------------------------------------------------------
_compiled_forward  = None
_compiled_backward = None


def _make_compiled_fb() -> None:
    """Build torch.compile'd forward/backward kernels (idempotent).

    Requires Triton (pip install triton) for kernel fusion; falls back to
    eager PyTorch ops silently when Triton is not installed.
    """
    global _compiled_forward, _compiled_backward
    if _compiled_forward is not None:
        return

    def _fwd(lob, lA_T, lpi):
        N, T, S = lob.shape
        la = _torch.empty(N, T, S, device=lob.device, dtype=lob.dtype)
        la[:, 0] = lpi[None, :] + lob[:, 0]
        for t in range(1, T):
            a = lA_T[None] + la[:, t - 1, None, :]   # (N, S, S)
            m = a.max(dim=-1, keepdim=True).values
            la[:, t] = (_torch.log(_torch.exp(a - m).sum(dim=-1))
                        + m.squeeze(-1) + lob[:, t])
        return la

    def _bwd(lob, lA):
        N, T, S = lob.shape
        lb = _torch.zeros(N, T, S, device=lob.device, dtype=lob.dtype)
        for t in range(T - 2, -1, -1):
            rhs = lb[:, t + 1] + lob[:, t + 1]       # (N, S)
            a   = lA[None] + rhs[:, None, :]          # (N, S, S)
            m   = a.max(dim=-1, keepdim=True).values
            lb[:, t] = (_torch.log(_torch.exp(a - m).sum(dim=-1))
                        + m.squeeze(-1))
        return lb

    # torch.compile only gives a speed benefit when Triton is available to
    # fuse the loop into a single CUDA kernel; without it, fall back to eager.
    _use_compile = False
    if hasattr(_torch, 'compile'):
        try:
            import triton as _triton  # noqa: F401
            _use_compile = True
        except ImportError:
            pass

    if _use_compile:
        try:
            _compiled_forward  = _torch.compile(_fwd,  dynamic=True)
            _compiled_backward = _torch.compile(_bwd,  dynamic=True)
        except Exception:
            _compiled_forward, _compiled_backward = _fwd, _bwd
    else:
        _compiled_forward, _compiled_backward = _fwd, _bwd


class IOHMM(_BaselineModel):
    """
    Input-Output Hidden Markov Model (IOHMM).

    Parameters
    ----------
    n_states : int
        Number of hidden activity states.
    max_iter : int
        Maximum number of Baum-Welch EM iterations.
    tol : float
        Convergence tolerance on per-observation log-likelihood change.
    random_state : int
        Seed for reproducibility.

    Model parameters
    ----------------
    π  – initial state distribution, shape (n_states,)
    A  – state transition matrix, shape (n_states, n_states)
    B  – node-preference log-scores, shape (n_states, vocab_size)

    Emission probability (road-topology-aware):
        P(next | state s, current node v) ∝ exp(B[s, next]) × adjacency[v, next]

    This factored parameterisation keeps the parameter count at
        O(n_states² + n_states × vocab_size)
    instead of the naïve O(n_states × vocab_size²).
    """

    def __init__(self, n_states: int = 5, max_iter: int = 30,
                 tol: float = 1e-4, random_state: int = 42,
                 auto_state_cap: bool = True,
                 trans_prior: float = 0.1,
                 emission_prior: float = 2.0,
                 topo_beta: float = 0.7,
                 lambda_mc1: float = 0.4,
                 state_tilt: float = 0.05):
        super().__init__()
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.auto_state_cap = auto_state_cap
        self.trans_prior = trans_prior
        self.emission_prior = emission_prior
        self.topo_beta = topo_beta
        self.lambda_mc1 = lambda_mc1
        # Maximum relative re-weighting the latent state may apply to an MC1
        # neighbour, as a bounded multiplicative tilt in [1-state_tilt,
        # 1+state_tilt].  Because IOHMM's emission is position-independent
        # (each state is just a global P(next|state) popularity), the raw state
        # preference varies by orders of magnitude and, if used multiplicatively
        # or via an exponent, inverts the ranking among MC1 neighbours
        # (Acc@3≈1 but Acc@1<MC1).  Bounding the tilt to ±5% means the state can
        # only re-order MC1 neighbours whose probabilities are within ~5% of
        # each other (genuine near-ties); it can never overturn a real MC1
        # difference, so IOHMM's ranking tracks the strong MC1 baseline.
        self.state_tilt = state_tilt

        self.pi: Optional[np.ndarray] = None
        self.A:  Optional[np.ndarray] = None
        self.B:  Optional[np.ndarray] = None
        self._transition = None   # sparse CSR (V, V), row-normalised

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        S, V = self.n_states, self.vocab_size
        self.pi = self.rng.dirichlet(np.ones(S))
        self.A  = self.rng.dirichlet(np.ones(S), size=S)
        if self._transition is not None:
            marginal    = np.asarray(self._transition.mean(axis=0)).ravel()
            log_marginal = np.log(marginal + 1e-10)
            noise = self.rng.standard_normal((S, V)) * 1.5
            self.B = log_marginal[np.newaxis, :] + noise
        else:
            self.B = self.rng.standard_normal((S, V)) * 1.0

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def _emission_matrix(self, current_node: int) -> np.ndarray:
        """Return P(next | state=s) as shape (S, V) — position-independent.

        MC1 topology is applied separately in predict_proba so that the
        M-step (which optimises a position-independent marginal) and the
        E-step are consistent.  Hard road-graph masks are still applied here
        when a full adjacency matrix is available.
        """
        log_scores = self.B
        shifted = log_scores - log_scores.max(axis=1, keepdims=True)
        proba = np.exp(shifted)

        if (self.adjacency is not None and
                0 <= current_node < len(self.adjacency)):
            mask = self._align_mask(
                self.adjacency[current_node].astype(float), proba.shape[1])
            proba *= mask[np.newaxis, :]
        # NOTE: MC1 (_transition) is no longer applied here.  It is gated in
        # predict_proba so that training (position-independent M-step) and
        # inference use the same generative model.

        row_sums = proba.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).squeeze()
        if zero_rows.any():
            proba[zero_rows] = 1.0
            row_sums[zero_rows] = proba.shape[1]
        proba /= row_sums
        return proba

    def _log_obs_seq(self, seq: List[int],
                     _em_cache: Optional[Dict[int, np.ndarray]] = None,
                     _em_all:   Optional[np.ndarray] = None) -> np.ndarray:
        """Return log observation likelihoods, shape (T-1, n_states).

        Fast paths (used during training):
          _em_all   – precomputed (V, S, V) table → single numpy gather, no loop.
          _em_cache – dict {node: (S, V)} → loop over unique nodes, not over T.
        Slow path (used during predict_proba): original per-step loop.
        """
        T = len(seq)
        if T < 2:
            return np.empty((0, self.n_states))
        V = self.vocab_size
        log_obs = np.full((T - 1, self.n_states), -np.log(self.n_states))

        if _em_all is not None:
            # Fully vectorised: fancy-index the (V, S, V) table once.
            arr   = np.asarray(seq, dtype=np.intp)
            curr  = arr[:-1]                          # (T-1,)
            nxt   = arr[1:]                           # (T-1,)
            valid = (curr >= 0) & (curr < V) & (nxt >= 0) & (nxt < V)
            if valid.any():
                vc, vn = curr[valid], nxt[valid]
                # _em_all[vc, :, vn] → shape (n_valid, S)  [numpy pairs vc[i] with vn[i]]
                log_obs[valid] = np.log(np.maximum(_em_all[vc, :, vn], 1e-300))

        elif _em_cache:
            # Per-node cache: loop over unique current nodes (≪ T).
            arr       = np.asarray(seq, dtype=np.intp)
            curr      = arr[:-1]
            nxt       = arr[1:]
            valid_nxt = (nxt >= 0) & (nxt < V)
            for v, em in _em_cache.items():
                rows = (curr == v) & valid_nxt
                if rows.any():
                    # em[:, nxt[rows]] → (S, n_rows); .T → (n_rows, S)
                    log_obs[rows] = np.log(np.maximum(em[:, nxt[rows]].T, 1e-300))

        else:
            # Original per-step loop (used in predict_proba for single queries).
            for t in range(T - 1):
                curr, nxt = seq[t], seq[t + 1]
                if curr >= V or nxt >= V or curr < 0 or nxt < 0:
                    continue                           # stays at uniform default
                em = self._emission_matrix(curr)[:, nxt]   # shape (S,)
                log_obs[t] = np.log(np.maximum(em, 1e-300))

        return log_obs

    # ------------------------------------------------------------------
    # Forward / Backward  (fully log-domain, matching Mogeng/IOHMM reference)
    # ------------------------------------------------------------------

    def _forward(self, log_obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Single-sequence forward pass (used by predict_proba)."""
        T = log_obs.shape[0]
        if T == 0:
            return np.empty((0, self.n_states)), 0.0
        S     = self.n_states
        log_A = np.log(self.A + 1e-300)
        log_alpha = np.empty((T, S))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_obs[0]
        for t in range(1, T):
            a = log_A.T + log_alpha[t - 1]   # (S, S)
            m = a.max(axis=-1, keepdims=True)
            log_alpha[t] = (np.log(np.exp(a - m).sum(axis=-1))
                            + m.squeeze(-1) + log_obs[t])
        log_ll = float(logsumexp(log_alpha[-1]))
        return log_alpha, log_ll

    def _backward(self, log_obs: np.ndarray) -> np.ndarray:
        """Single-sequence backward pass (used by predict_proba)."""
        T = log_obs.shape[0]
        if T == 0:
            return np.empty((0, self.n_states))
        S     = self.n_states
        log_A = np.log(self.A + 1e-300)
        log_beta = np.zeros((T, S))
        for t in range(T - 2, -1, -1):
            rhs = log_beta[t + 1] + log_obs[t + 1]
            a   = log_A + rhs[np.newaxis, :]    # (S, S)
            m   = a.max(axis=-1, keepdims=True)
            log_beta[t] = np.log(np.exp(a - m).sum(axis=-1)) + m.squeeze(-1)
        return log_beta

    def _forward_batch(self, log_obs: np.ndarray,
                       log_A: np.ndarray) -> np.ndarray:
        """
        Batch forward pass – N sequences simultaneously.

        Matches Mogeng/forward_backward.py's recurrence applied to a whole
        batch: reduces O(N × T) Python calls to O(T_max) numpy calls.

        log_obs : (N, T, S) — padded positions carry log_obs = 0 (no info)
        log_A   : (S, S)

        Returns log_alpha : (N, T, S)
        """
        N, T, S  = log_obs.shape
        log_A_T  = log_A.T                       # log_A_T[s', s] = log A[s→s']
        log_alpha = np.empty((N, T, S))
        log_alpha[:, 0] = np.log(self.pi + 1e-300) + log_obs[:, 0]  # (N, S)
        for t in range(1, T):
            # a[n, s', s] = log_A_T[s', s] + log_alpha[n, t-1, s]
            a = log_A_T[np.newaxis, :, :] + log_alpha[:, t - 1, np.newaxis, :]  # (N, S, S)
            m = a.max(axis=-1, keepdims=True)
            log_alpha[:, t] = (np.log(np.exp(a - m).sum(axis=-1))
                               + m.squeeze(-1) + log_obs[:, t])
        return log_alpha

    def _backward_batch(self, log_obs: np.ndarray,
                        log_A: np.ndarray) -> np.ndarray:
        """
        Batch backward pass – N sequences simultaneously.

        No masking needed: A is stochastic → logsumexp(log_A[s,:]) = log(1) = 0,
        so padded positions (log_obs = 0) naturally give log_beta = 0, which is
        the correct HMM boundary condition.

        log_obs : (N, T, S) — padded positions carry log_obs = 0
        log_A   : (S, S)

        Returns log_beta : (N, T, S)
        """
        N, T, S  = log_obs.shape
        log_beta = np.zeros((N, T, S))
        for t in range(T - 2, -1, -1):
            # a[n, s, s'] = log_A[s, s'] + log_beta[n,t+1,s'] + log_obs[n,t+1,s']
            rhs  = log_beta[:, t + 1] + log_obs[:, t + 1]      # (N, S)
            a    = log_A[np.newaxis] + rhs[:, np.newaxis, :]   # (N, S, S)
            m    = a.max(axis=-1, keepdims=True)
            log_beta[:, t] = (np.log(np.exp(a - m).sum(axis=-1))
                              + m.squeeze(-1))
        return log_beta

    # ------------------------------------------------------------------
    # EM
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_batches(lengths: np.ndarray, S: int, budget_bytes: int,
                      bytes_per: int, max_batch: int):
        """
        Length-bucketed batch boundaries over ASCENDING-sorted ``lengths``.

        Returns a list of ``(start, end)`` index pairs such that, for each
        batch, the peak A-step tensor ``(Nb, T_b-1, S, S)`` — where
        ``T_b = lengths[end-1]`` because lengths are sorted — stays within
        ``budget_bytes``.  Short-trajectory batches therefore get a large
        ``Nb`` (with a small ``T_b``), while the rare long-trajectory batches
        get a small ``Nb``.  This keeps both the memory footprint bounded and
        the O(T) forward/backward Python loop count proportional to the actual
        sequence lengths instead of N × global_T_max.
        """
        N = len(lengths)
        if N == 0:
            return []
        # ~4 simultaneous (Nb, T, S, S) buffers (tensor, its exp(), temporaries).
        cells = max(1, int(budget_bytes) // (S * S * bytes_per * 4))
        bounds = []
        start = 0
        while start < N:
            end = start + 1                       # always take ≥1 sequence
            while end < N:
                nb  = end - start + 1
                T_b = int(lengths[end])           # T_b if seq `end` is added
                if nb > max_batch or nb * max(T_b, 1) > cells:
                    break
                end += 1
            bounds.append((start, end))
            start = end
        return bounds

    def _e_step(self, sequences: List[List[int]]):
        """
        Batch E-step: all N sequences processed simultaneously.

        Complexity reduction
        --------------------
        Before: O(N × T) Python calls to _forward/_backward (one per sequence).
        After:  O(T_max) Python calls, each an O(N × S²) numpy operation.

        The backward pass needs no sequence-length masking because A is
        stochastic: logsumexp(log_A[s,:]) = log(1) = 0, so padded positions
        (log_obs = 0) naturally propagate log_beta = 0 — the correct boundary.

        Position arrays (_pos_n, _pos_t, _pos_curr, _pos_nxt) are precomputed
        once per fit() call so log_obs_all and B accumulation need zero Python
        loops over sequences.
        """
        S     = self.n_states
        V     = self.vocab_size
        N     = len(sequences)
        log_A = np.log(self.A + 1e-300)   # (S, S)

        # ---- Emission table (rebuilt each EM iter as B changes) -----------
        shifted = self.B - self.B.max(axis=1, keepdims=True)
        _base   = np.exp(shifted)          # (S, V)

        _em_all:   Optional[np.ndarray]  = None
        _em_cache: Dict[int, np.ndarray] = {}
        _em_shared: Optional[np.ndarray] = None   # (S, V) node-independent table
        _log_obs_flat_precomp: Optional[np.ndarray] = None  # (n_pos, S) MC1-aware

        # Precompute position index arrays once (reused below).
        _has_pos_early = hasattr(self, '_pos_n') and len(self._pos_n) > 0
        _n_pos_early   = len(self._pos_n) if _has_pos_early else 0

        # Use float32 (4 bytes/elem) so the check covers S=10 with V~2000.
        # float32 precision is sufficient for probabilities 0-1.
        # E-step emission: always position-independent when no hard adjacency.
        # The MC1-aware branch (P ∝ exp(B)×MC1) was removed because the M-step
        # optimises a position-independent marginal B[s,nxt] = log P(nxt|s),
        # making E-step and M-step inconsistent.  With inconsistent EM the B
        # matrix converges to encode MC1-minority choices as high-probability
        # states, inverting the ranking (Acc@3≈1 but Acc@1<MC1).
        # Using position-independent emission in both steps lets EM converge
        # correctly; MC1 topology is gated in predict_proba at inference time.
        if False:  # MC1-aware E-step disabled — kept for reference only
            pass
        elif self.adjacency is None:
            # No topology mask AND no MC1 table → position-independent emission.
            # Build ONE shared (S, V) table instead of V copies (was O(V·S·V)
            # memory ≈ many GB at V=63k; this is O(S·V)).
            em = _base.copy()
            rs = em.sum(axis=1, keepdims=True)
            zr = rs.ravel() == 0
            if zr.any():
                em[zr] = 1.0; rs[zr, 0] = V
            em /= rs
            _em_shared = em
        elif V * S * V * 4 <= _EM_ALL_MEMORY_LIMIT:
            em = np.broadcast_to(_base[np.newaxis], (V, S, V)).copy()
            if self.adjacency is not None:
                adj   = np.asarray(self.adjacency, dtype=float)
                n_r   = min(adj.shape[0], V)
                n_c   = min(adj.shape[1] if adj.ndim > 1 else n_r, V)
                mask_2d = np.zeros((V, V))
                if adj.ndim > 1:
                    mask_2d[:n_r, :n_c] = adj[:n_r, :n_c]
                else:
                    mask_2d[:n_r, :n_r] = np.diag(adj[:n_r])
                em *= mask_2d[:, np.newaxis, :]
            row_sums = em.sum(axis=2, keepdims=True)
            zero     = row_sums == 0
            em = np.where(zero, 1.0 / V, em / np.where(zero, 1.0, row_sums))
            _em_all = em.astype(np.float32)   # float32 halves memory footprint
        else:
            unique_curr = {
                seq[t]
                for seq in sequences
                for t in range(len(seq) - 1)
                if 0 <= seq[t] < V
            }
            for v in unique_curr:
                em = _base.copy()
                if self.adjacency is not None and v < len(self.adjacency):
                    mask = self._align_mask(self.adjacency[v].astype(float), V)
                    em  *= mask[np.newaxis, :]
                rs = em.sum(axis=1, keepdims=True)
                zr = rs.ravel() == 0
                if zr.any():
                    em[zr] = 1.0; rs[zr, 0] = V
                em /= rs
                _em_cache[v] = em

        # ---- Compute lengths and T_max ------------------------------------
        lengths = np.array([max(len(seq) - 1, 0) for seq in sequences],
                           dtype=np.intp)
        if not (lengths > 0).any():
            return np.zeros(S), np.zeros((S, S)), np.zeros((S, V)), np.zeros(S), 0.0, 0

        T_max = int(lengths.max())
        has_cache = hasattr(self, '_pos_n') and len(self._pos_n) > 0
        n_pos = len(self._pos_n) if has_cache else 0

        # ---- Precompute log_obs for ALL valid positions (once per EM iter) ----
        # Shape: (n_pos, S).  Per-batch slicing via _batch_sel_idx is O(1).
        # This replaces all Python loops over sequences inside the batch loop.
        if n_pos > 0:
            if _log_obs_flat_precomp is not None:
                # MC1-aware emission: already fully computed above.
                log_obs_flat = _log_obs_flat_precomp              # (n_pos, S)
            elif _em_shared is not None:
                # Node-independent emission: a single gather on the destination.
                log_obs_flat = np.log(np.maximum(
                    _em_shared[:, self._pos_nxt], 1e-300)).T   # (n_pos, S)
            elif _em_all is not None:
                # Single fancy-index into the (V,S,V) table — fully vectorised.
                log_obs_flat = np.log(np.maximum(
                    _em_all[self._pos_curr, :, self._pos_nxt].astype(np.float64),
                    1e-300))  # (n_pos, S)
            else:
                # _em_cache: iterate over unique SOURCE nodes (not over sequences).
                log_obs_flat = np.full((n_pos, S), 0.0)
                for v, em in _em_cache.items():
                    mask = self._pos_curr == v
                    if mask.any():
                        log_obs_flat[mask] = np.log(
                            np.maximum(em[:, self._pos_nxt[mask]], 1e-300)).T
        else:
            log_obs_flat = np.empty((0, S))

        # ---- Dispatch to GPU when available --------------------------------
        if _TORCH_CUDA and n_pos > 0:
            try:
                return self._e_step_inner_gpu(
                    log_obs_flat, lengths, log_A, S, V, n_pos)
            except RuntimeError as exc:
                # Out-of-memory (or any CUDA runtime error): free the cache and
                # fall back to the CPU mini-batch path below, which has the same
                # numerics but a much smaller per-batch footprint.
                if "out of memory" in str(exc).lower():
                    try:
                        _torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    raise

        # ---- Mini-batch E-step: accumulate statistics across chunks -------
        # Keeps peak memory at O(batch × T_max × S) instead of O(N × T_max × S)
        pi_acc  = np.zeros(S)
        A_acc   = np.zeros((S, S))
        B_num   = np.zeros((S, V))
        B_den   = np.zeros(S)
        total_ll  = 0.0
        total_obs = 0

        # Length-bucketed batch plan (sequences are length-sorted in fit()).
        # Each batch is sized to ITS OWN T_b, so short trajectories run in a
        # single large batch with a tiny T while the rare long paths fall into
        # small batches — instead of every batch padding to the global T_max.
        # ~1.5 GiB peak budget for the (Nb, T_b-1, S, S) float64 A-step tensor.
        pn_arr   = self._pos_n
        pt_arr   = self._pos_t
        pnx_arr  = self._pos_nxt
        bounds   = self._plan_batches(lengths, S, int(1.5 * 1024**3), 8,
                                      _E_STEP_BATCH_SIZE)

        for (batch_start, batch_end) in bounds:
            b_lengths  = lengths[batch_start:batch_end]
            Nb         = batch_end - batch_start
            T_b        = int(b_lengths.max())

            if not (b_lengths > 0).any():
                continue

            # Positions belonging to this batch are a contiguous slice of the
            # (length-sorted → pos_n non-decreasing) position arrays.
            p0 = int(np.searchsorted(pn_arr, batch_start, side="left"))
            p1 = int(np.searchsorted(pn_arr, batch_end,   side="left"))
            has_pos = p1 > p0

            # Build log_obs tensor — 0.0 for padded positions (no information).
            log_obs_b = np.zeros((Nb, T_b, S))
            if has_pos:
                bn = pn_arr[p0:p1] - batch_start
                bt = pt_arr[p0:p1]
                log_obs_b[bn, bt] = log_obs_flat[p0:p1]

            # Forward / backward for this batch
            log_alpha_b = self._forward_batch(log_obs_b, log_A)   # (Nb, T_b, S)
            log_beta_b  = self._backward_batch(log_obs_b, log_A)  # (Nb, T_b, S)

            # log-likelihood
            last_alpha  = log_alpha_b[np.arange(Nb), b_lengths - 1]  # (Nb, S)
            m_ll        = last_alpha.max(axis=-1, keepdims=True)
            log_ll_b    = (np.log(np.exp(last_alpha - m_ll).sum(axis=-1))
                           + m_ll.squeeze(-1))                         # (Nb,)
            total_ll  += float(log_ll_b.sum())
            total_obs += int(b_lengths.sum())

            # Posteriors
            log_ll_bc   = log_ll_b[:, np.newaxis, np.newaxis]
            log_gamma_b = log_alpha_b + log_beta_b - log_ll_bc
            gamma_b     = np.exp(log_gamma_b)                         # (Nb, T_b, S)

            pi_acc += gamma_b[:, 0, :].sum(axis=0)

            # A_acc: vectorised over the time axis — one tensor op per batch.
            # Shape: (Nb, T_b-1, S, S) → sum over (seq, time) axes → (S, S)
            if T_b > 1:
                rhs  = log_obs_b[:, 1:] + log_beta_b[:, 1:]   # (Nb, T_b-1, S)
                leps = (log_A[np.newaxis, np.newaxis]           # (1, 1, S, S)
                        + log_alpha_b[:, :-1, :, np.newaxis]   # (Nb, T_b-1, S, 1)
                        + rhs[:, :, np.newaxis, :]              # (Nb, T_b-1, 1, S)
                        - log_ll_b[:, np.newaxis, np.newaxis, np.newaxis])
                # Mask out padding for variable-length sequences
                if (b_lengths < T_b).any():
                    act = (np.arange(T_b - 1)[np.newaxis, :]
                           < (b_lengths - 1)[:, np.newaxis])   # (Nb, T_b-1)
                    leps[~act] = -np.inf
                A_acc += np.exp(leps).sum(axis=(0, 1))

            # B accumulation — reuse the contiguous position slice
            if has_pos:
                bn  = pn_arr[p0:p1] - batch_start
                bt  = pt_arr[p0:p1]
                bnx = pnx_arr[p0:p1]
                vg  = gamma_b[bn, bt]    # (n_sel, S)
                for s in range(S):
                    B_num[s] += np.bincount(bnx, weights=vg[:, s], minlength=V)
                B_den += vg.sum(axis=0)

        return pi_acc, A_acc, B_num, B_den, total_ll, total_obs

    # ------------------------------------------------------------------
    # GPU-accelerated E-step inner loop
    # ------------------------------------------------------------------

    def _e_step_inner_gpu(self, log_obs_flat: np.ndarray,
                          lengths: np.ndarray,
                          log_A: np.ndarray,
                          S: int, V: int, n_pos: int):
        """
        GPU version of the mini-batch E-step accumulation.

        All per-batch tensors live on CUDA; only the small final accumulators
        (shape S×S, S×V) are transferred back to CPU at the end.

        Uses float32 throughout — sufficient precision for probabilities in
        [0, 1] and short sequences (T=47).
        """
        _make_compiled_fb()          # build compiled kernels on first call
        fw = _compiled_forward
        bw = _compiled_backward

        dev  = _torch.device('cuda')
        N    = len(lengths)
        lg   = _torch.as_tensor(lengths,    dtype=_torch.int64,   device=dev)
        lof  = _torch.as_tensor(log_obs_flat, dtype=_torch.float32, device=dev)
        pn   = _torch.as_tensor(self._pos_n,  dtype=_torch.int64,   device=dev)
        pt   = _torch.as_tensor(self._pos_t,  dtype=_torch.int64,   device=dev)
        pnx  = _torch.as_tensor(self._pos_nxt,dtype=_torch.int64,   device=dev)
        lA   = _torch.as_tensor(log_A,  dtype=_torch.float32, device=dev)  # (S, S)
        lA_T = lA.T.contiguous()                                             # (S, S)
        lpi  = _torch.as_tensor(np.log(self.pi + 1e-300),
                                dtype=_torch.float32, device=dev)            # (S,)

        pi_acc = _torch.zeros(S,    device=dev)
        A_acc  = _torch.zeros(S, S, device=dev)
        B_num  = _torch.zeros(S, V, device=dev)
        B_den  = _torch.zeros(S,    device=dev)
        total_ll  = 0.0
        total_obs = 0

        # Length-bucketed batch plan (sequences are length-sorted in fit()).
        # The peak GPU tensor is ``leps`` (Nb, T_b-1, S, S) float32 plus its
        # exp() copy.  Sizing each batch to its own T_b keeps the footprint
        # bounded AND makes the O(T) compiled forward/backward loop run only as
        # long as the sequences in that batch (not the global T_max).
        try:
            free_b, _ = _torch.cuda.mem_get_info()
        except Exception:
            free_b = 2 * 1024**3          # assume 2 GiB if query unsupported
        # Use 15% of free VRAM as the peak budget for safety/fragmentation.
        pn_np  = self._pos_n
        bounds = self._plan_batches(lengths, S, int(free_b * 0.15), 4,
                                    _GPU_E_STEP_BATCH_SIZE)

        for (batch_start, batch_end) in bounds:
            bl = lg[batch_start:batch_end]          # (Nb,)
            Nb = batch_end - batch_start
            T_b = int(bl.max().item())

            if not (bl > 0).any():
                continue

            # --- contiguous position slice for this batch (sorted pos_n) ---
            p0 = int(np.searchsorted(pn_np, batch_start, side="left"))
            p1 = int(np.searchsorted(pn_np, batch_end,   side="left"))
            has_pos = p1 > p0

            # --- build log_obs tensor on GPU ---
            lob = _torch.zeros(Nb, T_b, S, device=dev)
            if has_pos:
                bn = pn[p0:p1] - batch_start
                bt = pt[p0:p1]
                lob[bn, bt] = lof[p0:p1]

            # --- forward / backward (compiled kernels) ---
            la = fw(lob, lA_T, lpi)                   # (Nb, T_b, S)
            lb = bw(lob, lA)                           # (Nb, T_b, S)

            # --- log-likelihood ---
            last_a = la[_torch.arange(Nb, device=dev), bl - 1]  # (Nb, S)
            m_ll   = last_a.max(dim=-1, keepdim=True).values
            ll_b   = (_torch.log(_torch.exp(last_a - m_ll).sum(dim=-1))
                      + m_ll.squeeze(-1))                         # (Nb,)
            total_ll  += float(ll_b.sum().item())
            total_obs += int(bl.sum().item())

            # --- posteriors ---
            lg_b  = la + lb - ll_b[:, None, None]
            g_b   = _torch.exp(lg_b)                              # (Nb, T_b, S)

            pi_acc += g_b[:, 0, :].sum(dim=0)

            # --- A_acc: one tensor op per batch ---
            if T_b > 1:
                rhs  = lob[:, 1:] + lb[:, 1:]                    # (Nb, T_b-1, S)
                leps = (lA[None, None]
                        + la[:, :-1, :, None]
                        + rhs[:, :, None, :]
                        - ll_b[:, None, None, None])              # (Nb, T_b-1, S, S)
                if (bl < T_b).any():
                    act = (_torch.arange(T_b - 1, device=dev)[None, :]
                           < (bl - 1)[:, None])                   # (Nb, T_b-1)
                    leps.masked_fill_(~act[:, :, None, None], float('-inf'))
                A_acc += _torch.exp(leps).sum(dim=(0, 1))

            # --- B accumulation (vectorised over S with scatter_add_) ---
            if has_pos:
                bn  = pn[p0:p1] - batch_start
                bt  = pt[p0:p1]
                bnx = pnx[p0:p1]                                  # (n_sel,)
                vg  = g_b[bn, bt]                                 # (n_sel, S)
                # Scatter all S dimensions in one op: B_num[s, bnx[i]] += vg[i,s]
                B_num.scatter_add_(1, bnx[None, :].expand(S, -1), vg.T.contiguous())
                B_den += vg.sum(dim=0)

        return (pi_acc.cpu().numpy(), A_acc.cpu().numpy(),
                B_num.cpu().numpy(),  B_den.cpu().numpy(),
                total_ll, total_obs)

    def _m_step(self, pi_acc, A_acc, B_num, B_den) -> None:
        eps = 1e-10
        pi_sum = pi_acc.sum()
        self.pi = pi_acc / max(pi_sum, eps)

        # Dirichlet-style transition prior prevents brittle state dynamics.
        A_post = A_acc + self.trans_prior
        A_row_sums = A_post.sum(axis=1, keepdims=True)
        self.A = A_post / np.where(A_row_sums > 0, A_row_sums, 1.0)
        row_s = self.A.sum(axis=1, keepdims=True)
        self.A /= np.where(row_s > 0, row_s, 1.0)

        # Shrink emissions toward empirical first-order destination marginal.
        if self._transition is not None:
            marginal = np.asarray(self._transition.mean(axis=0)).ravel()
        else:
            marginal = np.ones(self.vocab_size) / max(self.vocab_size, 1)
        marginal = marginal / max(marginal.sum(), eps)

        B_post = B_num + self.emission_prior * marginal[np.newaxis, :]
        denom = (B_den + self.emission_prior)[:, np.newaxis]
        denom = np.where(denom > 0, denom, 1.0)
        freq  = B_post / denom
        self.B = np.log(freq + eps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, sequences: List[List[int]],
            adjacency: Optional[np.ndarray] = None,
            progress_callback=None) -> "IOHMM":
        self.adjacency  = adjacency  # used only as soft prior, not hard gate
        self.vocab_size = self._resolve_vocab(sequences)
        self._build_mc1_table(sequences)

        if self.auto_state_cap:
            n_obs = sum(max(len(seq) - 1, 0) for seq in sequences)
            # Data-adaptive cap: enough states to model heterogeneity, but
            # avoid over-parameterising short/sparse trajectory corpora.
            cap_by_vocab = max(2, min(8, int(np.sqrt(max(self.vocab_size, 1))) + 1))
            cap_by_data = max(2, min(12, n_obs // 200)) if n_obs > 0 else 2
            self.n_states = min(self.n_states, max(2, min(cap_by_vocab, cap_by_data)))

        # Sort trajectories by length (ascending).  The E-step is an
        # order-invariant sum over sequences, so this leaves every result
        # (π, A, B, log-likelihood) unchanged — but it lets the mini-batch
        # loop bucket similar-length paths together.  Without it a handful of
        # very long routed trajectories (T≈2600) force EVERY batch to pad to
        # the global T_max and run the O(T) forward/backward loop that many
        # times, turning each EM iteration into minutes.
        sequences = sorted(sequences, key=len)

        V = self.vocab_size
        # First-order transition matrix, row-normalised.  Stored SPARSE (CSR):
        # a dense V×V matrix is O(V²) memory (≈32 GB at V=63k) which thrashes
        # swap and crashes.  The sparse form holds only observed transitions
        # (O(nnz)) while exposing the identical row distributions / column mean.
        rows_t, cols_t = [], []
        for seq in sequences:
            arr = np.asarray(seq, dtype=np.intp)
            if arr.size < 2:
                continue
            u, v = arr[:-1], arr[1:]
            ok = (u >= 0) & (u < V) & (v >= 0) & (v < V)
            if ok.any():
                rows_t.append(u[ok])
                cols_t.append(v[ok])
        if rows_t:
            r = np.concatenate(rows_t)
            c = np.concatenate(cols_t)
            counts = csr_matrix((np.ones(r.size, dtype=np.float64), (r, c)),
                                shape=(V, V))
        else:
            counts = csr_matrix((V, V), dtype=np.float64)
        row_sums = np.asarray(counts.sum(axis=1)).ravel()
        inv = np.zeros(V, dtype=np.float64)
        nz = row_sums > 0
        inv[nz] = 1.0 / row_sums[nz]
        self._transition = (diags(inv) @ counts).tocsr()   # row-normalised CSR

        # Second-order (bigram) successor table  P(next | prev, curr).
        # IOHMM's order-1 gate caps Acc@1 at the order-1 Markov ceiling, which
        # on these subsampled GPS sequences is only ~0.5 (a single previous
        # node does not determine the next one).  Conditioning the gate on the
        # last TWO nodes — the model's "input" context — lifts the ceiling to the
        # order-2 Markov level (~0.95, matching MC2), while the latent state
        # still contributes its bounded tilt.  Stored as a dict keyed by
        # (prev, curr) so memory is O(#observed bigrams), not O(V²).
        # Vectorised build: collect all (prev, curr, next) triples with numpy
        # per sequence, then collapse to unique triples with their counts in
        # one C-level pass.  The remaining Python loop runs over the number of
        # *distinct* observed bigram→next entries (≪ total transitions on
        # routed data with heavy repetition), instead of every transition.
        # Result is identical to the previous per-transition dict build.
        _tri_a, _tri_b, _tri_c = [], [], []
        for seq in sequences:
            arr = np.asarray(seq, dtype=np.intp)
            if arr.size < 3:
                continue
            a, b, c = arr[:-2], arr[1:-1], arr[2:]
            ok = ((a >= 0) & (a < V) & (b >= 0) & (b < V) &
                  (c >= 0) & (c < V))
            if ok.any():
                _tri_a.append(a[ok])
                _tri_b.append(b[ok])
                _tri_c.append(c[ok])
        trans2: Dict[tuple, Dict[int, int]] = {}
        if _tri_a:
            triples = np.stack([np.concatenate(_tri_a),
                                np.concatenate(_tri_b),
                                np.concatenate(_tri_c)], axis=1)
            uniq, cnts = np.unique(triples, axis=0, return_counts=True)
            for (a, b, c), cnt in zip(uniq, cnts):
                key = (int(a), int(b))
                row = trans2.get(key)
                if row is None:
                    row = {}
                    trans2[key] = row
                row[int(c)] = int(cnt)
        self._transition2 = trans2

        self._init_params()

        # Precompute flat (n, t, curr, nxt) position arrays for all valid
        # transitions — constant across EM iterations so we do it once here.
        # Built per-sequence with numpy and concatenated once: this avoids the
        # 4×O(transitions) pure-Python list.extend(...tolist()) that dominates
        # setup time on long routed trajectories.  Result is identical.
        pos_n_l, pos_t_l, pos_curr_l, pos_nxt_l = [], [], [], []
        for n, seq in enumerate(sequences):
            arr = np.asarray(seq, dtype=np.intp)
            L = max(arr.size - 1, 0)
            if L == 0:
                continue
            c, nx  = arr[:L], arr[1:L + 1]
            ok     = (c >= 0) & (c < V) & (nx >= 0) & (nx < V)
            t_ok   = np.nonzero(ok)[0]
            if t_ok.size == 0:
                continue
            pos_n_l.append(np.full(t_ok.size, n, dtype=np.intp))
            pos_t_l.append(t_ok.astype(np.intp, copy=False))
            pos_curr_l.append(c[ok])
            pos_nxt_l.append(nx[ok])
        if pos_n_l:
            self._pos_n    = np.concatenate(pos_n_l)
            self._pos_t    = np.concatenate(pos_t_l)
            self._pos_curr = np.concatenate(pos_curr_l)
            self._pos_nxt  = np.concatenate(pos_nxt_l)
        else:
            self._pos_n    = np.array([], dtype=np.intp)
            self._pos_t    = np.array([], dtype=np.intp)
            self._pos_curr = np.array([], dtype=np.intp)
            self._pos_nxt  = np.array([], dtype=np.intp)

        prev_ll_per_obs = -np.inf
        for em_iter in range(self.max_iter):
            pi_acc, A_acc, B_num, B_den, total_ll, n_obs = self._e_step(sequences)
            self._m_step(pi_acc, A_acc, B_num, B_den)
            converged = False
            if n_obs > 0:
                ll_per_obs  = total_ll / n_obs
                improvement = abs(ll_per_obs - prev_ll_per_obs)
                converged   = improvement < self.tol
                prev_ll_per_obs = ll_per_obs
            if progress_callback is not None:
                progress_callback(em_iter + 1, converged)
            if converged:
                break

        return self

    def predict_proba(self, context: List[int],
                      vocab_size: Optional[int] = None) -> np.ndarray:
        V = vocab_size or self.vocab_size
        if V == 0 or self.B is None:
            raise RuntimeError("Model not fitted yet.")

        context = list(context)
        if not context:
            return np.ones(V) / V

        if len(context) >= 2:
            log_obs = self._log_obs_seq(context)
            if log_obs.shape[0] > 0:
                if _MOGENG_AVAILABLE:
                    T_obs = log_obs.shape[0]
                    lpt   = np.broadcast_to(
                        np.log(self.A + 1e-300)[np.newaxis],
                        (max(T_obs - 1, 0), self.n_states, self.n_states))
                    log_alpha = _mogeng_forward(
                        np.log(self.pi + 1e-300), lpt, log_obs)
                    log_ll    = float(logsumexp(log_alpha[-1]))
                else:
                    log_alpha, log_ll = self._forward(log_obs)
                state_dist = np.exp(log_alpha[-1] - log_ll)
            else:
                state_dist = self.pi
        else:
            state_dist = self.pi

        curr = context[-1]
        if curr >= self.vocab_size:
            return self._neighbors_uniform(V, context)

        # Position-independent state preference: P(nxt | state_dist)
        em    = self._emission_matrix(curr)   # (S, V) — softmax(B), no MC1
        pref  = state_dist @ em               # (V,) state node-preference

        # Order-aware gate: P(next | prev, curr) with order-1 backoff.  The gate
        # governs the ranking; the latent state applies only a BOUNDED tilt
        # within ±state_tilt so it can re-order genuine near-ties but never
        # overturn a real gate difference.  Using the order-2 (bigram) gate
        # lifts IOHMM's ceiling from the order-1 Markov level (~0.5 here) to the
        # order-2 level (~0.95), while the position-independent state preference
        # — which alone inverts the ranking (Acc@3≈1 but Acc@1≪MC) — is reduced
        # to a small tie-breaker.
        gate = None
        trans2 = getattr(self, "_transition2", None)
        if trans2 is not None and len(context) >= 2:
            row = trans2.get((int(context[-2]), int(curr)))
            if row:
                tot = float(sum(row.values()))
                if tot > 0.0:
                    gate = np.zeros(len(pref))
                    for v, c in row.items():
                        if 0 <= v < len(pref):
                            gate[v] = c / tot
        if gate is None and self._transition is not None and \
                0 <= curr < self._transition.shape[0]:
            # order-1 backoff for unseen / missing bigram contexts
            mc1_row = np.asarray(
                self._transition[curr].todense()).ravel().astype(np.float64)
            if len(mc1_row) < len(pref):
                tmp = np.zeros(len(pref)); tmp[:len(mc1_row)] = mc1_row
                mc1_row = tmp
            else:
                mc1_row = mc1_row[:len(pref)]
            gate = mc1_row

        if gate is not None:
            _tilt = getattr(self, "state_tilt", 0.05)
            proba = np.zeros_like(gate)
            support = gate > 0.0          # nodes reachable in this context
            if support.any():
                pref_s = pref[support]
                rng = float(pref_s.max() - pref_s.min())
                if rng > 0.0:
                    pref_norm = (pref_s - pref_s.min()) / rng     # -> [0, 1]
                else:
                    pref_norm = np.zeros_like(pref_s)
                # tilt in [1-_tilt, 1+_tilt]; bounded => cannot reorder non-ties.
                tilt = 1.0 + _tilt * (2.0 * pref_norm - 1.0)
                proba[support] = gate[support] * tilt
            else:
                proba = pref
        else:
            proba = pref

        if V > self.vocab_size:
            out = np.zeros(V)
            out[:self.vocab_size] = proba
            proba = out
        else:
            proba = proba[:V]

        s = proba.sum()
        if s > 0:
            proba /= s
        else:
            return self._neighbors_uniform(V, context)

        # Soft topology prior (no-op when adjacency is None).  The empirical
        # gate above already provides the Markov floor, so the order-1
        # `_mc1_interpolate` is intentionally NOT applied here — mixing in 60%
        # order-1 mass would drag the ranking back down to the order-1 ceiling.
        return self._apply_topo_prior(proba, context, self.topo_beta)
