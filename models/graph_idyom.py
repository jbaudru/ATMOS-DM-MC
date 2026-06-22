"""
GraphIDyOM: Variable-Order Markov Chain with Dual LTM/STM Memory.

Implements the IDyOM dual-memory paradigm (Pearce, 2005) for road network
trajectory prediction using pure variable-order Markov chains.

Architecture follows IDyOM exactly:
- Long-Term Model (LTM): Population-level Markov statistics from training corpus
  graph transitions. Static, global knowledge.
- Short-Term Model (STM): Individual-level Markov statistics built dynamically
  from the current trajectory. Adaptive, piece-specific knowledge.
- Variable-order chains (orders 1..K) combined via entropy weighting:
  lower entropy (higher confidence) orders receive higher weight.
- LTM and STM combined via the same entropy-weighting mechanism.

No LSTM, no attention, no MLP. Pure statistical Markov chain inference,
wrapped as a PyTorch nn.Module for pipeline compatibility.
Output is log P(next | sequence), compatible with nn.CrossEntropyLoss.

References:
    Pearce, M.T. (2005). The Construction and Evaluation of Statistical Models
    of Melodic Structure in Music Perception and Composition. PhD thesis,
    City University, London.

Key Classes:
- GraphIDyOMEnhanced: Pure Markov chain dual LTM/STM model
- CustomNodeEmbedding: Graph-structural node embedding (optional utility)

Helper Functions:
- create_degree_features, create_pagerank_features, create_combined_features
- create_model_with_strategy, build_graph_structure_from_sequences
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable
import networkx as nx

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)


class CustomNodeEmbedding(nn.Module):
    """
    Flexible node embedding wrapper that accepts custom feature computation.
    
    Instead of learning embeddings, computes them from graph structure using
    a custom feature function.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 feature_fn: Callable[[int], np.ndarray]):
        """
        Args:
            vocab_size: Number of unique nodes
            embedding_dim: Desired embedding dimension
            feature_fn: Function that takes node_id -> np.ndarray of features
                       Will be projected to embedding_dim
        """
        super(CustomNodeEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.feature_fn = feature_fn
        
        # Cache features to avoid recomputation and CPU transfers
        self._feature_cache = {}  # {node_id: feature_array}
        self._projection = None
        self._feature_dim = None
        self._initialized = False
    
    def _initialize_projection(self, feature_dim: int, device: torch.device) -> None:
        """Initialize projection matrix on first use with correct feature dimension."""
        if not self._initialized:
            self._feature_dim = feature_dim
            self._projection = nn.Linear(self._feature_dim, self.embedding_dim).to(device)
            self._initialized = True
    
    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings for given nodes with caching.
        
        Args:
            node_indices: (batch_size,) node IDs
            
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        device = node_indices.device
        node_ids_cpu = node_indices.cpu().tolist()
        
        # Compute raw features for each node (with caching)
        features = []
        for node_id in node_ids_cpu:
            if node_id not in self._feature_cache:
                self._feature_cache[node_id] = self.feature_fn(int(node_id))
            features.append(self._feature_cache[node_id])
        
        features = np.array(features, dtype=np.float32)  # (batch_size, feature_dim)
        
        # Initialize projection if needed
        self._initialize_projection(features.shape[1], device)
        
        # Convert to tensor and project (faster with direct from_numpy)
        features_tensor = torch.from_numpy(features).to(device)
        embeddings = self._projection(features_tensor)
        
        return embeddings


class GraphIDyOMEnhanced(nn.Module):
    """
    GraphIDyOM: Pure variable-order Markov chain model with dual LTM/STM memory.

    Faithfully implements IDyOM (Pearce, 2005) for directed graph trajectory prediction:
    - No LSTM, no attention, no MLP — pure Markov chain statistical inference
    - LTM: Population-level transition probabilities from the training corpus graph
    - STM: Individual-level occurrence counts built dynamically from current sequence
    - Variable orders 1..K combined via IDyOM entropy weighting:
          w_k = H(P_k)^{-1} / sum_{k'} H(P_{k'})^{-1}
      where H(P) = -sum_v P(v) log P(v). Lower entropy = higher confidence = higher weight.
    - LTM and STM combined by the same entropy-weighting mechanism.

    Output is log P(next | sequence), compatible with nn.CrossEntropyLoss.
    The model has no learnable parameters; "training" means accumulating corpus
    edge counts into graph_structure (see build_graph_structure_from_sequences).

    Usage:
        model = GraphIDyOMEnhanced(vocab_size=vocab_size, max_order=10)
        logits = model(sequences, graph_structure=graph_structure)
    """

    STRATEGIES = {
        'degree': 'Degree-based node features (unused, API compat)',
        'pagerank': 'PageRank-based node features (unused, API compat)',
        'combined': 'Combined structural features (unused, API compat)',
    }

    def __init__(self,
                 vocab_size: int,
                 max_order: int = 10,
                 smoothing: float = 1e-4,
                 # Legacy parameters kept for API compatibility:
                 embedding_dim: int = 128,
                 embedding_strategy: str = 'degree',
                 custom_feature_fn: Optional[Callable] = None,
                 use_neural_weighting: bool = False,
                 dropout: float = 0.0):
        """
        Args:
            vocab_size: Number of unique nodes in the graph
            max_order: Maximum Markov order K (context window length)
            smoothing: Laplace smoothing added to all transition counts
        """
        super(GraphIDyOMEnhanced, self).__init__()

        self.vocab_size = vocab_size
        self.max_order = max_order
        self.smoothing = smoothing

        # Pre-compute exponential recency decay weights for all orders.
        # Weight of position j within order k: w_j = exp(-lambda*(k-j)) / Z
        # Most recent position (j=k) gets the highest weight.
        self._decay_rate = 0.5
        self._precomputed_decay_weights = self._compute_all_decay_weights(max_order)

        self._eps = 1e-8
        # Strength of the count-discounted (evidence) weighting used when
        # combining variable-order Markov distributions.  lambda = 0 recovers
        # the original pure inverse-entropy IDyOM rule; lambda > 0 increases
        # the impact of the evidence-confidence term below.
        # Aligned with the proven entropy-weighted Markov baseline
        # (SimpleMarkov order-10, count_discount=1.0): lambda=1.0 reproduces that
        # well-calibrated rule instead of the over-aggressive 1.5 discount that
        # was starving good high-order evidence.
        self._evidence_lambda = 1.0
        # Evidence confidence uses a saturating form N / (N + tau) so tiny
        # counts are discounted strongly while very large counts do not drown
        # all other orders. tau is the support level at which confidence is 0.5.
        # Aligned with the winning entropy-weighted baseline (tau = 8.0).
        self._evidence_tau = 8.0
        # Strength of the symmetric Dirichlet prior used to turn raw STM match
        # counts into a distribution.  An order/context observed only a handful
        # of times stays close to uniform (high entropy -> small inverse-entropy
        # weight), so spurious high-order matches on a small alphabet (e.g. the
        # 64 graph-embedding clusters) can no longer dominate the mixture; the
        # STM only sharpens and contributes once it has accumulated real support.
        # Tuned: lowered from 2.0 → 1.0 — user mobility is deterministic enough
        # that we want the STM to sharpen faster with fewer matches.
        self._stm_prior_strength = 1.0
        # Jelinek-Mercer back-off strength for the LTM k-gram distributions.
        # A high-order context with support N is interpolated with the reliable
        # order-1 distribution using lambda = N / (N + gamma): rarely-seen
        # contexts (small N) regress toward order-1 and therefore keep an
        # *honest* (high) entropy, so they can no longer hijack the
        # inverse-entropy mixture with a near-deterministic spike built from a
        # single observation.  Well-supported contexts (large N) keep their
        # sharp higher-order signal.  gamma = 0 recovers the raw (overconfident)
        # empirical estimate.
        # Disabled (gamma = 0): the pure-backoff baseline (MC10P) that fully
        # trusts the highest matching order out-performs the model, so diluting
        # sharp high-order contexts toward order-1 was destroying the very signal
        # that helps.  Single-observation overconfidence is instead controlled by
        # the evidence discount above, exactly as in the winning baseline.
        self._ltm_backoff_gamma = 0.0
        # --- Per-user STM personalisation gate ---------------------------------
        # The population LTM is already very confident (low entropy), so a pure
        # inverse-entropy LTM/STM mixture lets it drown the individual STM even
        # when the user's own history carries a strong, directly-relevant signal.
        # We therefore let the STM's weight grow with how much *user-specific*
        # evidence backs it:  w_stm <- (1 / H_stm) * (1 + alpha * conf),
        # with conf = N_stm / (N_stm + tau) where N_stm is the total number of
        # (context -> next) matches found in this user's history.
        #   * No history / no match (N_stm = 0)  -> conf = 0 -> reduces EXACTLY
        #     to the original pure inverse-entropy LTM/STM combination (safe).
        #   * Rich, relevant history (N_stm >> tau) -> conf -> 1 -> the
        #     personalised STM is allowed to override the population average.
        # Per-user STM personalisation gate (RE-ENABLED, order-aware).  The
        # population LTM is almost always lower-entropy than the sparse per-user
        # STM, so a pure inverse-entropy mix structurally drowns the individual
        # signal — the one source of information the population baselines cannot
        # access.  To genuinely exploit the *same user's recent path* we boost
        # the STM weight only when the user's own history backs it with a long,
        # well-supported, decisive context match:
        #     conf = (N/(N+tau)) * (k_max/K)^p * (max_v P_stm(v))^kappa
        #     w_stm <- w_stm * (1 + alpha * conf)
        # Every factor is in [0, 1], so with no / weak / flat user evidence
        # conf -> 0 and this reduces EXACTLY to the faithful inverse-entropy
        # IDyOM mix (users without relevant history are never harmed).  The
        # order term (k_max/K)^p is the safeguard the earlier naive gate lacked:
        # only a genuinely long exact match in the user's history — "this driver
        # has traversed this very sequence before" — is allowed to override the
        # population model, so the boost can no longer be triggered by spurious
        # low-order coincidences that previously collapsed AR accuracy.
        self._stm_personalization_alpha = 5.0
        # Support confidence saturates at tau total context matches.  A per-user
        # exact-context match is rare, so this is deliberately small.
        self._stm_personalization_tau   = 5.0
        # Exponent p on the order-specificity term (k_max / K).  p > 0 rewards
        # long user-history context matches; p = 0 disables the order gate.
        self._stm_order_power = 1.0
        # --- Peakedness gate ---------------------------------------------------
        # Support alone (N_stm) does not tell us whether the STM actually has a
        # *decisive* opinion for this context: a user with many matches but a
        # flat next-cluster distribution would still be up-weighted and only add
        # noise, while a user with a sharp, repeated personal habit may not get
        # enough push to override the population mode.  We therefore multiply the
        # confidence by the STM's own peakedness (its top-1 mass) raised to a
        # power kappa, so the personalised STM is only allowed to override the
        # LTM when the user's history is BOTH well-supported AND confident:
        #     conf <- (N_stm / (N_stm + tau)) * (max_v P_stm(v)) ** kappa
        # Peakedness exponent kappa: scale the STM confidence by its own top-1
        # mass so only sharp, decisive user histories override the population
        # LTM.  kappa = 0 removes the peakedness term.
        self._stm_peak_kappa = 0.5

    # ------------------------------------------------------------------
    # Decay weights
    # ------------------------------------------------------------------

    def _compute_all_decay_weights(self, max_order: int) -> Dict[int, np.ndarray]:
        """Pre-compute exponential recency decay weights for every Markov order."""
        weights = {}
        for order in range(1, max_order + 1):
            w = np.exp(-self._decay_rate * np.arange(order, dtype=np.float32))
            w = w / w.sum()
            w = w[::-1].copy()  # index 0 = oldest, index -1 = most recent
            weights[order] = w
        return weights

    # ------------------------------------------------------------------
    # IDyOM entropy-weighting
    # ------------------------------------------------------------------

    def _entropy(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Shannon entropy of a probability distribution.

        Args:
            dist: (batch, vocab) — rows should sum to 1
        Returns:
            (batch,) entropy in nats
        """
        safe = dist.clamp(min=self._eps)
        return -(safe * safe.log()).sum(dim=1)

    def _entropy_weighted_combination(self,
                                      distributions: List[torch.Tensor]) -> torch.Tensor:
        """
        IDyOM entropy-based weighted combination (Pearce, 2005).

        w_m = H(P_m)^{-1} / sum_{m'} H(P_{m'})^{-1}
        P_out = sum_m w_m * P_m

        Lower entropy (more confident) distribution receives higher weight.

        Args:
            distributions: List of M tensors, each (batch, vocab)
        Returns:
            (batch, vocab) weighted and renormalized combined distribution
        """
        stacked = torch.stack(distributions, dim=0)   # (M, batch, vocab)
        M = stacked.shape[0]

        # Entropy per distribution: (M, batch)
        entropies = torch.stack([self._entropy(stacked[m]) for m in range(M)], dim=0)

        # Inverse-entropy weights: (M, batch), normalized across M
        inv_ent = 1.0 / (entropies + self._eps)
        weights = inv_ent / inv_ent.sum(dim=0, keepdim=True)  # (M, batch)

        # Weighted sum: (batch, vocab)
        combined = (stacked * weights.unsqueeze(2)).sum(dim=0)
        combined = combined / (combined.sum(dim=1, keepdim=True) + self._eps)
        return combined

    def _stm_support_stats(self, stm_ev: List[torch.Tensor],
                           device: torch.device):
        """Aggregate per-order STM match counts.

        Returns ``(support, max_order)`` where ``support`` (batch,) is the total
        number of user-history context matches across all orders and
        ``max_order`` (batch,) is the highest Markov order at which the user's
        history matched the current context (0 if none).  Both are ``None`` when
        no STM evidence is available.
        """
        if not stm_ev:
            return None, None
        stacked = torch.stack(stm_ev, dim=0)                  # (K, batch) counts
        support = stacked.sum(dim=0)                          # (batch,)
        K = stacked.shape[0]
        orders = torch.arange(1, K + 1, device=device,
                              dtype=support.dtype).unsqueeze(1)   # (K, 1)
        matched = (stacked > 0).to(support.dtype)             # (K, batch)
        max_order = (orders * matched).max(dim=0).values      # (batch,)
        return support, max_order

    def _combine_ltm_stm(self, p_ltm: torch.Tensor, p_stm: torch.Tensor,
                         stm_support: Optional[torch.Tensor] = None,
                         stm_max_order: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evidence-gated inverse-entropy combination of the LTM and STM.

        Pure inverse-entropy weighting (Pearce 2005) lets the very confident
        population LTM drown the individual STM.  To exploit the *same user's
        recent path*, the STM weight is boosted in proportion to how strong,
        specific and decisive the user-history evidence is:

            w_ltm = (H(P_ltm) + eps)^{-1}
            conf  = (N/(N+tau)) * (k_max/K)^p * (max_v P_stm(v))^kappa   in [0,1)
            w_stm = (H(P_stm) + eps)^{-1} * (1 + alpha * conf)
            P     = (w_ltm * P_ltm + w_stm * P_stm) / (w_ltm + w_stm)

        With ``alpha = 0``, ``stm_support = 0`` or no high-order match this
        reduces *exactly* to the original pure inverse-entropy LTM/STM
        combination, so a user with no relevant history is never harmed; a user
        whose recent path contains a long, decisive match for the current
        context lets the personalised STM override the population average.

        Args:
            p_ltm:        (batch, vocab) combined LTM distribution.
            p_stm:        (batch, vocab) combined STM distribution.
            stm_support:  optional (batch,) total user-history match count N.
            stm_max_order: optional (batch,) highest Markov order at which the
                user's history matched the current context (0 = no match).
        Returns:
            (batch, vocab) personalised combined distribution.
        """
        h_ltm = self._entropy(p_ltm)                       # (batch,)
        h_stm = self._entropy(p_stm)                       # (batch,)
        w_ltm = 1.0 / (h_ltm + self._eps)
        w_stm = 1.0 / (h_stm + self._eps)

        if stm_support is not None and self._stm_personalization_alpha > 0.0:
            tau  = self._stm_personalization_tau
            conf = stm_support / (stm_support + tau)        # (batch,) in [0, 1)
            # Order specificity: only a long exact match in the user's own
            # history (high k_max) is strong enough to override the population.
            if stm_max_order is not None and self._stm_order_power > 0.0:
                spec = (stm_max_order.to(conf.dtype)
                        / float(self.max_order)).clamp(0.0, 1.0)
                conf = conf * spec.pow(self._stm_order_power)
            # Peakedness: only decisive (sharp) STM opinions are trusted.
            if self._stm_peak_kappa > 0.0:
                peak = p_stm.max(dim=1).values              # (batch,)
                conf = conf * peak.pow(self._stm_peak_kappa)
            w_stm = w_stm * (1.0 + self._stm_personalization_alpha * conf)

        denom = (w_ltm + w_stm).unsqueeze(1) + self._eps    # (batch, 1)
        combined = (w_ltm.unsqueeze(1) * p_ltm + w_stm.unsqueeze(1) * p_stm) / denom
        combined = combined / (combined.sum(dim=1, keepdim=True) + self._eps)
        return combined

    def _evidence_weighted_combination(self,
                                       distributions: List[torch.Tensor],
                                       evidence: Optional[List[torch.Tensor]] = None
                                       ) -> torch.Tensor:
        """
        Count-discounted entropy-weighted combination of variable-order models.

        Extends the IDyOM inverse-entropy rule with an evidence discount so a
        high-order context seen only a handful of times (hence near-zero
        entropy) can no longer dominate the mixture:

            conf_k = (N_k / (N_k + tau))^{lambda}
            w_k    = conf_k * (H(P_k) + eps)^{-1}
                     / sum_{k'} conf_{k'} * (H(P_{k'}) + eps)^{-1}

        where ``N_k`` is the amount of training/history support for order ``k``,
        ``lambda = self._evidence_lambda``, and ``tau = self._evidence_tau``.
        Orders with ``N_k = 0`` are excluded. If a row has no evidence at any
        order (all ``N_k = 0``) the method falls back to the pure
        inverse-entropy weighting for that row,
        and if ``evidence is None`` it reduces exactly to
        :meth:`_entropy_weighted_combination`.

        Args:
            distributions: List of M tensors, each (batch, vocab).
            evidence:      Optional list of M tensors, each (batch,), giving the
                           per-order context support count for every batch row.
        Returns:
            (batch, vocab) weighted and renormalized combined distribution.
        """
        stacked = torch.stack(distributions, dim=0)   # (M, batch, vocab)
        M = stacked.shape[0]

        entropies = torch.stack([self._entropy(stacked[m]) for m in range(M)], dim=0)
        inv_ent = 1.0 / (entropies + self._eps)        # (M, batch)
        base_weights = inv_ent / inv_ent.sum(dim=0, keepdim=True)

        if evidence is None:
            weights = base_weights
        else:
            ev = torch.stack(evidence, dim=0).to(inv_ent.dtype).clamp(min=0.0)  # (M, batch)
            conf = ev / (ev + self._evidence_tau)
            disc = conf.pow(self._evidence_lambda)
            raw = disc * inv_ent                       # (M, batch)
            col = raw.sum(dim=0, keepdim=True)         # (1, batch)
            has_ev = col > self._eps
            disc_weights = raw / (col + self._eps)
            weights = torch.where(has_ev, disc_weights, base_weights)

        combined = (stacked * weights.unsqueeze(2)).sum(dim=0)
        combined = combined / (combined.sum(dim=1, keepdim=True) + self._eps)
        return combined

    # ------------------------------------------------------------------
    # LTM: population-level Markov chains from training corpus graph
    # ------------------------------------------------------------------

    def _compute_markov_from_graph(self, sequences: torch.Tensor,
                                   graph_structure: Dict) -> List[torch.Tensor]:
        """
        LTM: Variable-order Markov distributions from the training corpus graph.

        For each order k (1..K):
        - Take the last k nodes of the sequence
        - Look up their rows in the corpus transition matrix
        - Combine rows with exponential recency decay weights
        This approximates P_LTM(next | last-k context) using the population graph.

        Args:
            sequences: (batch, seq_len)
            graph_structure: dict with 'edge_weights' (vocab, vocab)
        Returns:
            Tuple ``(distributions, evidence)`` where ``distributions`` is a
            list of K probability distributions, each (batch, vocab), and
            ``evidence`` is a list of K (batch,) tensors giving the per-order
            training support count used for count-discounted weighting.
        """
        if graph_structure is None:
            return self._compute_markov_from_sequences(sequences)

        edge_weights  = graph_structure.get('edge_weights')
        first_order_sparse = graph_structure.get('first_order_sparse')
        kgram_tables  = graph_structure.get('kgram_tables', {})
        batch_size, seq_len = sequences.shape
        device = sequences.device
        # Laplace-smoothed, row-normalised 1st-order transition matrix (dense path).
        transition = None
        if isinstance(edge_weights, torch.Tensor):
            transition = edge_weights.to(device) + self.smoothing
            transition = transition / transition.sum(dim=1, keepdim=True)

        seqs_np = sequences.cpu().numpy().astype(np.int32)
        _transition_np = None   # lazy CPU copy, only needed for k-gram back-off
        _fallback_np = None
        _first_order_row_cache: Dict[int, np.ndarray] = {}

        def _fallback_from_sparse(last_nodes: np.ndarray) -> np.ndarray:
            if not first_order_sparse:
                return np.full((len(last_nodes), self.vocab_size),
                               1.0 / max(1, self.vocab_size), dtype=np.float32)

            row_ptr  = first_order_sparse['_row_ptr']
            next_idx = first_order_sparse['_next_idx']
            next_cnt = first_order_sparse['_next_cnt']

            out = np.empty((len(last_nodes), self.vocab_size), dtype=np.float32)
            for i, node in enumerate(last_nodes.astype(np.int64)):
                n = int(node)
                cached = _first_order_row_cache.get(n)
                if cached is None:
                    s = int(row_ptr[n])
                    e = int(row_ptr[n + 1])
                    idxs = next_idx[s:e]
                    cnts = next_cnt[s:e]

                    denom = float(cnts.sum()) + self.smoothing * self.vocab_size
                    if denom <= 0.0:
                        dense_row = np.full(self.vocab_size,
                                            1.0 / max(1, self.vocab_size),
                                            dtype=np.float32)
                    else:
                        base = self.smoothing / denom
                        dense_row = np.full(self.vocab_size, base, dtype=np.float32)
                        if len(idxs) > 0:
                            dense_row[idxs] += cnts / denom
                    _first_order_row_cache[n] = dense_row
                    cached = dense_row
                out[i] = cached
            return out

        def _get_fallback_np() -> np.ndarray:
            nonlocal _transition_np, _fallback_np
            if _fallback_np is not None:
                return _fallback_np
            if transition is not None:
                if _transition_np is None:
                    _transition_np = transition.cpu().numpy()
                _fallback_np = _transition_np[seqs_np[:, -1]]
            else:
                _fallback_np = _fallback_from_sparse(seqs_np[:, -1])
            return _fallback_np

        # Per-node total out-transition counts, used as the order-1 evidence
        # (training support) for count-discounted weighting.  Prefer an
        # explicit 'out_counts' vector (saved at train time); otherwise derive
        # it from the sparse first-order counts when available.
        out_counts_vec = None
        oc = graph_structure.get('out_counts')
        if oc is not None:
            out_counts_vec = (oc.detach().cpu().numpy() if isinstance(oc, torch.Tensor)
                              else np.asarray(oc)).astype(np.float64)
        elif first_order_sparse:
            _rp = first_order_sparse['_row_ptr']
            _nc = first_order_sparse['_next_cnt'].astype(np.float64)
            _csum = np.concatenate([[0.0], np.cumsum(_nc)])
            out_counts_vec = (_csum[_rp[1:]] - _csum[_rp[:-1]]).astype(np.float64)

        distributions = []
        evidence = []
        for order in range(1, self.max_order + 1):
            ev_np = np.zeros(batch_size, dtype=np.float32)
            if order == 1:
                # Fast: single GPU matrix row-lookup
                if transition is not None:
                    dist = transition[sequences[:, -1]]                # (batch, vocab)
                else:
                    dist = torch.from_numpy(_get_fallback_np().astype(np.float32)).to(device)
                if out_counts_vec is not None:
                    ev_np = out_counts_vec[seqs_np[:, -1]].astype(np.float32)
                else:
                    ev_np = np.ones(batch_size, dtype=np.float32)

            elif order <= seq_len and order in kgram_tables:
                # Vectorised k-gram lookup via pre-sorted ID arrays; back-off to order 1
                table = kgram_tables[order]

                if '_sorted_ids' in table:
                    # Fast vectorised path using searchsorted
                    sorted_ids  = table['_sorted_ids']   # (n_unique,)
                    if len(sorted_ids) == 0:
                        dist = torch.from_numpy(_get_fallback_np().astype(np.float32)).to(device)
                        distributions.append(dist)
                        evidence.append(torch.from_numpy(ev_np).to(device))
                        continue
                    # Encode batch contexts as int64
                    ctx_win = seqs_np[:, -order:]  # (B, order)
                    ctx_ids = np.zeros(batch_size, dtype=np.int64)
                    for j in range(order):
                        ctx_ids = ctx_ids * np.int64(self.vocab_size) + ctx_win[:, j].astype(np.int64)
                    # searchsorted → row index candidate
                    ri = np.searchsorted(sorted_ids, ctx_ids)
                    ri = np.clip(ri, 0, len(sorted_ids) - 1)
                    found = sorted_ids[ri] == ctx_ids  # (B,) bool
                    # Build dist: found → dist_matrix row, not-found → order-1 back-off
                    fallback = _get_fallback_np()  # (B, V)
                    if '_dist_matrix' in table:
                        dist_matrix = table['_dist_matrix']  # (n_unique, V)
                        dist_np = np.where(found[:, np.newaxis], dist_matrix[ri], fallback)
                        # Legacy dense format does not retain raw counts; mark
                        # matched rows as non-zero evidence so they are kept.
                        ev_np = found.astype(np.float32)
                    elif ('_row_ptr' in table and '_next_idx' in table and
                          '_next_cnt' in table):
                        # Sparse k-gram rows: materialize only rows touched by this batch.
                        row_ptr  = table['_row_ptr']
                        next_idx = table['_next_idx']
                        next_cnt = table['_next_cnt']
                        gamma = self._ltm_backoff_gamma

                        dist_np = fallback.copy()
                        found_rows = np.where(found)[0]
                        # Cache the (empirical distribution, lambda, support) per
                        # k-gram row; the final per-sample distribution still
                        # depends on that sample's order-1 back-off row.
                        row_cache: Dict[int, tuple] = {}
                        for bi in found_rows:
                            row_i = int(ri[bi])
                            cached = row_cache.get(row_i)
                            if cached is None:
                                s = int(row_ptr[row_i])
                                e = int(row_ptr[row_i + 1])
                                idxs = next_idx[s:e]
                                cnts = next_cnt[s:e]

                                total = float(cnts.sum())
                                emp = np.zeros(self.vocab_size, dtype=np.float32)
                                if len(idxs) > 0 and total > 0.0:
                                    emp[idxs] = (cnts / total).astype(np.float32)
                                # Jelinek-Mercer weight: trust the high order in
                                # proportion to how much evidence backs it.
                                lam = total / (total + gamma) if total > 0.0 else 0.0
                                row_cache[row_i] = (emp, np.float32(lam), total)
                                cached = row_cache[row_i]
                            emp, lam, total = cached
                            # Interpolate toward this sample's order-1 row so a
                            # low-support context keeps an honest (high) entropy.
                            dist_np[bi] = lam * emp + (1.0 - lam) * fallback[bi]
                            # Evidence = total observations of this k-gram context.
                            ev_np[bi] = total
                    else:
                        dist_np = fallback
                    dist = torch.from_numpy(dist_np.astype(np.float32)).to(device)

                else:
                    # Fallback Python loop (slow, only if lookup arrays not built)
                    dist_np = np.empty((batch_size, self.vocab_size), dtype=np.float32)
                    for b in range(batch_size):
                        ctx = tuple(int(seqs_np[b, -order + j]) for j in range(order))
                        if ctx in table:
                            if isinstance(table[ctx], dict):
                                nxt_to_cnt = table[ctx]
                                total = float(sum(nxt_to_cnt.values()))
                                denom = total + self.smoothing * self.vocab_size
                                base = self.smoothing / denom
                                row = np.full(self.vocab_size, base, dtype=np.float32)
                                for nxt, cnt in nxt_to_cnt.items():
                                    row[int(nxt)] += float(cnt) / denom
                                dist_np[b] = row
                                ev_np[b] = total
                            else:
                                counts     = table[ctx] + self.smoothing
                                dist_np[b] = counts / counts.sum()
                                ev_np[b]   = float(table[ctx].sum())
                        else:
                            dist_np[b] = _get_fallback_np()[b]
                    dist = torch.from_numpy(dist_np).to(device)

            else:
                # Order exceeds available k-gram data: fall back to order 1.
                # No genuine support at this order → zero evidence (excluded).
                if transition is not None:
                    dist = transition[sequences[:, -1]]
                else:
                    dist = torch.from_numpy(_get_fallback_np().astype(np.float32)).to(device)

            distributions.append(dist)
            evidence.append(torch.from_numpy(ev_np).to(device))

        return distributions, evidence

    # ------------------------------------------------------------------
    # STM: individual-level Markov chains from recent user history
    # ------------------------------------------------------------------

    def _compute_markov_from_sequences(self, sequences: torch.Tensor,
                                       user_history: Optional[torch.Tensor] = None,
                                       user_history_lengths: Optional[torch.Tensor] = None
                                       ) -> List[torch.Tensor]:
        """
        STM: Variable-order Markov distributions from the user's recent history.

        Faithfully implements the IDyOM STM (Pearce, 2005) for *trajectory* data,
        adapted to address a structural issue with the original sequence-only
        formulation: in a single road-trip trajectory the same k-gram almost
        never reappears (a driver does not loop through the same intersection
        twice on the same trip), so scanning only the current sequence yields
        empty STM counts and the model silently collapses to a uniform prior.

        To make STM contribute meaningful information we instead scan the
        *user's recent history* (the last N nodes the same driver visited
        across previous trips) for occurrences of the query context
        ``c^{(k)} = v_{t-k+1:t}``.  For each match, the node that immediately
        followed is counted; the resulting counts are Laplace-smoothed and
        normalised.  If no per-user history is supplied (legacy behaviour),
        the current sequence is used as the history buffer, which preserves
        the previous API and unit-test contracts.

        Args:
            sequences: (B, seq_len) — current trajectory windows (LTM/query side).
                The last ``k`` nodes of each row form the query context for
                Markov order ``k``.
            user_history: optional (B, H) tensor of node IDs.  Row ``b``
                contains the most recent ``H`` nodes the user associated with
                trajectory ``b`` has visited across their previous trips,
                ordered chronologically (oldest first, newest last).  Padding
                values < 0 or >= vocab_size are ignored.  When ``None``, the
                current ``sequences`` tensor is used in its place.
            user_history_lengths: optional (B,) tensor of valid lengths into
                ``user_history``.  If ``None``, the full row width is scanned
                and out-of-vocab values are filtered out automatically.

        Returns:
            Tuple ``(distributions, evidence)`` where ``distributions`` is a
            list of K probability distributions, each (B, vocab), and
            ``evidence`` is a list of K (B,) tensors giving the number of
            history matches found for each order (the per-order support count
            used for count-discounted weighting).
        """
        batch_size, seq_len = sequences.shape
        device = sequences.device
        seqs_np = sequences.cpu().numpy().astype(np.int32)
        # Query contexts always come from the current sequence; the history
        # buffer is only used for counting matches and their successors.
        if user_history is not None:
            hist_np = user_history.cpu().numpy().astype(np.int32)
            hist_len = hist_np.shape[1]
            if user_history_lengths is not None:
                hist_lengths = user_history_lengths.cpu().numpy().astype(np.int32)
            else:
                hist_lengths = np.full(batch_size, hist_len, dtype=np.int32)
        else:
            hist_np = seqs_np
            hist_len = seq_len
            hist_lengths = np.full(batch_size, seq_len, dtype=np.int32)

        uniform  = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float32)
        V  = self.vocab_size

        distributions = []
        evidence = []
        for order in range(1, self.max_order + 1):
            # We need at least order+1 nodes in the history to form a (ctx, next)
            # pair, AND at least `order` nodes in the current sequence to form
            # the query context.  If the current window is shorter than `order`
            # (common with de Bruijn encoding, whose k-gram sequences are
            # order-1 nodes shorter), the order-k context cannot be formed, so
            # we back off to uniform — mirroring the LTM's `order <= seq_len`
            # guard.  Without this, `seqs_np[:, -order:]` silently returns only
            # `seq_len` columns and broadcasting against the full-width history
            # windows fails.
            n_windows_full = hist_len - order
            if n_windows_full <= 0 or order > seq_len:
                distributions.append(torch.from_numpy(
                    np.tile(uniform, (batch_size, 1))).to(device))
                evidence.append(torch.zeros(batch_size, device=device))
                continue

            # query[b] = last `order` nodes of the current sequence
            query = seqs_np[:, -order:]  # (B, order)

            # Build sliding windows over the history buffer:
            #   windows[b, i, :] = hist_np[b, i:i+order]
            #   nexts[b, i]      = hist_np[b, i+order]
            if order == 1:
                query1d  = query[:, 0]
                windows1d = hist_np[:, :n_windows_full]
                matches   = (windows1d == query1d[:, np.newaxis])  # (B, n_windows_full)
            else:
                windows = np.stack(
                    [hist_np[:, i:i + order] for i in range(n_windows_full)],
                    axis=1,
                )  # (B, n_windows_full, order)
                matches = np.all(windows == query[:, np.newaxis, :], axis=2)

            nexts = hist_np[:, order:order + n_windows_full]  # (B, n_windows_full)

            # Mask out positions beyond the per-row valid history length.
            # A pair starting at i is valid iff i + order < hist_lengths[b].
            valid_pos = (np.arange(n_windows_full)[np.newaxis, :] + order
                         < hist_lengths[:, np.newaxis])  # (B, n_windows_full)
            matches = matches & valid_pos

            # Scatter matched counts into (B, V)
            m_flat = matches.ravel().astype(np.float32)
            n_flat = nexts.ravel().astype(np.int64)
            b_flat = np.repeat(np.arange(batch_size, dtype=np.int64), n_windows_full)
            valid  = (m_flat > 0) & (n_flat >= 0) & (n_flat < V)
            if valid.any():
                lin_idx = b_flat[valid] * V + n_flat[valid]
                counts  = np.bincount(
                    lin_idx, weights=m_flat[valid], minlength=batch_size * V,
                ).reshape(batch_size, V).astype(np.float32)
            else:
                counts = np.zeros((batch_size, V), dtype=np.float32)

            any_match = matches.any(axis=1)  # (B,)
            # Posterior mean under a symmetric Dirichlet(beta) prior:
            #   P(v) = (count_v + beta/V) / (N + beta),   N = matches in the row.
            # Low-support rows (small N) stay near uniform -> high entropy ->
            # small inverse-entropy weight, so a few spurious high-order matches
            # on a small alphabet can no longer hijack the variable-order
            # mixture.  As N grows the STM sharpens and contributes more.
            beta = self._stm_prior_strength
            n_match = counts.sum(axis=1, keepdims=True)  # raw match count N per row
            post = (counts + beta / V) / (n_match + beta)
            dist_np = np.where(any_match[:, np.newaxis], post, uniform[np.newaxis, :])

            distributions.append(torch.from_numpy(dist_np.astype(np.float32)).to(device))
            # Evidence = number of (ctx, next) matches found in the user's history.
            ev_np = matches.sum(axis=1).astype(np.float32)
            evidence.append(torch.from_numpy(ev_np).to(device))

        return distributions, evidence

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, sequences: torch.Tensor,
                graph_structure: Optional[Dict] = None,
                user_history: Optional[torch.Tensor] = None,
                user_history_lengths: Optional[torch.Tensor] = None,
                return_components: bool = False):
        """
        Predict next node using dual IDyOM-style LTM/STM Markov chains.

        Steps (following IDyOM Pearce 2005):
        1. LTM: K Markov distributions from corpus graph transitions
        2. Entropy-weight combine K LTM orders  →  P_LTM
        3. STM: K Markov distributions from the user's recent history (or the
           current sequence if no history is provided)
        4. Entropy-weight combine K STM orders  →  P_STM
        5. Entropy-weight combine [P_LTM, P_STM]  →  P_combined
        6. Return log(P_combined)  (compatible with nn.CrossEntropyLoss)

        Args:
            sequences: (batch, seq_len) node index sequences (current trip).
            graph_structure: dict with 'edge_weights' (vocab, vocab) for LTM.
            user_history: optional (batch, H) tensor of recent node IDs from
                previous trips of the same user.  When ``None``, the STM falls
                back to scanning the current sequence (legacy behaviour, kept
                for backward compatibility — note this typically yields an
                uninformative STM on single trajectories).
            user_history_lengths: optional (batch,) tensor of valid lengths
                for each row of ``user_history``.

        Returns:
            logits: (batch, vocab_size) log-probabilities
        """
        # LTM branch: evidence-discounted inverse-entropy weighting over orders.
        # Uses the per-order training support count (already computed by
        # _compute_markov_from_graph via Jelinek-Mercer back-off) to discount
        # low-evidence high-order distributions, preventing overconfident spikes
        # from rare k-gram contexts hijacking the mixture.
        ltm_dists, ltm_ev = self._compute_markov_from_graph(sequences, graph_structure)
        p_ltm = self._evidence_weighted_combination(ltm_dists, ltm_ev)   # (batch, vocab)

        # STM branch: evidence-discounted inverse-entropy weighting over orders.
        # The per-order match count from the user's history discounts spurious
        # high-order STM matches on small alphabets.
        stm_dists, stm_ev = self._compute_markov_from_sequences(
            sequences, user_history=user_history,
            user_history_lengths=user_history_lengths,
        )
        p_stm = self._evidence_weighted_combination(stm_dists, stm_ev)   # (batch, vocab)

        # IDyOM-style LTM + STM combination, but with an order-aware per-user
        # personalisation gate: the STM overrides the population LTM only when
        # the user's recent path contains a long, well-supported, decisive match
        # for the current context.  With no such evidence this reduces to the
        # pure inverse-entropy combination.
        stm_support, stm_max_order = self._stm_support_stats(stm_ev, p_ltm.device)
        p_combined = self._combine_ltm_stm(p_ltm, p_stm, stm_support,
                                           stm_max_order)   # (batch, vocab)

        if return_components:
            # Diagnostic hook: expose the LTM-only / STM-only distributions and
            # the per-row user-history support so callers can measure how much
            # headroom the STM actually provides (LTM vs STM vs oracle Acc@1).
            if stm_support is None:
                stm_support = torch.zeros(sequences.shape[0], device=p_ltm.device)
            return (torch.log(p_combined.clamp(min=self._eps)),
                    p_ltm, p_stm, stm_support)

        return torch.log(p_combined.clamp(min=self._eps))

    def forward_with_explanations(self, sequences: torch.Tensor,
                                  graph_structure: Optional[Dict] = None,
                                  user_history: Optional[torch.Tensor] = None,
                                  user_history_lengths: Optional[torch.Tensor] = None
                                  ) -> Dict:
        """
        Forward pass with intermediate distributions for explainability.

        Returns dict with:
            logits:    (batch, vocab)  log-probabilities
            ltm_probs: (batch, vocab)  combined LTM distribution
            stm_probs: (batch, vocab)  combined STM distribution
            alpha_ltm: (batch,)        entropy-weight given to LTM
            alpha_stm: (batch,)        entropy-weight given to STM
        """
        ltm_dists, ltm_ev = self._compute_markov_from_graph(sequences, graph_structure)
        p_ltm = self._evidence_weighted_combination(ltm_dists, ltm_ev)

        stm_dists, stm_ev = self._compute_markov_from_sequences(
            sequences, user_history=user_history,
            user_history_lengths=user_history_lengths,
        )
        p_stm = self._evidence_weighted_combination(stm_dists, stm_ev)

        # Per-user STM evidence (total matches + highest matched order) drives the gate.
        stm_support, stm_max_order = self._stm_support_stats(stm_ev, p_ltm.device)

        # Evidence-gated weights, reported for explainability. With no user
        # history this matches the pure inverse-entropy weights.
        h_ltm = self._entropy(p_ltm)                              # (batch,)
        h_stm = self._entropy(p_stm)
        inv_h_ltm = 1.0 / (h_ltm + self._eps)
        inv_h_stm = 1.0 / (h_stm + self._eps)
        if stm_support is not None and self._stm_personalization_alpha > 0.0:
            conf = stm_support / (stm_support + self._stm_personalization_tau)
            inv_h_stm = inv_h_stm * (1.0 + self._stm_personalization_alpha * conf)
        total_inv = inv_h_ltm + inv_h_stm
        alpha_ltm = inv_h_ltm / total_inv
        alpha_stm = inv_h_stm / total_inv

        p_combined = self._combine_ltm_stm(p_ltm, p_stm, stm_support, stm_max_order)
        logits = torch.log(p_combined.clamp(min=self._eps))

        return {
            'logits': logits,
            'ltm_probs': p_ltm,
            'stm_probs': p_stm,
            'alpha_ltm': alpha_ltm,
            'alpha_stm': alpha_stm,
        }

    @classmethod
    def get_strategy_info(cls) -> Dict[str, str]:
        """Get available strategies and their descriptions."""
        return cls.STRATEGIES.copy()


# ============================================================================
# HELPER FUNCTIONS: Strategy Implementations
# ============================================================================

def create_degree_features(graph: nx.DiGraph, vocab_size: int) -> Callable:
    """
    Create feature function for degree-based embedding.
    
    Features: normalized in-degree and out-degree
    """
    in_degrees = np.zeros(vocab_size)
    out_degrees = np.zeros(vocab_size)
    
    for node in graph.nodes():
        if node < vocab_size:
            in_degrees[node] = graph.in_degree(node)
            out_degrees[node] = graph.out_degree(node)
    
    in_degrees = in_degrees / (in_degrees.max() + 1e-6)
    out_degrees = out_degrees / (out_degrees.max() + 1e-6)
    
    def fn(node_id: int) -> np.ndarray:
        return np.array([in_degrees[node_id], out_degrees[node_id]])
    
    return fn


def create_pagerank_features(graph: nx.DiGraph, vocab_size: int) -> Callable:
    """
    Create feature function for PageRank-based embedding.
    
    Features: PageRank scores and in-degree
    """
    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except Exception:
        pagerank = {i: 1.0 / vocab_size for i in range(vocab_size)}
    
    pagerank_arr = np.array([pagerank.get(i, 1.0/vocab_size) for i in range(vocab_size)])
    pagerank_arr = pagerank_arr / (pagerank_arr.max() + 1e-6)
    
    in_degrees = np.array([graph.in_degree(i) for i in range(vocab_size)])
    in_degrees = in_degrees / (in_degrees.max() + 1e-6)
    
    def fn(node_id: int) -> np.ndarray:
        return np.array([pagerank_arr[node_id], in_degrees[node_id]])
    
    return fn


def create_combined_features(graph: nx.DiGraph, vocab_size: int) -> Callable:
    """
    Create feature function for combined features.
    
    Features: in-degree, out-degree, PageRank, and clustering coefficient
    """
    # Degree
    in_degrees = np.array([graph.in_degree(i) for i in range(vocab_size)])
    out_degrees = np.array([graph.out_degree(i) for i in range(vocab_size)])
    in_degrees = in_degrees / (in_degrees.max() + 1e-6)
    out_degrees = out_degrees / (out_degrees.max() + 1e-6)
    
    # PageRank
    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except Exception:
        pagerank = {i: 1.0 / vocab_size for i in range(vocab_size)}
    pagerank_arr = np.array([pagerank.get(i, 1.0/vocab_size) for i in range(vocab_size)])
    pagerank_arr = pagerank_arr / (pagerank_arr.max() + 1e-6)
    
    # Clustering
    undirected = graph.to_undirected()
    clustering = np.array([nx.clustering(undirected, i) if i in undirected else 0.0 
                          for i in range(vocab_size)])
    
    def fn(node_id: int) -> np.ndarray:
        return np.array([in_degrees[node_id], out_degrees[node_id], 
                        pagerank_arr[node_id], clustering[node_id]])
    
    return fn


# Mapping of strategy names to feature functions
FEATURE_CREATORS = {
    'degree': create_degree_features,
    'pagerank': create_pagerank_features,
    'combined': create_combined_features,
}


def create_model_with_strategy(
    graph: nx.DiGraph,
    vocab_size: int,
    embedding_strategy: str = 'degree',
    embedding_dim: int = 128,
    max_order: int = 7,
    use_neural_weighting: bool = False
) -> GraphIDyOMEnhanced:
    """
    Create a GraphIDyOM pure Markov chain model.

    The graph argument is accepted for API compatibility but is not used
    internally — the graph structure is supplied at inference time via
    graph_structure (see build_graph_structure_from_sequences).
    embedding_strategy, embedding_dim, and use_neural_weighting are ignored.

    Args:
        graph: NetworkX DiGraph (unused, kept for API compat)
        vocab_size: Number of unique nodes
        max_order: Maximum Markov order K

    Returns:
        Initialized GraphIDyOMEnhanced model

    Example:
        >>> import networkx as nx
        >>> graph = nx.DiGraph()
        >>> graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        >>> model = create_model_with_strategy(graph, vocab_size=3, max_order=5)
    """
    return GraphIDyOMEnhanced(vocab_size=vocab_size, max_order=max_order)


# ============================================================================
# HELPER: Build graph structure from sequences
# ============================================================================

def build_graph_structure_from_sequences(sequences: torch.Tensor, vocab_size: int,
                                          max_order: int = 1) -> Dict:
    """
    Build graph structure from node sequences.

    For the LTM:
      - 1st-order transition matrix (vectorised, GPU-friendly).
      - True k-gram count tables for orders 2..min(max_order, seq_len-1),
        stored as {context_tuple: np.ndarray(vocab_size)} dicts.
        These give P_LTM(next | last-k context) with back-off to order 1
        when the k-gram has never been seen in training.

    Args:
        sequences: (N, seq_len) tensor of node indices
        vocab_size: Number of unique nodes
        max_order: Maximum Markov order; k-gram tables built for 2..max_order

    Returns:
        Dict with 'edge_weights', 'adjacency', 'in_degrees', 'out_degrees',
        'kgram_tables'.
    """
    batch_size, seq_len = sequences.shape

    # --- 1st-order transition matrix (vectorised) ---
    from_nodes = sequences[:, :-1].reshape(-1)
    to_nodes   = sequences[:, 1:].reshape(-1)
    valid_mask = ((from_nodes >= 0) & (from_nodes < vocab_size) &
                  (to_nodes   >= 0) & (to_nodes   < vocab_size))
    from_nodes = from_nodes[valid_mask]
    to_nodes   = to_nodes[valid_mask]

    # Building a dense VxV matrix can explode for large token vocabularies
    # (notably debruijn encoding). Use sparse first-order rows above a threshold.
    dense_edge_max_vocab = 12000
    edge_weights = None
    first_order_sparse = None
    if vocab_size <= dense_edge_max_vocab:
        edge_counts = torch.zeros(vocab_size, vocab_size, device='cpu', dtype=torch.float32)
        edge_indices = from_nodes.to('cpu') * vocab_size + to_nodes.to('cpu')
        edge_counts.view(-1).scatter_add_(
            0, edge_indices, torch.ones_like(edge_indices, dtype=torch.float32))

        row_sums   = edge_counts.sum(dim=1, keepdim=True)
        row_sums   = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        edge_weights = edge_counts / row_sums   # rows sum to 1; self-loops only if present in data

        # Per-node total out-transition counts (order-1 training support).
        out_counts = edge_counts.sum(dim=1)

        adjacency  = (edge_counts > 0).float()
        adjacency.fill_diagonal_(1.0)           # self-loops for degree computation
        in_degrees  = adjacency.sum(dim=0)
        out_degrees = adjacency.sum(dim=1)
    else:
        from_np = from_nodes.detach().cpu().numpy().astype(np.int64)
        to_np   = to_nodes.detach().cpu().numpy().astype(np.int64)

        if len(from_np) > 0:
            lin = from_np * np.int64(vocab_size) + to_np
            uniq_lin, uniq_cnt = np.unique(lin, return_counts=True)
            src = (uniq_lin // np.int64(vocab_size)).astype(np.int64)
            dst = (uniq_lin %  np.int64(vocab_size)).astype(np.int32)
            cnt = uniq_cnt.astype(np.float32)

            order = np.argsort(src, kind='mergesort')
            src = src[order]
            dst = dst[order]
            cnt = cnt[order]
        else:
            src = np.empty(0, dtype=np.int64)
            dst = np.empty(0, dtype=np.int32)
            cnt = np.empty(0, dtype=np.float32)

        row_ptr = np.zeros(vocab_size + 1, dtype=np.int64)
        if len(src) > 0:
            np.add.at(row_ptr, src + 1, 1)
        row_ptr = np.cumsum(row_ptr)

        out_deg_np = np.diff(row_ptr).astype(np.float32)
        in_deg_np  = np.zeros(vocab_size, dtype=np.float32)
        if len(dst) > 0:
            np.add.at(in_deg_np, dst, 1.0)

        # Keep these keys available for API compatibility.
        adjacency   = torch.empty((0, 0), dtype=torch.float32)
        in_degrees  = torch.from_numpy(in_deg_np + 1.0)
        out_degrees = torch.from_numpy(out_deg_np + 1.0)
        # Per-node total out-transition counts (order-1 training support).
        out_counts_np = np.zeros(vocab_size, dtype=np.float32)
        if len(src) > 0:
            np.add.at(out_counts_np, src.astype(np.int64), cnt)
        out_counts = torch.from_numpy(out_counts_np)
        first_order_sparse = {
            '_row_ptr': row_ptr,
            '_next_idx': dst,
            '_next_cnt': cnt,
        }

    result = {
        'adjacency':    adjacency,
        'edge_weights': edge_weights,
        'first_order_sparse': first_order_sparse,
        'out_counts':   out_counts,
        'in_degrees':   in_degrees,
        'out_degrees':  out_degrees,
        'kgram_tables': {},
    }

    # --- k-gram tables for orders 2..min(max_order, seq_len-1) ---
    # Each table[k] maps context tuple (k ints) -> raw count array (vocab_size,).
    # Counts are stored un-normalised; smoothing + normalisation happens in forward().
    # Each table stores precomputed lookup arrays for fast vectorised inference:
    #   table['_sorted_ids']   int64 (n_unique,)   – sorted encoded context IDs
    #   table['_row_ptr']      int64 (n_unique+1,) – CSR row pointers
    #   table['_next_idx']     int32 (nnz,)        – next-node IDs per row
    #   table['_next_cnt']     float32 (nnz,)      – raw counts per (context,next)
    # Legacy format with table['_dist_matrix'] is still supported at inference.
    max_k = min(max_order, seq_len - 1)
    if max_k >= 2:
        # Use int64-encoded keys where possible (V^(k+1) must fit in int64)
        _INT64_MAX = np.iinfo(np.int64).max
        seqs_np = sequences.cpu().numpy().astype(np.int64)
        V = np.int64(vocab_size)
        for k in tqdm(range(2, max_k + 1), desc="Building k-gram tables", leave=False):
            n_pos = seq_len - k
            if n_pos <= 0:
                result['kgram_tables'][k] = {}
                continue

            # Check whether combined key fits in int64: V^(k+1) <= INT64_MAX?
            overflow = False
            v_pow = 1
            for _ in range(k + 1):
                if v_pow > _INT64_MAX // vocab_size:
                    overflow = True
                    break
                v_pow *= vocab_size

            if not overflow:
                # --- Fully vectorised path ---
                # Encode (ctx[0], ..., ctx[k-1], nxt) as a single int64 key
                # key = ctx[0]*V^k + ... + ctx[k-1]*V + nxt  (mixed-radix, base V)
                all_keys = np.empty(len(seqs_np) * n_pos, dtype=np.int64)
                for i in tqdm(range(n_pos), desc=f"  k={k} windows", leave=False):
                    ctx_win = seqs_np[:, i:i + k]   # (N, k)
                    nxt_col = seqs_np[:, i + k]      # (N,)
                    key = np.zeros(len(seqs_np), dtype=np.int64)
                    for j in range(k):
                        key = key * V + ctx_win[:, j]
                    key = key * V + nxt_col
                    all_keys[i * len(seqs_np):(i + 1) * len(seqs_np)] = key

                unique_keys, cts = np.unique(all_keys, return_counts=True)

                # Decode nxt (lowest bits) and ctx (remaining bits)
                nxt_arr  = (unique_keys % V).astype(np.int32)
                ctx_enc  = unique_keys // V
                ctx_cols = []
                tmp = ctx_enc.copy()
                for _ in range(k):
                    ctx_cols.append((tmp % V).astype(np.int32))
                    tmp //= V
                ctx_cols.reverse()   # ctx_cols[0] = first context element

                table: Dict[tuple, Dict[int, float]] = {}
                for idx in range(len(unique_keys)):
                    nv = int(nxt_arr[idx])
                    if 0 <= nv < vocab_size:
                        ctx_t = tuple(int(ctx_cols[j][idx]) for j in range(k))
                        if ctx_t not in table:
                            table[ctx_t] = {}
                        table[ctx_t][nv] = table[ctx_t].get(nv, 0.0) + float(cts[idx])

            else:
                # --- Fallback: loop over positions (n_pos ≤ seq_len), batch via np.unique ---
                table = {}
                for i in tqdm(range(n_pos), desc=f"  k={k} windows", leave=False):
                    ctx_win = seqs_np[:, i:i + k]            # (N, k)
                    nxt_col = seqs_np[:, i + k][:, np.newaxis]  # (N, 1)
                    pairs   = np.concatenate([ctx_win, nxt_col], axis=1)  # (N, k+1)
                    upairs, ucts = np.unique(pairs, axis=0, return_counts=True)
                    for pair, cnt in zip(upairs, ucts):
                        nv = int(pair[k])
                        if 0 <= nv < vocab_size:
                            ctx_t = tuple(int(pair[j]) for j in range(k))
                            if ctx_t not in table:
                                table[ctx_t] = {}
                            table[ctx_t][nv] = table[ctx_t].get(nv, 0.0) + float(cnt)

            # Pre-build fast lookup arrays for vectorised inference in _compute_markov_from_graph.
            # Store k-gram rows in sparse CSR-like form to avoid O(n_ctx * vocab) memory.
            if table:
                ctx_list = list(table.keys())
                # Encode context tuples → int64 IDs for fast searchsorted lookup
                ids = np.empty(len(ctx_list), dtype=np.int64)
                for ii, ctx_t in enumerate(ctx_list):
                    key_i = np.int64(0)
                    for vv in ctx_t:
                        key_i = key_i * V + np.int64(vv)
                    ids[ii] = key_i
                sort_order = np.argsort(ids)
                sorted_ctx = [ctx_list[i] for i in sort_order]

                row_ptr = [0]
                next_idx_flat: List[int] = []
                next_cnt_flat: List[float] = []
                for ctx_t in sorted_ctx:
                    nxt_to_cnt = table[ctx_t]
                    nxt_ids = sorted(nxt_to_cnt.keys())
                    for nxt in nxt_ids:
                        next_idx_flat.append(int(nxt))
                        next_cnt_flat.append(float(nxt_to_cnt[nxt]))
                    row_ptr.append(len(next_idx_flat))

                table = {
                    '_sorted_ids': ids[sort_order],
                    '_row_ptr': np.array(row_ptr, dtype=np.int64),
                    '_next_idx': np.array(next_idx_flat, dtype=np.int32),
                    '_next_cnt': np.array(next_cnt_flat, dtype=np.float32),
                }
            result['kgram_tables'][k] = table

    return result
