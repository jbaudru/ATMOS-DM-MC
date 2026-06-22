"""
Node encoding strategies for trajectory prediction.

This module implements three different node encoding approaches:
1. Categorical ID Encoding: Simple integer node IDs (baseline)
2. de Bruijn (Higher-order) Encoding: k-grams of nodes for explicit higher-order context
3. Node2Vec Encoding: Learned graph embeddings augmented with bearing angles
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder

try:
    from node2vec import Node2Vec
except ImportError:
    Node2Vec = None


class CategoricalIDEncoding:
    """
    Baseline: Direct categorical encoding - each node maps to a discrete integer ID.
    
    This is the simplest encoding, mapping each road intersection to an integer
    in the range [0, vocab_size).
    """
    
    def __init__(self, node_encoder: LabelEncoder):
        self.node_encoder = node_encoder
        self.vocab_size = len(node_encoder.classes_)
        self.name = "categorical_id"
        
    def encode_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Encode trajectory using categorical IDs.
        
        Args:
            trajectory: Array of node IDs
            
        Returns:
            Encoded trajectory (same as input after label encoding)
        """
        return self.node_encoder.transform(trajectory)
    
    def decode_trajectory(self, encoded: np.ndarray) -> np.ndarray:
        """Decode trajectory back to original node IDs."""
        return self.node_encoder.inverse_transform(encoded)
    
    def get_config(self) -> Dict:
        return {
            'encoding_type': 'categorical_id',
            'vocab_size': self.vocab_size,
            'description': 'Simple categorical integer encoding'
        }


class DeBruijnEncoding:
    """
    Higher-order (de Bruijn) encoding: k-grams of base nodes.
    
    Instead of encoding individual nodes, encodes k-grams (sequences of k nodes).
    This makes higher-order Markov chains explicit in the state space.
    
    For order k, state space size is at most N^k where N is the number of base nodes.
    """
    
    def __init__(self, node_encoder: LabelEncoder, order: int = 2):
        """
        Args:
            node_encoder: LabelEncoder for base nodes
            order: k for k-gram encoding (default 2)
        """
        self.node_encoder = node_encoder
        self.order = order
        self.base_vocab_size = len(node_encoder.classes_)
        
        # Create mapping from k-grams to IDs
        self.kgram_to_id = {}
        self.id_to_kgram = {}
        self.vocab_size = 0
        
        self.name = f"debruijn_order{order}"
    
    def build_kgram_vocabulary(self, trajectories: List[np.ndarray]) -> None:
        """
        Build vocabulary of valid k-grams from training trajectories.
        
        Args:
            trajectories: List of trajectory arrays
        """
        valid_kgrams = set()
        
        for traj in trajectories:
            for i in range(len(traj) - self.order + 1):
                kgram = tuple(traj[i:i+self.order])
                valid_kgrams.add(kgram)
        
        # Assign IDs to valid k-grams
        self.vocab_size = len(valid_kgrams)
        for idx, kgram in enumerate(sorted(valid_kgrams)):
            self.kgram_to_id[kgram] = idx
            self.id_to_kgram[idx] = kgram
    
    def encode_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Encode trajectory as sequence of k-grams.
        
        Args:
            trajectory: Array of base node IDs
            
        Returns:
            Encoded trajectory of k-gram IDs
        """
        # Encode base nodes first
        encoded_base = self.node_encoder.transform(trajectory)
        
        # Convert to k-grams
        kgram_encoded = []
        for i in range(len(encoded_base) - self.order + 1):
            kgram = tuple(encoded_base[i:i+self.order])
            if kgram in self.kgram_to_id:
                kgram_encoded.append(self.kgram_to_id[kgram])
        
        return np.array(kgram_encoded, dtype=np.long)
    
    def decode_trajectory(self, encoded: np.ndarray) -> np.ndarray:
        """Decode k-gram trajectory back to base node IDs."""
        # Convert k-gram IDs to base nodes
        # Use first node of each k-gram, plus final node of last k-gram
        decoded = []
        
        for idx in encoded:
            if idx in self.id_to_kgram:
                kgram = self.id_to_kgram[idx]
                if not decoded:
                    # First k-gram - add all nodes
                    decoded.extend(kgram)
                else:
                    # Subsequent k-grams - add only the last node
                    decoded.append(kgram[-1])
        
        # Decode to original node IDs
        return self.node_encoder.inverse_transform(decoded)
    
    def get_config(self) -> Dict:
        return {
            'encoding_type': 'debruijn',
            'order': self.order,
            'base_vocab_size': self.base_vocab_size,
            'kgram_vocab_size': self.vocab_size,
            'description': f'{self.order}-gram de Bruijn encoding'
        }


class Node2VecEncoding:
    """
    Node2Vec encoding: Learned graph embeddings augmented with bearing angles.
    
    Based on the paper specification: each intersection v_t is represented as a 
    continuous vector e_t ∈ ℝ^d obtained via Node2Vec, augmented with a 
    discretized bearing angle θ_t ∈ [0, 2π):
    
        φ(v_t) = [e_t || θ_t] ∈ ℝ^(d+1)
    
    where || denotes concatenation.
    
    Node2Vec learns embeddings by simulating biased random walks on the graph,
    capturing both local and global network structure.
    """
    
    def __init__(self, node_encoder: LabelEncoder, graph: nx.DiGraph, 
                 embedding_dim: int = 64, walks_per_node: int = 10,
                 walk_length: int = 80, p: float = 1.0, q: float = 1.0,
                 window_size: int = 10, min_count: int = 1, workers: int = 4):
        """
        Args:
            node_encoder: LabelEncoder for base nodes
            graph: NetworkX graph for embedding learning
            embedding_dim: Dimension of Node2Vec embeddings (default 64)
            walks_per_node: Number of random walks per node (default 10)
            walk_length: Length of each random walk (default 80)
            p: BFS parameter (default 1.0)
            q: DFS parameter (default 1.0)
            window_size: Context window for word2vec (default 10)
            min_count: Minimum word count for word2vec (default 1)
            workers: Number of workers for training (default 4)
        """
        if Node2Vec is None:
            raise ImportError("node2vec is required for Node2VecEncoding. Install with: pip install node2vec")
        
        self.node_encoder = node_encoder
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.base_vocab_size = len(node_encoder.classes_)
        
        # For Node2Vec: embedding_dim + 1 (for bearing angle)
        self.vocab_size = self.base_vocab_size
        self.output_dim = embedding_dim + 1  # embeddings + bearing angle
        
        self.name = "node2vec"
        
        # Train Node2Vec embeddings
        self._train_node2vec(walks_per_node, walk_length, p, q, 
                            window_size, min_count, workers)
        
        # Compute bearing angles
        self._compute_bearing_angles()
    
    def _train_node2vec(self, walks_per_node: int, walk_length: int, 
                       p: float, q: float, window_size: int, 
                       min_count: int, workers: int) -> None:
        """
        Train Node2Vec embeddings on the graph.
        
        Args:
            walks_per_node: Number of random walks per node
            walk_length: Length of each random walk
            p: Return parameter (controls how likely the walk returns to prev node)
            q: In-out parameter (controls BFS vs DFS)
            window_size: Skip-gram window size
            min_count: Minimum word count for word2vec
            workers: Number of workers for training
        """
        print(f"Training Node2Vec embeddings (dim={self.embedding_dim})...")
        print(f"  Walks per node: {walks_per_node}")
        print(f"  Walk length: {walk_length}")
        print(f"  p={p}, q={q}")
        
        # Create Node2Vec object
        node2vec = Node2Vec(
            graph=self.graph,
            dimensions=self.embedding_dim,
            walk_length=walk_length,
            num_walks=walks_per_node,
            workers=workers,
            p=p,
            q=q,
            seed=42
        )
        
        # Train embeddings (using word2vec)
        model = node2vec.fit(window=window_size, min_count=min_count, workers=workers)
        
        # Extract embeddings as numpy array
        self.embeddings = np.zeros((self.base_vocab_size, self.embedding_dim))
        
        for node_id in range(self.base_vocab_size):
            if str(node_id) in model.wv:
                self.embeddings[node_id] = model.wv[str(node_id)]
            else:
                # Random initialization for nodes not in walks
                self.embeddings[node_id] = np.random.randn(self.embedding_dim) * 0.01
        
        print(f"  Node2Vec training complete. Embeddings shape: {self.embeddings.shape}")
    
    def _compute_bearing_angles(self) -> None:
        """
        Compute discretized bearing angles for each node.
        
        The bearing angle represents the predominant direction of outgoing edges.
        For each node, we compute the weighted average direction to its neighbors.
        
        Discretized to 16 levels in [0, 2π):
            θ_t = floor(atan2(Δy, Δx) / (2π / 16)) * (2π / 16)
        """
        self.bearing_angles = np.zeros(self.base_vocab_size)
        num_bins = 16  # Discretize bearing angle to 16 levels
        
        # For each node, compute predominant direction to outgoing neighbors
        for node_id in range(self.base_vocab_size):
            if self.graph.out_degree(node_id) > 0:
                # Get outgoing neighbors and their directions
                angles = []
                for neighbor in self.graph.successors(node_id):
                    if neighbor < self.base_vocab_size:
                        # Approximate direction as angle based on node IDs
                        # In real scenario, would use lat/lon coordinates
                        angle = (2 * np.pi * (neighbor - node_id)) / self.base_vocab_size
                        angle = angle % (2 * np.pi)
                        angles.append(angle)
                
                if angles:
                    # Circular mean of angles
                    sin_sum = np.sum(np.sin(angles))
                    cos_sum = np.sum(np.cos(angles))
                    mean_angle = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
                    
                    # Discretize to 16 levels
                    self.bearing_angles[node_id] = (mean_angle / (2 * np.pi))
                else:
                    self.bearing_angles[node_id] = 0.0
            else:
                self.bearing_angles[node_id] = 0.0
    
    def encode_trajectory(self, trajectory: np.ndarray) -> torch.Tensor:
        """
        Encode trajectory as Node2Vec embeddings + bearing angles.
        
        Args:
            trajectory: Array of node IDs
            
        Returns:
            Tensor of shape (len(trajectory), embedding_dim + 1)
        """
        # Encode base nodes
        encoded_base = self.node_encoder.transform(trajectory)
        
        # Get embeddings and bearing angles
        encodings = []
        for node_id in encoded_base:
            if node_id < len(self.embeddings):
                # Concatenate: [embedding_vector || bearing_angle]
                embedding = self.embeddings[node_id]
                bearing = self.bearing_angles[node_id]
                full_encoding = np.concatenate([embedding, [bearing]])
                encodings.append(full_encoding)
        
        return torch.tensor(np.array(encodings), dtype=torch.float32)
    
    def decode_trajectory(self, encoded: torch.Tensor) -> np.ndarray:
        """
        Decode Node2Vec embeddings back to node IDs (approximate).
        
        Args:
            encoded: Tensor of shape (trajectory_length, embedding_dim + 1)
            
        Returns:
            Approximate node IDs based on nearest neighbor in embedding space
        """
        # Extract only embeddings (remove bearing angle for matching)
        encoded_np = encoded.numpy() if isinstance(encoded, torch.Tensor) else encoded
        embeddings_only = encoded_np[:, :-1]  # Remove bearing angle column
        
        decoded = []
        for emb in embeddings_only:
            # Find nearest node in embedding space
            distances = np.linalg.norm(self.embeddings - emb, axis=1)
            nearest_node = np.argmin(distances)
            decoded.append(nearest_node)
        
        return self.node_encoder.inverse_transform(decoded)
    
    def get_config(self) -> Dict:
        return {
            'encoding_type': 'node2vec',
            'embedding_dim': self.embedding_dim,
            'output_dim': self.output_dim,
            'base_vocab_size': self.base_vocab_size,
            'description': 'Node2Vec embeddings augmented with bearing angles'
        }


class GraphStructuralDiscretizedEncoding:
    """
    Discretized graph embedding compatible with discrete Markov chains.

    Clusters every node into one of ``n_clusters`` groups with k-means and
    replaces each node by its cluster ID, producing an integer state space in
    ``[0, n_clusters)`` that variable-order Markov models can consume directly.

    The clustering feature space is selected by ``method``:

    * ``spatial``    – the node's ``(x, y)`` coordinates, so clusters are
      geographically coherent (nearby intersections group together).  This
      retains the most next-node signal and is the recommended default;
      requires ``coords``.
    * ``node2vec``   – Node2Vec graph embeddings (optional ``node2vec``
      dependency); falls back to ``spatial`` then ``structural``.
    * ``structural`` – legacy degree / in-degree / out-degree / PageRank
      features.

    Using many clusters (e.g. 512–2048) keeps the discretisation fine-grained:
    with a coarse vocabulary (e.g. 64) ~1000 nodes share a cluster and almost
    all node identity is lost.
    """

    def __init__(self, node_encoder: LabelEncoder, graph: nx.DiGraph,
                 n_clusters: int = 1024, method: str = "spatial",
                 coords: Optional[np.ndarray] = None, seed: int = 42):
        """
        Args:
            node_encoder : LabelEncoder mapping raw node IDs to [0, V).
            graph        : NetworkX DiGraph whose nodes are in [0, V).
            n_clusters   : Number of k-means clusters (= discrete vocab size).
            method       : 'spatial', 'node2vec', or 'structural'.
            coords       : Optional (V, 2) array of node coordinates for the
                           spatial method.
            seed         : Random seed for k-means.
        """
        self.node_encoder   = node_encoder
        self.base_vocab_size = len(node_encoder.classes_)
        self._fit(graph, n_clusters, method, coords, seed)

    # ------------------------------------------------------------------
    @staticmethod
    def _impute_columns(features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float64).copy()
        for c in range(feats.shape[1]):
            col = feats[:, c]
            bad = ~np.isfinite(col)
            if bad.all():
                col[:] = 0.0
            elif bad.any():
                col[bad] = float(np.nanmean(col[~bad]))
        return feats

    def _structural_features(self, G: nx.DiGraph) -> np.ndarray:
        n = self.base_vocab_size
        features = np.zeros((n, 4), dtype=np.float64)
        for v in range(n):
            if G.has_node(v):
                features[v, 0] = G.degree(v)
                features[v, 1] = G.in_degree(v)
                features[v, 2] = G.out_degree(v)
        try:
            pr = nx.pagerank(G, max_iter=200)
            for v, rank in pr.items():
                if isinstance(v, int) and 0 <= v < n:
                    features[v, 3] = rank
        except Exception:
            pass  # leave PageRank column zero if it fails
        return features

    def _node2vec_features(self, G: nx.DiGraph, dim: int = 64,
                           seed: int = 42) -> Optional[np.ndarray]:
        if Node2Vec is None:
            return None
        try:
            n2v = Node2Vec(G, dimensions=dim, walk_length=40, num_walks=10,
                           workers=4, seed=seed, quiet=True)
            model = n2v.fit(window=10, min_count=1)
        except Exception:
            return None
        emb = np.zeros((self.base_vocab_size, dim), dtype=np.float64)
        for i in range(self.base_vocab_size):
            if str(i) in model.wv:
                emb[i] = model.wv[str(i)]
        return emb

    # ------------------------------------------------------------------
    def _fit(self, G: nx.DiGraph, n_clusters: int, method: str,
             coords: Optional[np.ndarray], seed: int) -> None:
        """Compute clustering features and run k-means."""
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        n = self.base_vocab_size
        method = (method or "spatial").lower()
        used_method = method
        features: Optional[np.ndarray] = None

        if method == "spatial":
            if coords is not None and np.isfinite(coords).any():
                features = self._impute_columns(coords)
            else:
                used_method = "structural"
        elif method == "node2vec":
            emb = self._node2vec_features(G, seed=seed)
            if emb is not None:
                features = emb
            elif coords is not None and np.isfinite(coords).any():
                used_method = "spatial"
                features = self._impute_columns(coords)
            else:
                used_method = "structural"
        elif method != "structural":
            used_method = "structural"

        if features is None:
            used_method = "structural"
            features = self._structural_features(G)

        features_scaled = StandardScaler().fit_transform(features)
        actual_k = min(n_clusters, n)
        if actual_k > 256 or n > 20_000:
            km = MiniBatchKMeans(n_clusters=actual_k, random_state=seed,
                                 batch_size=max(1024, 3 * actual_k),
                                 n_init=3, max_iter=200)
        else:
            km = KMeans(n_clusters=actual_k, random_state=seed, n_init="auto")
        self.node_to_cluster: np.ndarray = km.fit_predict(
            features_scaled).astype(np.int64)
        self.vocab_size = actual_k
        self.method     = used_method
        self.name       = f"graph_embedding_{actual_k}c_{used_method}"

    # ------------------------------------------------------------------
    def encode_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Encode a raw trajectory (original node IDs) to cluster IDs.

        Args:
            trajectory: 1-D array of raw node IDs (before label encoding).

        Returns:
            1-D array of integer cluster IDs in [0, vocab_size).
        """
        encoded_base = self.node_encoder.transform(trajectory)
        return self.node_to_cluster[encoded_base]

    def encode_from_categorical(self, cat_trajectory: np.ndarray) -> np.ndarray:
        """
        Encode an already-categorical trajectory (integer IDs in [0, V)) to
        cluster IDs.  Use this when node IDs have already been remapped.

        Args:
            cat_trajectory: 1-D array of integers in [0, base_vocab_size).

        Returns:
            1-D array of integer cluster IDs in [0, vocab_size).
        """
        return self.node_to_cluster[np.asarray(cat_trajectory, dtype=np.int64)]

    def decode_trajectory(self, encoded: np.ndarray) -> np.ndarray:
        """Decoding is lossy (many-to-one); returns cluster IDs unchanged."""
        return encoded

    def get_config(self) -> Dict:
        feature_desc = {
            "spatial":    ["x", "y"],
            "node2vec":   ["node2vec_embedding"],
            "structural": ["degree", "in_degree", "out_degree", "pagerank"],
        }.get(getattr(self, "method", "structural"),
              ["degree", "in_degree", "out_degree", "pagerank"])
        return {
            "encoding_type": "graph_embedding",
            "method":        getattr(self, "method", "structural"),
            "n_clusters":    self.vocab_size,
            "base_vocab_size": self.base_vocab_size,
            "features":      feature_desc,
            "description":   (
                f"K-means discretisation ({self.vocab_size} clusters) on "
                f"{getattr(self, 'method', 'structural')} features"
            ),
        }


def create_encoding(encoding_type: str, node_encoder: LabelEncoder, 
                   graph: Optional[nx.DiGraph] = None,
                   trajectories: Optional[List[np.ndarray]] = None,
                   **kwargs) -> object:
    """
    Factory function to create encoding strategy.
    
    Args:
        encoding_type: 'categorical_id', 'debruijn', or 'node2vec'
        node_encoder: LabelEncoder for base nodes
        graph: NetworkX graph (required for node2vec)
        trajectories: Training trajectories (required for debruijn)
        **kwargs: Additional arguments passed to encoding class
        
    Returns:
        Encoding object
    """
    if encoding_type == 'categorical_id':
        return CategoricalIDEncoding(node_encoder)
    
    elif encoding_type == 'debruijn':
        order = kwargs.get('order', 2)
        encoding = DeBruijnEncoding(node_encoder, order=order)
        if trajectories is not None:
            encoding.build_kgram_vocabulary(trajectories)
        return encoding
    
    elif encoding_type == 'graph_embedding':
        # Discretised structural encoding – compatible with discrete Markov models.
        if graph is None:
            raise ValueError("graph_embedding encoding requires a NetworkX graph")
        n_clusters = kwargs.get('n_clusters', 1024)
        method     = kwargs.get('method', 'spatial')
        coords     = kwargs.get('coords', None)
        return GraphStructuralDiscretizedEncoding(node_encoder, graph,
                                                  n_clusters=n_clusters,
                                                  method=method, coords=coords)

    elif encoding_type == 'node2vec':
        # Original continuous Node2Vec (requires the node2vec package).
        if Node2Vec is None:
            raise ImportError(
                "node2vec is required for node2vec encoding. "
                "Install with: pip install node2vec"
            )
        if graph is None:
            raise ValueError("node2vec encoding requires a NetworkX graph")
        embedding_dim  = kwargs.get('embedding_dim', 64)
        walks_per_node = kwargs.get('walks_per_node', 10)
        walk_length    = kwargs.get('walk_length', 80)
        p              = kwargs.get('p', 1.0)
        q              = kwargs.get('q', 1.0)
        return Node2VecEncoding(node_encoder, graph,
                                embedding_dim=embedding_dim,
                                walks_per_node=walks_per_node,
                                walk_length=walk_length, p=p, q=q)

    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
