"""
Neural network models and probabilistic baselines for trajectory prediction.
"""

from .graph_idyom import GraphIDyOMEnhanced, CustomNodeEmbedding, create_model_with_strategy
from .baselines import AKOM, CPTPlus, MOGen, IOHMM, SimpleMarkov, get_baseline
from .node_decode import (
    NodeSpaceAdapter,
    NodeMetricAccumulator,
    load_kgram_to_id,
    load_node_to_cluster,
)

# Backwards compatibility alias
GraphIDyOMPredictor = GraphIDyOMEnhanced

__all__ = [
    'GraphIDyOMEnhanced',
    'GraphIDyOMPredictor',
    'CustomNodeEmbedding',
    'create_model_with_strategy',
    # Baselines
    'AKOM',
    'CPTPlus',
    'MOGen',
    'IOHMM',
    'SimpleMarkov',
    'get_baseline',
    # Node-ID decoding
    'NodeSpaceAdapter',
    'NodeMetricAccumulator',
    'load_kgram_to_id',
    'load_node_to_cluster',
]
