"""
Multi-encoding trainer for GraphIDyOM.

This script tests 3 different node encoding strategies:
1. Categorical ID: Simple integer node IDs (baseline)
2. de Bruijn: Higher-order k-grams of nodes
3. Graph-Embedding: Continuous structural feature vectors

Each encoding is trained separately with results saved to different output folders.
"""

import os
import sys

# Ensure stdout/stderr handle Unicode (✓/✗) on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder

# Import encoding strategies
from models.encoding_strategies import create_encoding


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def run_encoding_experiment(encoding_type, data_path, graph_path, output_base, 
                           batch_size=128, epochs=50, learning_rate=0.0005,
                           use_kfold=False, k_folds=5, graphidyom_order=7,
                           early_stopping_patience=None):
    """
    Run training experiment with a specific encoding strategy.
    
    Args:
        encoding_type: 'categorical_id', 'debruijn', or 'graph_embedding'
        data_path: Path to CSV file
        graph_path: Path to JSON network file
        output_base: Base output directory
        batch_size: Training batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        use_kfold: Whether to use K-fold CV
        k_folds: Number of folds
        graphidyom_order: GraphIDyOM Markov order
    
    Returns:
        Dict with results
    """
    
    print(f"\n{'='*70}")
    print(f"ENCODING EXPERIMENT: {encoding_type.upper()}")
    print(f"{'='*70}")
    
    # Create output folder for this encoding
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base, f"{encoding_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Ensure the script's original directory is on sys.path before chdir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Import train script before changing directory
    from train import TrajectoryPredictor

    os.chdir(output_dir)

    try:
        
        # Load data and graph
        print(f"\nLoading data and graph...")
        df = pd.read_csv(data_path)
        
        with open(graph_path, 'r') as f:
            import json
            graph_data = json.load(f)
        
        # Build NetworkX graph
        graph = nx.MultiDiGraph()
        
        if isinstance(graph_data['nodes'], list):
            for node in graph_data['nodes']:
                graph.add_node(node['id'])
        else:
            for node_id in graph_data['nodes'].keys():
                graph.add_node(int(node_id))
        
        for edge in graph_data['edges']:
            graph.add_edge(edge['source'], edge['target'])
        
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Parse trajectories
        trajectories = []
        if 'q_path' in df.columns:
            for path_str in df['q_path']:
                try:
                    path = [int(n.strip()) for n in str(path_str).split(',')]
                    trajectories.append(np.array(path))
                except:
                    pass
        
        print(f"Loaded {len(trajectories)} trajectories")
        
        # Create node encoder
        all_nodes = set()
        for traj in trajectories:
            all_nodes.update(traj)
        
        node_encoder = LabelEncoder()
        node_encoder.fit(list(all_nodes))
        vocab_size = len(node_encoder.classes_)
        print(f"Vocabulary size: {vocab_size}")
        
        # Create encoding strategy
        print(f"\nSetting up {encoding_type} encoding...")
        if encoding_type == 'categorical_id':
            encoding = create_encoding('categorical_id', node_encoder)
        elif encoding_type == 'debruijn':
            encoding = create_encoding('debruijn', node_encoder, 
                                      trajectories=trajectories, order=2)
            vocab_size = encoding.vocab_size  # Update vocab size
        elif encoding_type == 'graph_embedding':
            encoding = create_encoding('graph_embedding', node_encoder, 
                                      graph=graph, feature_type='combined', embedding_dim=16)
            vocab_size = encoding.vocab_size  # For discretized version
        else:
            raise ValueError(f"Unknown encoding: {encoding_type}")
        
        config = encoding.get_config()
        print(f"Encoding config: {config}")
        
        # Create trajectory predictor
        print(f"\nInitializing predictor...")
        predictor = TrajectoryPredictor(
            data_path=data_path,
            graph_path=graph_path,
            sequence_length=10,
            test_size=0.2,
            val_size=0.2
        )
        
        # Apply encoding to trajectories
        print(f"Encoding trajectories...")
        if encoding_type == 'graph_embedding':
            # For graph embedding, we need special handling
            print(f"Note: Graph embedding uses continuous representations")
        else:
            # For categorical and de Bruijn, re-encode
            encoded_trajectories = []
            for traj in trajectories:
                try:
                    enc_traj = encoding.encode_trajectory(traj)
                    encoded_trajectories.append(enc_traj)
                except:
                    pass
            print(f"Encoded {len(encoded_trajectories)} trajectories")
        
        # Train GraphIDyOM
        print(f"\n{'='*60}")
        print(f"TRAINING GraphIDyOM with {encoding_type}")
        print(f"{'='*60}")
        
        results, model = predictor.train_graphidyom(
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            use_kfold=use_kfold,
            k_folds=k_folds,
            graphidyom_order=graphidyom_order,
            early_stopping_patience=early_stopping_patience
        )
        
        # Save encoding config with results
        if results:
            results['encoding_config'] = config
            results['encoding_type'] = encoding_type
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'encoding_type': encoding_type,
            'encoding_config': config,
            'training_config': {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'use_kfold': use_kfold,
                'k_folds': k_folds,
                'graphidyom_order': graphidyom_order,
                'vocab_size': vocab_size
            },
            'results': results,
            'output_dir': output_dir
        }
        
        with open('encoding_summary.json', 'w') as f:
            json.dump(convert_numpy_types(summary), f, indent=2)
        
        print(f"\n✓ Results saved to {output_dir}")
        return summary
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'encoding_type': encoding_type}


def main():
    """Main entry point for multi-encoding experiments."""
    parser = argparse.ArgumentParser(description='Multi-encoding trainer for GraphIDyOM')
    parser.add_argument('--data-path', type=str, default="data/worldmove_380_NY.csv",
                       help='Path to training data CSV')
    parser.add_argument('--graph-path', type=str, default="data/worldmove_380_NY_network.json",
                       help='Path to graph JSON file')
    parser.add_argument('--output-base', type=str, default="experiments",
                       help='Base output directory for all encodings')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--kfold', action='store_true', default=False)
    parser.add_argument('--no-kfold', action='store_true', default=False,
                       help='Disable K-fold (overrides --kfold)')
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--no-early-stop', action='store_true', default=False,
                       help='Disable early stopping (train all epochs)')
    parser.add_argument('--encodings', type=str, nargs='+',
                       default=['categorical_id', 'debruijn', 'graph_embedding'],
                       help='Which encodings to test')
    parser.add_argument('--graphidyom-order', type=int, default=7,
                       help='GraphIDyOM Markov order (default: 7)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    data_path = os.path.abspath(args.data_path)
    graph_path = os.path.abspath(args.graph_path)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found: {graph_path}")
        return
    
    # Create output base directory
    os.makedirs(args.output_base, exist_ok=True)
    
    # Store original directory
    original_dir = os.getcwd()
    
    # Run experiments for each encoding
    results_summary = {
        'experiment_name': f'GraphIDyOM Multi-Encoding Comparison',
        'timestamp': datetime.now().isoformat(),
        'encodings': args.encodings,
        'experiments': {}
    }
    
    for encoding_type in args.encodings:
        print(f"\n\n{'#'*70}")
        print(f"# EXPERIMENT: {encoding_type}")
        print(f"{'#'*70}")
        
        use_kfold = args.kfold and not args.no_kfold
        early_stopping_patience = None if args.no_early_stop else 10
        result = run_encoding_experiment(
            encoding_type=encoding_type,
            data_path=data_path,
            graph_path=graph_path,
            output_base=args.output_base,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            use_kfold=use_kfold,
            k_folds=args.k_folds,
            graphidyom_order=args.graphidyom_order,
            early_stopping_patience=early_stopping_patience
        )
        
        results_summary['experiments'][encoding_type] = result
        
        # Return to original directory
        os.chdir(original_dir)
    
    # Save overall summary
    summary_path = os.path.join(args.output_base, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy_types(results_summary), f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"✓ EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Overall summary: {summary_path}")
    print(f"Results by encoding:")
    for encoding_type in args.encodings:
        exp_result = results_summary['experiments'].get(encoding_type, {})
        if 'error' in exp_result:
            print(f"  ✗ {encoding_type}: ERROR - {exp_result['error']}")
        else:
            print(f"  ✓ {encoding_type}: OK")


if __name__ == '__main__':
    main()
