"""
Training script for trajectory prediction models.
Implements LSTM, Transformer, and GNN models to predict agent trajectories
in road networks using simulation data.
"""

import pandas as pd
import numpy as np
import json
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import networkx as nx
import os
import sys
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Import models from separate files
from models import GraphIDyOMPredictor
from models.graph_idyom import build_graph_structure_from_sequences

# Unused model stubs (kept for flag compatibility)
LSTMTrajectoryPredictor = None
TransformerTrajectoryPredictor = None
GNNTrajectoryPredictor = None

# =============================================================================
# GLOBAL CONFIGURATION - Set which models to train
# =============================================================================
TRAIN_LSTM = False
TRAIN_TRANSFORMER = False
TRAIN_GNN = False
TRAIN_GRAPHIDYOM = True

# K-Fold Cross-Validation Configuration
K_FOLDS = 5  # Number of folds for cross-validation
USE_KFOLD = True  # Whether to use K-Fold CV or simple train/val/test split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TrajectoryDataset(Dataset):
    """Dataset for trajectory sequences"""
    
    def __init__(self, sequences, targets, node_encoder, sequence_length=10):
        self.sequences = sequences
        self.targets = targets
        self.node_encoder = node_encoder
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sequence, target


class TrajectoryPredictor:
    """Main class for training and evaluating trajectory prediction models
    
    IMPORTANT: For GraphIDyOM training, use GPU-optimized graph structure building:
    - Use: graph_struct = build_graph_structure_from_sequences(sequences, predictor.vocab_size)
    - Or: graph_struct = predictor.build_graph_structure(sequences)  # sequences should be on GPU
    - Do NOT convert to CPU/numpy before building graph structure
    - The new implementation is fully vectorized and GPU-accelerated
    """
    
    def __init__(self, data_path, graph_path, sequence_length=10, test_size=0.2, val_size=0.2):
        self.data_path = data_path
        self.graph_path = graph_path
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.val_size = val_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.node_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Load and preprocess data
        self.load_data()
        self.load_graph()
        self.preprocess_data()
        
    def load_data(self):
        """Load trajectory data from CSV"""
        print("Loading trajectory data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} trips")
        
        # Handle different column formats
        if 'q_path' in self.df.columns:
            # Parse q_path column (comma-separated string of node IDs)
            def parse_q_path(path_str):
                if isinstance(path_str, str):
                    return [int(node.strip()) for node in path_str.split(',') if node.strip()]
                return []
            self.df['route_nodes'] = self.df['q_path'].apply(parse_q_path)
        elif 'route_taken' in self.df.columns:
            # Parse route_taken column (string representation of list)
            self.df['route_nodes'] = self.df['route_taken'].apply(ast.literal_eval)
        else:
            raise ValueError("No valid path column found. Expected 'q_path' or 'route_taken'.")
        
        # Filter out routes that are too short
        self.df = self.df[self.df['route_nodes'].apply(len) >= self.sequence_length + 1]
        print(f"After filtering short routes: {len(self.df)} trips")
        
    def load_graph(self):
        """Load network graph"""
        print("Loading network graph...")
        with open(self.graph_path, 'r') as f:
            graph_data = json.load(f)
        
        # Create NetworkX graph
        self.graph = nx.MultiDiGraph()
        
        # Add nodes - handle both list and dict formats
        if isinstance(graph_data['nodes'], list):
            # Nodes are in list format: [{'id': ..., 'x': ..., 'y': ..., ...}, ...]
            for node in graph_data['nodes']:
                node_id = node['id']
                # Add position and other attributes
                attrs = {k: v for k, v in node.items() if k != 'id'}
                if 'x' in node and 'y' in node:
                    attrs['pos'] = (node['x'], node['y'])
                self.graph.add_node(node_id, **attrs)
        else:
            # Nodes are in dict format: {'node_id': {'x': ..., 'y': ..., ...}, ...}
            for node_id, node_data in graph_data['nodes'].items():
                self.graph.add_node(int(node_id), **node_data)
        
        # Add edges - handle both 'edges' and 'links' keys
        edges_key = 'links' if 'links' in graph_data else 'edges'
        
        if edges_key in graph_data:
            for edge_data in graph_data[edges_key]:
                source = edge_data.get('source')
                target = edge_data.get('target')
                if source is not None and target is not None:
                    # Remove source/target from edge attributes to avoid duplication
                    edge_attrs = {k: v for k, v in edge_data.items() 
                                if k not in ['source', 'target']}
                    self.graph.add_edge(source, target, **edge_attrs)
        
        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def preprocess_data(self):
        """Preprocess data for training"""
        print("Preprocessing data...")
        
        # Collect all unique nodes
        all_nodes = set()
        for route in self.df['route_nodes']:
            all_nodes.update(route)
        
        # Fit node encoder
        self.node_encoder.fit(list(all_nodes))
        self.vocab_size = len(self.node_encoder.classes_)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Create sequences and targets
        sequences = []
        targets = []
        
        for route in self.df['route_nodes']:
            route_encoded = self.node_encoder.transform(route)
            
            # Create sliding window sequences
            for i in range(len(route_encoded) - self.sequence_length):
                sequence = route_encoded[i:i + self.sequence_length]
                target = route_encoded[i + self.sequence_length]
                sequences.append(sequence)
                targets.append(target)
        
        self.sequences = np.array(sequences)
        self.targets = np.array(targets)
        
        print(f"Created {len(sequences)} training sequences")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.sequences, self.targets, test_size=self.test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_size/(1-self.test_size), random_state=42
        )
        
        # Create datasets
        self.train_dataset = TrajectoryDataset(X_train, y_train, self.node_encoder, self.sequence_length)
        self.val_dataset = TrajectoryDataset(X_val, y_val, self.node_encoder, self.sequence_length)
        self.test_dataset = TrajectoryDataset(X_test, y_test, self.node_encoder, self.sequence_length)
        
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
        # Prepare graph data for GNN
        self.prepare_graph_data()
        
    def prepare_graph_data(self):
        """Prepare graph data for GNN model"""
        # Use the same node encoding as the LabelEncoder for consistency
        encoded_nodes = list(range(self.vocab_size))  # 0 to vocab_size-1
        
        # Create mapping from original node IDs to encoded indices
        original_to_encoded = {}
        for i, original_node in enumerate(self.node_encoder.classes_):
            original_to_encoded[original_node] = i
        
        edge_list = []
        for edge in self.graph.edges():
            source_encoded = original_to_encoded.get(edge[0])
            target_encoded = original_to_encoded.get(edge[1])
            if source_encoded is not None and target_encoded is not None:
                edge_list.append([source_encoded, target_encoded])
        
        if edge_list:
            self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # Create self-loops for all nodes if no edges found
            self.edge_index = torch.tensor([[i, i] for i in range(self.vocab_size)], dtype=torch.long).t().contiguous()
        
        # Add self-loops to ensure all nodes have at least one connection
        num_nodes = self.vocab_size
        self_loops = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
        self.edge_index = torch.cat([self.edge_index, self_loops], dim=1)
        
        # Remove duplicate edges
        self.edge_index = torch.unique(self.edge_index, dim=1)
        
        print(f"Graph edges for GNN: {self.edge_index.shape[1]}")
        print(f"Node vocabulary size: {self.vocab_size}")
        print(f"Edge index range: [{self.edge_index.min().item()}, {self.edge_index.max().item()}]")
    
    def build_graph_structure(self, sequences):
        """
        Build graph structure for GraphIDyOM from sequences (GPU-optimized).
        
        Args:
            sequences: torch.Tensor of shape (N, seq_len) with node indices on GPU
            
        Returns:
            dict with keys: 'adjacency', 'edge_weights', 'in_degrees', 'out_degrees'
        """
        # Ensure sequences are on the correct device
        if isinstance(sequences, np.ndarray):
            sequences = torch.tensor(sequences, dtype=torch.long, device=self.device)
        else:
            sequences = sequences.to(self.device)
        
        # Build graph structure directly on GPU (fully vectorized, no CPU conversion)
        graph_struct = build_graph_structure_from_sequences(sequences, self.vocab_size)
        
        return graph_struct
    
    def create_model(self, model_type, graphidyom_order=2):
        """Create a model instance based on type
        
        Args:
            model_type: Type of model to create
            graphidyom_order: Markov order for GraphIDyOM (default: 2)
        """
        if model_type == 'lstm':
            return LSTMTrajectoryPredictor(self.vocab_size)
        elif model_type == 'transformer':
            return TransformerTrajectoryPredictor(self.vocab_size)
        elif model_type == 'gnn':
            return GNNTrajectoryPredictor(self.vocab_size)
        elif model_type == 'graphidyom':
            return GraphIDyOMPredictor(
                vocab_size=self.vocab_size,
                max_order=graphidyom_order,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_graphidyom_markov(self, graphidyom_order, train_seqs, val_seqs=None,
                                  test_seqs=None, val_targets=None, test_targets=None,
                                  output_name='best_graphidyom_model.pth'):
        """
        Pure IDyOM training: build graph_structure from training sequences.

        No gradient descent — the population-level Markov statistics ARE the model.
        Returns in the same format as train_model() for pipeline compatibility:
            (model, [0.0], [val_loss], [val_accuracy])
        The graph_structure is saved to `output_name` and also returned inside model.
        """
        print(f"\n  [GraphIDyOM] Building LTM graph structure (order={graphidyom_order})...")

        # LTM: accumulate edge counts from all training sequences
        seqs_tensor = torch.tensor(train_seqs, dtype=torch.long, device=self.device)
        graph_structure = build_graph_structure_from_sequences(seqs_tensor, self.vocab_size)
        print(f"  [GraphIDyOM] Graph structure built: {self.vocab_size}x{self.vocab_size} edge matrix")

        # Instantiate model (no parameters — pure Markov)
        model = GraphIDyOMPredictor(vocab_size=self.vocab_size, max_order=graphidyom_order)
        model = model.to(self.device)
        model.eval()

        # --- Validation evaluation ---
        val_loss = 0.0
        val_accuracy = 0.0
        if val_seqs is not None and val_targets is not None:
            criterion = nn.NLLLoss()  # model outputs log-probs
            val_dataset = TrajectoryDataset(val_seqs, val_targets, self.node_encoder, self.sequence_length)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            correct = total = 0
            with torch.no_grad():
                for batch_seq, batch_tgt in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_tgt = batch_tgt.to(self.device)
                    outputs = model(batch_seq, graph_structure=graph_structure)
                    val_loss += criterion(outputs, batch_tgt).item()
                    predicted = outputs.argmax(dim=1)
                    correct += (predicted == batch_tgt).sum().item()
                    total += batch_tgt.size(0)
            val_loss = val_loss / max(len(val_loader), 1)
            val_accuracy = 100.0 * correct / max(total, 1)
            print(f"  [GraphIDyOM] Val accuracy: {val_accuracy:.2f}%  Val loss: {val_loss:.4f}")

        # --- Test evaluation ---
        test_accuracy = 0.0
        if test_seqs is not None and test_targets is not None:
            test_dataset = TrajectoryDataset(test_seqs, test_targets, self.node_encoder, self.sequence_length)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            correct = total = 0
            with torch.no_grad():
                for batch_seq, batch_tgt in test_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_tgt = batch_tgt.to(self.device)
                    outputs = model(batch_seq, graph_structure=graph_structure)
                    predicted = outputs.argmax(dim=1)
                    correct += (predicted == batch_tgt).sum().item()
                    total += batch_tgt.size(0)
            test_accuracy = 100.0 * correct / max(total, 1)
            print(f"  [GraphIDyOM] Test accuracy: {test_accuracy:.2f}%")

        # Save: graph_structure IS the trained model
        save_dict = {
            'edge_weights': graph_structure['edge_weights'].cpu(),
            'vocab_size': self.vocab_size,
            'max_order': graphidyom_order,
            'smoothing': model.smoothing,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
        }
        torch.save(save_dict, output_name)
        print(f"  [GraphIDyOM] Saved to {output_name}")

        # Attach graph_structure to model for downstream use
        model._pretrained_graph_structure = graph_structure

        return model, [0.0], [val_loss], [val_accuracy]

    def train_model(self, model_type='lstm', batch_size=64, epochs=50, learning_rate=0.001, graphidyom_order=2):
        """Train a specific model

        Args:
            graphidyom_order: Markov order for GraphIDyOM (default: 2)
        """
        print(f"\nTraining {model_type.upper()} model...")
        if model_type == 'graphidyom':
            print(f"  GraphIDyOM Markov order: {graphidyom_order}")
            # Pure Markov IDyOM: no gradient descent, just build graph structure
            train_seqs = self.train_dataset.sequences
            val_seqs   = self.val_dataset.sequences
            val_tgts   = self.val_dataset.targets
            test_seqs  = self.test_dataset.sequences
            test_tgts  = self.test_dataset.targets
            return self._train_graphidyom_markov(
                graphidyom_order=graphidyom_order,
                train_seqs=train_seqs,
                val_seqs=val_seqs, val_targets=val_tgts,
                test_seqs=test_seqs, test_targets=test_tgts,
                output_name='best_graphidyom_model.pth',
            )
        
        # Create model
        model = self.create_model(model_type, graphidyom_order=graphidyom_order)
        model = model.to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.01)
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10
        
        epoch_bar = tqdm(range(epochs), desc=f"Training [{model_type}]", unit="epoch", file=sys.stdout)
        for epoch in epoch_bar:
            # Training phase
            model.train()
            train_loss = 0.0
            
            batch_bar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{epochs} [train]", unit="batch", leave=False, file=sys.stdout)
            for batch_sequences, batch_targets in batch_bar:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                if model_type == 'gnn':
                    edge_index = self.edge_index.to(self.device)
                    if batch_sequences.min() < 0 or batch_sequences.max() >= self.vocab_size:
                        continue
                    if edge_index.min() < 0 or edge_index.max() >= self.vocab_size:
                        continue
                    outputs = model(batch_sequences, edge_index)
                else:
                    outputs = model(batch_sequences)
                
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_sequences, batch_targets in tqdm(val_loader, desc=f"  Ep {epoch+1}/{epochs} [val]", unit="batch", leave=False, file=sys.stdout):
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    if model_type == 'gnn':
                        edge_index = self.edge_index.to(self.device)
                        if batch_sequences.min() < 0 or batch_sequences.max() >= self.vocab_size:
                            continue
                        outputs = model(batch_sequences, edge_index)
                    else:
                        outputs = model(batch_sequences)
                    
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_targets.size(0)
                    correct += (predicted == batch_targets).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_{model_type}_model.pth')
            else:
                patience_counter += 1
            
            
            epoch_bar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", val_acc=f"{val_accuracy:.2f}%")
            
            if patience_counter >= early_stopping_patience:
                tqdm.write(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model_type}_model.pth'))
        
        # Save training metrics to JSON
        training_results = {
            'model_type': model_type,
            'config': {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'vocab_size': self.vocab_size,
                'sequence_length': self.sequence_length
            },
            'metrics': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
                'final_epoch': len(train_losses)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON file
        with open(f'{model_type}_training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print(f"Training results saved to {model_type}_training_results.json")
        
        return model, train_losses, val_losses, val_accuracies
    
    def evaluate_model(self, model, model_type='lstm'):
        """Evaluate model on test set"""
        print(f"\nEvaluating {model_type.upper()} model...")
        
        model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                if model_type == 'gnn':
                    edge_index = self.edge_index.to(self.device)
                    # Add debugging for test
                    if batch_sequences.min() < 0 or batch_sequences.max() >= self.vocab_size:
                        print(f"Warning: Test sequence indices out of range: [{batch_sequences.min()}, {batch_sequences.max()}]")
                        continue
                    outputs = model(batch_sequences, edge_index)
                else:
                    outputs = model(batch_sequences)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        
        print(f"{model_type.upper()} Test Accuracy: {accuracy:.4f}")
        
        return accuracy, all_predictions, all_targets
    
    def train_graphidyom(self, batch_size=64, epochs=50, learning_rate=0.001,
                         use_kfold=False, k_folds=5, graphidyom_order=7,
                         early_stopping_patience=10):
        """
        Wrapper used by train_multi_encoding.py.
        Returns (results_dict, model).
        """
        if use_kfold:
            fold_results, model, aggregated_metrics = self.train_model_kfold(
                model_type='graphidyom',
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                k_folds=k_folds,
                graphidyom_order=graphidyom_order,
            )
            results = {
                'fold_results': fold_results,
                'aggregated_metrics': aggregated_metrics,
            }
        else:
            model, train_losses, val_losses, val_accuracies = self.train_model(
                model_type='graphidyom',
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                graphidyom_order=graphidyom_order,
            )
            results = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'best_val_loss': min(val_losses) if val_losses else None,
                'final_epoch': len(train_losses),
            }
        return results, model

    def train_model_kfold(self, model_type='lstm', batch_size=64, epochs=50, learning_rate=0.001, k_folds=5, graphidyom_order=2):
        """
        Train a model using K-Fold cross-validation for robust evaluation.
        
        Args:
            model_type: Type of model ('lstm', 'transformer', 'gnn')
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            k_folds: Number of folds for cross-validation
            graphidyom_order: Markov order for GraphIDyOM (default: 2)
        
        Returns:
            fold_results: List of results for each fold
            best_model: Best performing model across all folds
            aggregated_metrics: Aggregated metrics across all folds
        """
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} with {k_folds}-Fold Cross-Validation")
        if model_type == 'graphidyom':
            print(f"GraphIDyOM Markov order: {graphidyom_order}")
        print(f"{'='*60}")
        
        # Get all sequences and targets
        all_sequences = self.sequences
        all_targets = self.targets

        # ---------- GraphIDyOM: pure Markov, no gradient descent ----------
        if model_type == 'graphidyom':
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_results = []
            best_fold_val_acc = 0.0
            best_model = None
            best_fold_idx = 0

            for fold, (train_val_idx, test_idx) in enumerate(kfold.split(all_sequences)):
                print(f"\n--- Fold {fold + 1}/{k_folds} ---")
                train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)
                print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

                output_name = f'best_graphidyom_model_fold{fold+1}.pth'
                model, _, [val_loss], [val_acc] = self._train_graphidyom_markov(
                    graphidyom_order=graphidyom_order,
                    train_seqs=all_sequences[train_idx],
                    val_seqs=all_sequences[val_idx], val_targets=all_targets[val_idx],
                    test_seqs=all_sequences[test_idx], test_targets=all_targets[test_idx],
                    output_name=output_name,
                )

                # test_accuracy was computed inside _train_graphidyom_markov
                save_dict = torch.load(output_name, weights_only=False)
                test_acc = save_dict.get('test_accuracy', 0.0)

                fold_result = {
                    'fold': fold + 1,
                    'train_losses': [0.0],
                    'val_losses': [val_loss],
                    'val_accuracies': [val_acc],
                    'test_accuracy': test_acc,
                    'best_val_loss': val_loss,
                    'best_val_accuracy': val_acc,
                    'num_epochs': 1,
                }
                fold_results.append(fold_result)
                print(f"  Fold {fold + 1} Test Accuracy: {test_acc:.2f}%")

                if val_acc > best_fold_val_acc:
                    best_fold_val_acc = val_acc
                    best_model = model
                    best_fold_idx = fold + 1
                    # Copy the best fold's file to the canonical name
                    import shutil
                    shutil.copy(output_name, 'best_graphidyom_model_kfold.pth')
                    print(f"  Saved new best model (fold {fold + 1}) to best_graphidyom_model_kfold.pth")

            test_accuracies = [f['test_accuracy'] for f in fold_results]
            val_accuracies_best = [f['best_val_accuracy'] for f in fold_results]
            aggregated_metrics = {
                'mean_test_accuracy': np.mean(test_accuracies),
                'std_test_accuracy': np.std(test_accuracies),
                'min_test_accuracy': np.min(test_accuracies),
                'max_test_accuracy': np.max(test_accuracies),
                'mean_val_accuracy': np.mean(val_accuracies_best),
                'std_val_accuracy': np.std(val_accuracies_best),
                'best_fold': best_fold_idx,
                'k_folds': k_folds,
            }
            print(f"\n{'='*60}")
            print(f"GraphIDyOM K-Fold Results:")
            print(f"  Test Accuracy: {aggregated_metrics['mean_test_accuracy']:.2f}% "
                  f"(±{aggregated_metrics['std_test_accuracy']:.2f}%)")
            print(f"  Val  Accuracy: {aggregated_metrics['mean_val_accuracy']:.2f}% "
                  f"(±{aggregated_metrics['std_val_accuracy']:.2f}%)")
            print(f"  Best Fold: {best_fold_idx}")
            print(f"{'='*60}\n")

            kfold_results = {
                'model_type': model_type,
                'config': {'k_folds': k_folds, 'batch_size': batch_size, 'epochs': 1,
                           'learning_rate': 0.0, 'vocab_size': self.vocab_size,
                           'graphidyom_order': graphidyom_order},
                'fold_results': fold_results,
                'aggregated_metrics': aggregated_metrics,
                'timestamp': datetime.now().isoformat(),
            }
            with open('graphidyom_kfold_results.json', 'w') as f:
                json.dump(kfold_results, f, indent=2)
            print("K-Fold results saved to graphidyom_kfold_results.json")

            return fold_results, best_model, aggregated_metrics
        # -------------------------------------------------------------------

        # Initialize K-Fold for non-graphidyom models (LSTM / GNN / Transformer)
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_results = []
        best_fold_val_acc = 0
        best_model_state = None
        best_fold_idx = 0
        
        for fold, (train_val_idx, test_idx) in enumerate(kfold.split(all_sequences)):
            print(f"\n--- Fold {fold + 1}/{k_folds} ---")
            
            # Split train_val into train and validation
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.2, random_state=42
            )
            
            # Create datasets for this fold
            train_dataset = TrajectoryDataset(
                all_sequences[train_idx], all_targets[train_idx],
                self.node_encoder, self.sequence_length
            )
            val_dataset = TrajectoryDataset(
                all_sequences[val_idx], all_targets[val_idx],
                self.node_encoder, self.sequence_length
            )
            test_dataset = TrajectoryDataset(
                all_sequences[test_idx], all_targets[test_idx],
                self.node_encoder, self.sequence_length
            )
            
            print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            # Create model
            model = self.create_model(model_type, graphidyom_order=graphidyom_order)
            model = model.to(self.device)
            
            # Build graph structure for GraphIDyOM (only once per fold)
            graph_structure = None
            if model_type == 'graphidyom':
                print(f"  Building graph structure for GraphIDyOM from training data...")
                graph_structure = self.build_graph_structure(all_sequences[train_idx])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.01)
            
            # Training history for this fold
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            best_val_loss = float('inf')
            best_fold_model_state = None
            patience_counter = 0
            early_stopping_patience = 10
            
            epoch_bar = tqdm(range(epochs), desc=f"Fold {fold+1} [{model_type}]", unit="epoch", file=sys.stdout)
            for epoch in epoch_bar:
                # Training phase
                model.train()
                train_loss = 0.0
                
                batch_bar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{epochs} [train]", unit="batch", leave=False, file=sys.stdout)
                for batch_sequences, batch_targets in batch_bar:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if model_type == 'gnn':
                        edge_index = self.edge_index.to(self.device)
                        outputs = model(batch_sequences, edge_index)
                    elif model_type == 'graphidyom':
                        outputs = model(batch_sequences, graph_structure=graph_structure)
                    else:
                        outputs = model(batch_sequences)
                    
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    batch_bar.set_postfix(loss=f"{loss.item():.4f}")
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_sequences, batch_targets in tqdm(val_loader, desc=f"  Ep {epoch+1}/{epochs} [val]", unit="batch", leave=False, file=sys.stdout):
                        batch_sequences = batch_sequences.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        if model_type == 'gnn':
                            edge_index = self.edge_index.to(self.device)
                            outputs = model(batch_sequences, edge_index)
                        elif model_type == 'graphidyom':
                            outputs = model(batch_sequences, graph_structure=graph_structure)
                        else:
                            outputs = model(batch_sequences)
                        
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_targets.size(0)
                        correct += (predicted == batch_targets).sum().item()
                
                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * correct / total
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                val_accuracies.append(val_accuracy)
                
                scheduler.step()
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_fold_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                epoch_bar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", val_acc=f"{val_accuracy:.2f}%")
                
                if patience_counter >= early_stopping_patience:
                    tqdm.write(f"  Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model for this fold and evaluate on test set
            if best_fold_model_state is None:
                print(f"  WARNING: best_fold_model_state is None for fold {fold + 1}. Using current model state.")
                best_fold_model_state = model.state_dict().copy()
            
            model.load_state_dict(best_fold_model_state)
            model.eval()
            
            test_correct = 0
            test_total = 0
            all_predictions = []
            all_targets_list = []
            
            with torch.no_grad():
                for batch_sequences, batch_targets in test_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    if model_type == 'gnn':
                        edge_index = self.edge_index.to(self.device)
                        outputs = model(batch_sequences, edge_index)
                    elif model_type == 'graphidyom':
                        outputs = model(batch_sequences, graph_structure=graph_structure)
                    else:
                        outputs = model(batch_sequences)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_targets.size(0)
                    test_correct += (predicted == batch_targets).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets_list.extend(batch_targets.cpu().numpy())
            
            test_accuracy = 100 * test_correct / test_total
            
            fold_result = {
                'fold': fold + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'test_accuracy': test_accuracy,
                'best_val_loss': best_val_loss,
                'best_val_accuracy': max(val_accuracies),
                'num_epochs': len(train_losses)
            }
            fold_results.append(fold_result)
            
            print(f"  Fold {fold + 1} Test Accuracy: {test_accuracy:.2f}%")
            
            # Track best model across all folds and save immediately
            if max(val_accuracies) > best_fold_val_acc:
                best_fold_val_acc = max(val_accuracies)
                best_model_state = best_fold_model_state
                best_fold_idx = fold + 1
                # Save best model to disk immediately (don't wait for all folds)
                model_save_path = f'best_{model_type}_model_kfold.pth'
                try:
                    torch.save(best_model_state, model_save_path)
                    # Verify file was created
                    if os.path.exists(model_save_path):
                        file_size = os.path.getsize(model_save_path)
                        print(f"  Saved new best model (fold {fold + 1}) to {model_save_path} ({file_size / (1024*1024):.2f} MB)")
                    else:
                        print(f"  ERROR: File was not created at {model_save_path}")
                except Exception as e:
                    print(f"  ERROR saving model: {e}")
        
        # Aggregate metrics across all folds
        test_accuracies = [f['test_accuracy'] for f in fold_results]
        val_accuracies_best = [f['best_val_accuracy'] for f in fold_results]
        
        aggregated_metrics = {
            'mean_test_accuracy': np.mean(test_accuracies),
            'std_test_accuracy': np.std(test_accuracies),
            'min_test_accuracy': np.min(test_accuracies),
            'max_test_accuracy': np.max(test_accuracies),
            'mean_val_accuracy': np.mean(val_accuracies_best),
            'std_val_accuracy': np.std(val_accuracies_best),
            'best_fold': best_fold_idx,
            'k_folds': k_folds
        }
        
        print(f"\n{'='*60}")
        print(f"{model_type.upper()} K-Fold Cross-Validation Results:")
        print(f"{'='*60}")
        print(f"  Test Accuracy: {aggregated_metrics['mean_test_accuracy']:.2f}% (┬▒{aggregated_metrics['std_test_accuracy']:.2f}%)")
        print(f"  Val Accuracy:  {aggregated_metrics['mean_val_accuracy']:.2f}% (┬▒{aggregated_metrics['std_val_accuracy']:.2f}%)")
        print(f"  Range: [{aggregated_metrics['min_test_accuracy']:.2f}%, {aggregated_metrics['max_test_accuracy']:.2f}%]")
        print(f"  Best Fold: {best_fold_idx}")
        
        # Load best model (already saved to disk during training)
        best_model = self.create_model(model_type)
        best_model.load_state_dict(best_model_state)
        best_model = best_model.to(self.device)
        
        # Save K-Fold results to JSON
        kfold_results = {
            'model_type': model_type,
            'config': {
                'k_folds': k_folds,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'vocab_size': self.vocab_size,
                'sequence_length': self.sequence_length
            },
            'fold_results': fold_results,
            'aggregated_metrics': aggregated_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{model_type}_kfold_results.json', 'w') as f:
            json.dump(kfold_results, f, indent=2)
        
        print(f"K-Fold results saved to {model_type}_kfold_results.json")
        
        # Verify saved files
        print(f"\n{'='*60}")
        print("Verifying saved files:")
        model_file = f'best_{model_type}_model_kfold.pth'
        results_file = f'{model_type}_kfold_results.json'
        
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024*1024)
            print(f"  Ô£ô {model_file} ({file_size:.2f} MB)")
        else:
            print(f"  Ô£ù {model_file} - NOT FOUND")
        
        if os.path.exists(results_file):
            file_size = os.path.getsize(results_file) / 1024
            print(f"  Ô£ô {results_file} ({file_size:.2f} KB)")
        else:
            print(f"  Ô£ù {results_file} - NOT FOUND")
        
        current_dir = os.getcwd()
        print(f"  Current directory: {current_dir}")
        print(f"{'='*60}\n")
        
        return fold_results, best_model, aggregated_metrics

    def predict_full_path(self, model, initial_sequence, num_steps, model_type='lstm'):
        """
        Generate a full path using autoregressive prediction.
        
        Args:
            model: Trained model
            initial_sequence: Starting sequence tensor of shape (seq_len,) or (batch_size, seq_len)
            num_steps: Number of nodes to predict
            model_type: Type of model ('lstm', 'transformer', 'gnn')
        
        Returns:
            predicted_path: List of predicted node indices
        """
        model.eval()
        
        # Ensure initial_sequence is 2D: (batch_size, seq_len)
        if initial_sequence.dim() == 1:
            current_sequence = initial_sequence.unsqueeze(0).to(self.device)
        else:
            current_sequence = initial_sequence.to(self.device)
        
        predicted_path = []
        
        with torch.no_grad():
            for _ in range(num_steps):
                # Get model prediction
                if model_type == 'gnn':
                    edge_index = self.edge_index.to(self.device)
                    output = model(current_sequence, edge_index)
                else:
                    output = model(current_sequence)
                
                # Get predicted next node (greedy decoding)
                next_node = torch.argmax(output, dim=1)
                predicted_path.append(next_node.item())
                
                # Shift sequence: remove first element, append prediction
                # current_sequence shape: (1, seq_len)
                current_sequence = torch.cat([
                    current_sequence[:, 1:],  # Remove first node
                    next_node.unsqueeze(1)    # Append predicted node
                ], dim=1)
        
        return predicted_path
    
    def predict_full_path_beam_search(self, model, initial_sequence, num_steps, model_type='lstm', beam_width=3):
        """
        Generate a full path using beam search for better quality predictions.
        
        Args:
            model: Trained model
            initial_sequence: Starting sequence tensor of shape (seq_len,)
            num_steps: Number of nodes to predict
            model_type: Type of model ('lstm', 'transformer', 'gnn')
            beam_width: Number of beams to keep
        
        Returns:
            best_path: List of predicted node indices (best beam)
            all_beams: List of (path, score) tuples for all beams
        """
        model.eval()
        
        if initial_sequence.dim() == 1:
            initial_sequence = initial_sequence.unsqueeze(0)
        initial_sequence = initial_sequence.to(self.device)
        
        # Initialize beams: list of (sequence, path, cumulative_log_prob)
        beams = [(initial_sequence, [], 0.0)]
        
        with torch.no_grad():
            for step in range(num_steps):
                all_candidates = []
                
                for seq, path, score in beams:
                    # Get model prediction
                    if model_type == 'gnn':
                        edge_index = self.edge_index.to(self.device)
                        output = model(seq, edge_index)
                    else:
                        output = model(seq)
                    
                    # Get log probabilities
                    log_probs = torch.log_softmax(output, dim=1)
                    
                    # Get top-k predictions
                    top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=1)
                    
                    for i in range(beam_width):
                        next_node = top_k_indices[0, i].item()
                        next_log_prob = top_k_log_probs[0, i].item()
                        
                        # Create new sequence
                        new_seq = torch.cat([
                            seq[:, 1:],
                            torch.tensor([[next_node]], device=self.device)
                        ], dim=1)
                        
                        new_path = path + [next_node]
                        new_score = score + next_log_prob
                        
                        all_candidates.append((new_seq, new_path, new_score))
                
                # Select top beams
                all_candidates.sort(key=lambda x: x[2], reverse=True)
                beams = all_candidates[:beam_width]
        
        # Return best path and all beams
        best_path = beams[0][1]
        all_beams = [(path, score) for _, path, score in beams]
        
        return best_path, all_beams
    
    def evaluate_full_path(self, model, model_type='lstm', num_samples=100, use_beam_search=False, beam_width=3):
        """
        Evaluate model on full path prediction using autoregressive generation.
        
        Args:
            model: Trained model
            model_type: Type of model
            num_samples: Number of test routes to evaluate
            use_beam_search: Whether to use beam search instead of greedy decoding
            beam_width: Beam width for beam search
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {model_type.upper()} model for full path prediction...")
        
        model.eval()
        
        # Get test routes (full routes, not just sequences)
        test_routes = []
        for route in self.df['route_nodes'].values:
            if len(route) >= self.sequence_length + 2:  # Need at least seq_len + some nodes to predict
                route_encoded = self.node_encoder.transform(route)
                test_routes.append(route_encoded)
        
        # Limit samples
        if len(test_routes) > num_samples:
            np.random.seed(42)
            indices = np.random.choice(len(test_routes), num_samples, replace=False)
            test_routes = [test_routes[i] for i in indices]
        
        print(f"Evaluating on {len(test_routes)} routes...")
        
        # Metrics
        all_node_accuracies = []
        all_path_lengths = []
        all_edit_distances = []
        exact_matches = 0
        top_k_accuracies = {1: [], 3: [], 5: []}
        
        for route in test_routes:
            # Use first seq_len nodes as initial sequence
            initial_seq = torch.tensor(route[:self.sequence_length], dtype=torch.long)
            
            # Ground truth: remaining nodes after initial sequence
            ground_truth = route[self.sequence_length:]
            num_steps = len(ground_truth)
            
            if num_steps == 0:
                continue
            
            # Predict full path
            if use_beam_search:
                predicted_path, _ = self.predict_full_path_beam_search(
                    model, initial_seq, num_steps, model_type, beam_width
                )
            else:
                predicted_path = self.predict_full_path(
                    model, initial_seq, num_steps, model_type
                )
            
            # Calculate node-level accuracy
            correct_nodes = sum(1 for p, g in zip(predicted_path, ground_truth) if p == g)
            node_accuracy = correct_nodes / len(ground_truth)
            all_node_accuracies.append(node_accuracy)
            all_path_lengths.append(len(ground_truth))
            
            # Calculate edit distance (Levenshtein distance)
            edit_dist = self._levenshtein_distance(predicted_path, list(ground_truth))
            all_edit_distances.append(edit_dist / max(len(predicted_path), len(ground_truth)))
            
            # Check for exact match
            if predicted_path == list(ground_truth):
                exact_matches += 1
            
            # Calculate top-k accuracy for first prediction
            with torch.no_grad():
                initial_seq_batch = initial_seq.unsqueeze(0).to(self.device)
                if model_type == 'gnn':
                    edge_index = self.edge_index.to(self.device)
                    output = model(initial_seq_batch, edge_index)
                else:
                    output = model(initial_seq_batch)
                
                for k in [1, 3, 5]:
                    _, top_k_pred = torch.topk(output, k, dim=1)
                    top_k_pred = top_k_pred.cpu().numpy().flatten()
                    if ground_truth[0] in top_k_pred:
                        top_k_accuracies[k].append(1)
                    else:
                        top_k_accuracies[k].append(0)
        
        # Aggregate metrics
        results = {
            'mean_node_accuracy': np.mean(all_node_accuracies),
            'std_node_accuracy': np.std(all_node_accuracies),
            'exact_match_rate': exact_matches / len(test_routes) if test_routes else 0,
            'mean_normalized_edit_distance': np.mean(all_edit_distances),
            'mean_path_length': np.mean(all_path_lengths),
            'top_1_accuracy': np.mean(top_k_accuracies[1]),
            'top_3_accuracy': np.mean(top_k_accuracies[3]),
            'top_5_accuracy': np.mean(top_k_accuracies[5]),
            'num_samples': len(test_routes)
        }
        
        print(f"\n{model_type.upper()} Full Path Prediction Results:")
        print(f"  Mean Node Accuracy: {results['mean_node_accuracy']:.4f} (┬▒{results['std_node_accuracy']:.4f})")
        print(f"  Exact Match Rate: {results['exact_match_rate']:.4f}")
        print(f"  Mean Normalized Edit Distance: {results['mean_normalized_edit_distance']:.4f}")
        print(f"  Top-1 Accuracy (first step): {results['top_1_accuracy']:.4f}")
        print(f"  Top-3 Accuracy (first step): {results['top_3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy (first step): {results['top_5_accuracy']:.4f}")
        
        return results
    
    def _levenshtein_distance(self, seq1, seq2):
        """Calculate Levenshtein (edit) distance between two sequences."""
        m, n = len(seq1), len(seq2)
        
        # Create distance matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                       dp[i][j-1],      # insertion
                                       dp[i-1][j-1])    # substitution
        
        return dp[m][n]
    
    def generate_path_from_origin_destination(self, model, origin, destination, model_type='lstm', max_steps=50):
        """
        Generate a full path from origin to destination using the model.
        
        Args:
            model: Trained model
            origin: Starting node (will be used to create initial sequence)
            destination: Target destination node
            model_type: Type of model
            max_steps: Maximum number of steps to predict
        
        Returns:
            predicted_path: Full predicted path including origin
            reached_destination: Whether the destination was reached
        """
        model.eval()
        
        # Encode origin and destination
        origin_encoded = self.node_encoder.transform([origin])[0]
        destination_encoded = self.node_encoder.transform([destination])[0]
        
        # Initialize sequence with origin repeated (or use padding)
        # This is a simplified approach - in practice you might want a proper start token
        initial_sequence = torch.full((self.sequence_length,), origin_encoded, dtype=torch.long)
        current_sequence = initial_sequence.unsqueeze(0).to(self.device)
        
        predicted_path = [origin_encoded]
        reached_destination = False
        
        with torch.no_grad():
            for step in range(max_steps):
                # Get model prediction
                if model_type == 'gnn':
                    edge_index = self.edge_index.to(self.device)
                    output = model(current_sequence, edge_index)
                else:
                    output = model(current_sequence)
                
                # Get predicted next node
                next_node = torch.argmax(output, dim=1).item()
                predicted_path.append(next_node)
                
                # Check if destination reached
                if next_node == destination_encoded:
                    reached_destination = True
                    break
                
                # Check for loops (if we've visited this node before recently)
                if next_node in predicted_path[-10:-1]:  # Avoid recent loops
                    # Try second best prediction
                    _, top_2 = torch.topk(output, 2, dim=1)
                    next_node = top_2[0, 1].item()
                    predicted_path[-1] = next_node
                    if next_node == destination_encoded:
                        reached_destination = True
                        break
                
                # Update sequence
                current_sequence = torch.cat([
                    current_sequence[:, 1:],
                    torch.tensor([[next_node]], device=self.device)
                ], dim=1)
        
        # Decode path back to original node IDs
        decoded_path = self.node_encoder.inverse_transform(predicted_path)
        
        return list(decoded_path), reached_destination
    
    def compare_models(self, train_lstm=True, train_transformer=True, train_gnn=True, 
                       train_graphidyom=True, use_kfold=False, k_folds=5, graphidyom_order=7):
        """Train and compare selected models, optionally using K-Fold cross-validation
        
        Args:
            graphidyom_order: Markov order for GraphIDyOM model (default: 2)
        """
        results = {}
        models = {}
        
        # Model configurations
        model_configs = {
            'lstm': {'batch_size': 64, 'epochs': 50, 'learning_rate': 0.001, 'enabled': train_lstm},
            'transformer': {'batch_size': 32, 'epochs': 50, 'learning_rate': 0.0001, 'enabled': train_transformer},
            'gnn': {'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001, 'enabled': train_gnn},
            'graphidyom': {'batch_size': 128, 'epochs': 50, 'learning_rate': 0.0005, 'enabled': train_graphidyom}
        }
        
        for model_type, config in model_configs.items():
            if not config['enabled']:
                print(f"\n{'='*50}")
                print(f"Skipping {model_type.upper()} Model (disabled)")
                print(f"{'='*50}")
                continue
                
            print(f"\n{'='*50}")
            print(f"Training {model_type.upper()} Model")
            if use_kfold:
                print(f"Using {k_folds}-Fold Cross-Validation")
            print(f"{'='*50}")
            
            # Remove 'enabled' from config before passing to train_model
            train_config = {k: v for k, v in config.items() if k != 'enabled'}
            
            if use_kfold:
                # Train with K-Fold cross-validation
                fold_results, model, aggregated_metrics = self.train_model_kfold(
                    model_type=model_type, k_folds=k_folds, graphidyom_order=graphidyom_order, **train_config
                )
                
                # Evaluate full path prediction using best model
                full_path_results = self.evaluate_full_path(model, model_type, num_samples=100)
                
                # Store results
                results[model_type] = {
                    'accuracy': aggregated_metrics['mean_test_accuracy'] / 100,  # Convert to 0-1 scale
                    'kfold_results': fold_results,
                    'aggregated_metrics': aggregated_metrics,
                    'config': train_config,
                    'full_path_results': full_path_results,
                    'use_kfold': True
                }
            else:
                # Train model with simple train/val/test split
                model, train_losses, val_losses, val_accuracies = self.train_model(
                    model_type=model_type, graphidyom_order=graphidyom_order, **train_config
                )
                
                # Evaluate model (next-step prediction)
                accuracy, predictions, targets = self.evaluate_model(model, model_type)
                
                # Evaluate full path prediction (autoregressive)
                full_path_results = self.evaluate_full_path(model, model_type, num_samples=100)
                
                # Store results
                results[model_type] = {
                    'accuracy': accuracy,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'predictions': predictions,
                    'targets': targets,
                    'config': train_config,
                    'full_path_results': full_path_results,
                    'use_kfold': False
                }
            
            models[model_type] = model
        
        # Save comparison results to JSON - Load existing file if it exists and merge
        json_file_path = 'model_comparison_results.json'
        
        # Load existing results if the file exists
        if os.path.exists(json_file_path):
            print(f"\nLoading existing results from {json_file_path}...")
            with open(json_file_path, 'r') as f:
                comparison_results = json.load(f)
            print(f"Existing models found: {list(comparison_results.get('models', {}).keys())}")
        else:
            # Create new results structure
            comparison_results = {
                'timestamp': datetime.now().isoformat(),
                'experiment_info': {
                    'sequence_length': self.sequence_length,
                    'vocab_size': self.vocab_size,
                    'test_size': self.test_size,
                    'val_size': self.val_size,
                    'use_kfold': use_kfold,
                    'k_folds': k_folds if use_kfold else None,
                    'models_trained': {
                        'lstm': train_lstm,
                        'transformer': train_transformer,
                        'gnn': train_gnn,
                        'graphidyom': train_graphidyom
                    }
                },
                'models': {}
            }
        
        # Update timestamp to reflect the latest training
        comparison_results['timestamp'] = datetime.now().isoformat()
        
        # Update experiment info with new training flags
        if 'experiment_info' in comparison_results:
            comparison_results['experiment_info']['models_trained'] = {
                'lstm': train_lstm,
                'transformer': train_transformer,
                'gnn': train_gnn,
                'graphidyom': train_graphidyom
            }
        
        # Add/update results for trained models
        for model_type, result in results.items():
            if result.get('use_kfold', False):
                # K-Fold results
                comparison_results['models'][model_type] = {
                    'test_accuracy': result['accuracy'],
                    'config': result['config'],
                    'kfold_metrics': result['aggregated_metrics'],
                    'fold_test_accuracies': [f['test_accuracy'] for f in result['kfold_results']],
                    'full_path_metrics': result['full_path_results']
                }
            else:
                # Standard training results
                comparison_results['models'][model_type] = {
                    'test_accuracy': result['accuracy'],
                    'config': result['config'],
                    'train_losses': result.get('train_losses', []),
                    'val_losses': result.get('val_losses', []),
                    'val_accuracies': result.get('val_accuracies', []),
                    'final_train_loss': result['train_losses'][-1] if result.get('train_losses') else None,
                    'final_val_loss': result['val_losses'][-1] if result.get('val_losses') else None,
                    'best_val_accuracy': max(result['val_accuracies']) if result.get('val_accuracies') else None,
                    'full_path_metrics': result['full_path_results']
                }
        
        # Save to JSON file
        with open(json_file_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\nComparison results saved to {json_file_path}")
        print(f"Models in results file: {list(comparison_results['models'].keys())}")
        
        return results, models


def main():
    """Main training function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train trajectory prediction models')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer', 'gnn', 'graphidyom', 'all'], 
                       default='all', help='Model type to train (default: all)')
    parser.add_argument('--data-path', type=str, default="data/subnet_ci.csv",
                       help='Path to training data CSV file')
    parser.add_argument('--graph-path', type=str, default="data/subnet_NY.json",
                       help='Path to graph JSON file')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Sequence length for trajectory prediction (default: 10)')
    parser.add_argument('--output-dir', type=str, default="model_outputs",
                       help='Output directory for models and results (default: model_outputs)')
    parser.add_argument('--kfold', action='store_true', default=None,
                       help='Use K-Fold cross-validation (overrides USE_KFOLD config)')
    parser.add_argument('--no-kfold', action='store_true', default=None,
                       help='Disable K-Fold cross-validation')
    parser.add_argument('--k-folds', type=int, default=None,
                       help=f'Number of folds for K-Fold CV (default: {K_FOLDS})')
    parser.add_argument('--graphidyom-order', type=int, default=7,
                       help='Markov order for GraphIDyOM model (default: 7)')
    
    args = parser.parse_args()
    
    # Determine K-Fold settings (command line args override global config)
    use_kfold = USE_KFOLD
    if args.kfold:
        use_kfold = True
    elif args.no_kfold:
        use_kfold = False
    k_folds = args.k_folds if args.k_folds is not None else K_FOLDS
    
    # Paths - convert to absolute paths BEFORE changing directory
    data_path = os.path.abspath(args.data_path)
    graph_path = os.path.abspath(args.graph_path)
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    # Initialize predictor with absolute paths
    predictor = TrajectoryPredictor(
        data_path=data_path,
        graph_path=graph_path,
        sequence_length=args.sequence_length,
        test_size=0.2,
        val_size=0.2
    )
    
    if args.model == 'all':
        # Train and compare models based on global flags
        print("\n" + "="*60)
        print("MODEL TRAINING CONFIGURATION")
        print("="*60)
        print(f"  TRAIN_LSTM: {TRAIN_LSTM}")
        print(f"  TRAIN_TRANSFORMER: {TRAIN_TRANSFORMER}")
        print(f"  TRAIN_GNN: {TRAIN_GNN}")
        print(f"  TRAIN_GRAPHIDYOM: {TRAIN_GRAPHIDYOM}")
        print(f"  USE_KFOLD: {use_kfold}")
        if use_kfold:
            print(f"  K_FOLDS: {k_folds}")
        
        results, models = predictor.compare_models(
            train_lstm=TRAIN_LSTM,
            train_transformer=TRAIN_TRANSFORMER,
            train_gnn=TRAIN_GNN,
            train_graphidyom=TRAIN_GRAPHIDYOM,
            use_kfold=use_kfold,
            k_folds=k_folds,
            graphidyom_order=args.graphidyom_order
        )
        
        if results:
            # Print final results
            print("\n" + "="*60)
            if use_kfold:
                print(f"FINAL RESULTS - {k_folds}-FOLD CROSS-VALIDATION")
            else:
                print("FINAL RESULTS - NEXT-STEP PREDICTION")
            print("="*60)
            
            for model_type, result in results.items():
                if use_kfold:
                    metrics = result['aggregated_metrics']
                    print(f"{model_type.upper()}: Test Accuracy = {metrics['mean_test_accuracy']:.2f}% (┬▒{metrics['std_test_accuracy']:.2f}%)")
                else:
                    print(f"{model_type.upper()}: Test Accuracy = {result['accuracy']:.4f}")
            
            # Print full path results
            print("\n" + "="*60)
            print("FINAL RESULTS - FULL PATH PREDICTION (AUTOREGRESSIVE)")
            print("="*60)
            
            for model_type, result in results.items():
                fp = result['full_path_results']
                print(f"\n{model_type.upper()}:")
                print(f"  Node Accuracy: {fp['mean_node_accuracy']:.4f} (┬▒{fp['std_node_accuracy']:.4f})")
                print(f"  Exact Match Rate: {fp['exact_match_rate']:.4f}")
                print(f"  Normalized Edit Distance: {fp['mean_normalized_edit_distance']:.4f}")
            
            # Find best model (by full path node accuracy)
            best_model = max(results.keys(), key=lambda x: results[x]['full_path_results']['mean_node_accuracy'])
            print(f"\nBest Model (Full Path): {best_model.upper()} with node accuracy {results[best_model]['full_path_results']['mean_node_accuracy']:.4f}")
        else:
            print("\nNo models were trained. Enable at least one model in the global configuration.")
        
    else:
        # Train single model
        print(f"\n{'='*50}")
        print(f"Training {args.model.upper()} Model")
        if use_kfold:
            print(f"Using {k_folds}-Fold Cross-Validation")
        print(f"{'='*50}")
        
        # Use model-specific configurations or command line args
        model_configs = {
            'lstm': {'batch_size': args.batch_size, 'epochs': args.epochs, 'learning_rate': args.learning_rate},
            'transformer': {'batch_size': min(args.batch_size, 32), 'epochs': args.epochs, 'learning_rate': max(args.learning_rate * 0.1, 0.0001)},
            'gnn': {'batch_size': min(args.batch_size, 32), 'epochs': args.epochs, 'learning_rate': args.learning_rate},
            'graphidyom': {'batch_size': args.batch_size, 'epochs': args.epochs, 'learning_rate': args.learning_rate}
        }
        
        config = model_configs.get(args.model, model_configs['lstm'])
        
        if use_kfold:
            # Train with K-Fold cross-validation
            fold_results, model, aggregated_metrics = predictor.train_model_kfold(
                model_type=args.model, k_folds=k_folds, graphidyom_order=args.graphidyom_order, **config
            )
            
            # Evaluate full path prediction (autoregressive)
            full_path_results = predictor.evaluate_full_path(model, args.model, num_samples=100)
            
            print(f"\n{args.model.upper()} Final Results ({k_folds}-Fold CV):")
            print(f"\n--- Next-Step Prediction ---")
            print(f"Test Accuracy: {aggregated_metrics['mean_test_accuracy']:.2f}% (┬▒{aggregated_metrics['std_test_accuracy']:.2f}%)")
            print(f"Val Accuracy:  {aggregated_metrics['mean_val_accuracy']:.2f}% (┬▒{aggregated_metrics['std_val_accuracy']:.2f}%)")
            print(f"Min/Max Test:  [{aggregated_metrics['min_test_accuracy']:.2f}%, {aggregated_metrics['max_test_accuracy']:.2f}%]")
            print(f"\n--- Full Path Prediction (Autoregressive) ---")
            print(f"Node Accuracy: {full_path_results['mean_node_accuracy']:.4f} (┬▒{full_path_results['std_node_accuracy']:.4f})")
            print(f"Exact Match Rate: {full_path_results['exact_match_rate']:.4f}")
            print(f"Normalized Edit Distance: {full_path_results['mean_normalized_edit_distance']:.4f}")
            print(f"Top-1/3/5 First Step Acc: {full_path_results['top_1_accuracy']:.4f} / {full_path_results['top_3_accuracy']:.4f} / {full_path_results['top_5_accuracy']:.4f}")
        else:
            # Train model with simple split
            model, train_losses, val_losses, val_accuracies = predictor.train_model(
                model_type=args.model, graphidyom_order=args.graphidyom_order, **config
            )
            
            # Evaluate model (next-step)
            accuracy, predictions, targets = predictor.evaluate_model(model, args.model)
            
            # Evaluate full path prediction (autoregressive)
            full_path_results = predictor.evaluate_full_path(model, args.model, num_samples=100)
            
            print(f"\n{args.model.upper()} Final Results:")
            print(f"\n--- Next-Step Prediction ---")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Best Validation Accuracy: {max(val_accuracies):.4f}")
            print(f"Final Training Loss: {train_losses[-1]:.4f}")
            print(f"Final Validation Loss: {val_losses[-1]:.4f}")
            print(f"\n--- Full Path Prediction (Autoregressive) ---")
            print(f"Node Accuracy: {full_path_results['mean_node_accuracy']:.4f} (┬▒{full_path_results['std_node_accuracy']:.4f})")
            print(f"Exact Match Rate: {full_path_results['exact_match_rate']:.4f}")
            print(f"Normalized Edit Distance: {full_path_results['mean_normalized_edit_distance']:.4f}")
            print(f"Top-1/3/5 First Step Acc: {full_path_results['top_1_accuracy']:.4f} / {full_path_results['top_3_accuracy']:.4f} / {full_path_results['top_5_accuracy']:.4f}")
    
    print(f"\nAll outputs saved in '{args.output_dir}' directory")
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate plots using the separate plotting module
    if args.model == 'all' and results:
        print("\nGenerating comparison plots...")
        try:
            from plot_results import ModelResultsPlotter
            plotter = ModelResultsPlotter(".")  # Current directory (model_outputs)
            if plotter.load_results():
                plotter.print_summary()
                plotter.plot_model_comparison(show_plot=False)  # Save plots without showing
                print("Plots generated successfully!")
        except ImportError:
            print("plot_results.py not found. Please run plotting manually.")
        except Exception as e:
            print(f"Error generating plots: {e}")
            print("You can generate plots manually by running: python plot_results.py")
    else:
        print(f"\nTo visualize {args.model} training results, you can use:")
        print(f"python plot_single_model.py --model {args.model}")
    
    # Generate path visualizations (predicted vs actual)
    print("\n" + "="*60)
    print("GENERATING PATH VISUALIZATIONS")
    print("="*60)
    try:
        # Go back to main directory for visualization
        os.chdir("..")
        from visualize_paths import PathVisualizer
        
        visualizer = PathVisualizer(
            data_path=data_path,
            graph_path=graph_path,
            model_dir=args.output_dir
        )
        
        if args.model == 'all':
            # Visualize all trained models
            for model_type in ['lstm', 'transformer', 'gnn']:
                if (model_type == 'lstm' and TRAIN_LSTM) or \
                   (model_type == 'transformer' and TRAIN_TRANSFORMER) or \
                   (model_type == 'gnn' and TRAIN_GNN):
                    visualizer.visualize_comparisons(model_type, num_examples=10)
        else:
            visualizer.visualize_comparisons(args.model, num_examples=10)
        
        print("\nPath visualizations saved to model_outputs directory!")
    except Exception as e:
        print(f"Error generating path visualizations: {e}")
        print("You can generate visualizations manually by running:")
        print("  python visualize_paths.py --model all")


if __name__ == "__main__":
    main()
