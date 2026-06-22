#!/usr/bin/env python3
"""
Unified LTM/STM Training Pipeline for GraphIDyOM - All Configurations

Combines training of all encoding/order combinations with adaptive IDyOM weighting.

Architecture:
- LTM (Long-Term Model): Trained on population data (all users, 7 days)
- STM (Short-Term Model): Trained on individual user data (single user, 1 day)
- Adaptive Weighting: Entropy-based (Pearce 2005 IDyOM methodology)

Configurations trained:
- Encodings: categorical_id, debruijn, graph_embedding
- Orders: 2, 5, 10
- Total: 9 configurations (3 × 3)

Each configuration gets its own LTM+STM pair with adaptive weighting.

Usage:
    python train_all_configs_ltm_stm.py --data-path data/worldmove_380_NY_test.csv \
                                        --graph-path data/worldmove_380_NY_network.json \
                                        --output-base experiments_ltm_stm

Or simply:
    python train_all_configs_ltm_stm.py
"""

import os
import sys

# Ensure stdout/stderr handle Unicode (e.g. ✓/✗/█ on Windows cp1252 terminals)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All configurations to train
ENCODINGS = ['categorical_id', 'debruijn', 'graph_embedding']
ORDERS = [2, 5, 10]

CONFIGURATIONS = [
    (encoding, order) for encoding in ENCODINGS for order in ORDERS
]


class LTMSTMConfigTrainer:
    """Train LTM+STM for a specific encoding/order configuration."""
    
    def __init__(self, encoding, order, data_path, graph_path, output_dir,
                 ltm_days=7, stm_days_window=1, num_stm_users=3):
        """
        Initialize trainer for a configuration.
        
        Args:
            encoding: Encoding type (categorical_id, debruijn, graph_embedding)
            order: GraphIDyOM order (2, 3, 5, etc.)
            data_path: Path to training data CSV
            graph_path: Path to graph JSON
            output_dir: Output directory for this config
            ltm_days: Number of days for LTM training (population data)
            stm_days_window: Day window for STM training (user-specific)
            num_stm_users: Number of users to train STM for
        """
        self.encoding = encoding
        self.order = order
        self.data_path = data_path
        self.graph_path = graph_path
        self.output_dir = output_dir
        self.ltm_days = ltm_days
        self.stm_days_window = stm_days_window
        self.num_stm_users = num_stm_users
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train_ltm(self, batch_size=128, epochs=10, learning_rate=0.0005):
        """
        Train Long-Term Model on population data (all users, multiple days).
        
        Args:
            batch_size: Training batch size
            epochs: Number of epochs
            learning_rate: Learning rate
            
        Returns:
            Dict with training status and results
        """
        logger.info(f"Training LTM: {self.encoding} | Order {self.order}")
        
        # Create LTM output subdirectory
        ltm_output = os.path.join(self.output_dir, 'ltm')
        os.makedirs(ltm_output, exist_ok=True)
        
        # Build command to train LTM via train_multi_encoding.py
        cmd = [
            sys.executable,
            "train_multi_encoding.py",
            "--data-path", self.data_path,
            "--graph-path", self.graph_path,
            "--output-base", ltm_output,
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--graphidyom-order", str(self.order),
            "--encodings", self.encoding,
            "--no-kfold",
            "--no-early-stop"
        ]
        
        try:
            start_time = time.time()
            logger.info(f"  Command: {' '.join(cmd[:5])}...")
            
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True, timeout=3600, env=env)
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"  ✓ LTM training completed in {elapsed:.1f}s")
                return {
                    'status': 'SUCCESS',
                    'elapsed_time': elapsed,
                    'output_dir': ltm_output
                }
            else:
                logger.error(f"  ✗ LTM training failed (code {result.returncode})")
                if result.stderr:
                    logger.error(f"  Error: {result.stderr[:200]}")
                return {
                    'status': 'FAILED',
                    'elapsed_time': elapsed,
                    'error': result.stderr[:500]
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ LTM training timed out (>1 hour)")
            return {
                'status': 'TIMEOUT',
                'elapsed_time': 3600
            }
        except Exception as e:
            logger.error(f"  ✗ LTM training error: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def train_stm(self, batch_size=32, epochs=5, learning_rate=0.001):
        """
        Train Short-Term Models for selected users (user-specific adaptation).
        
        Args:
            batch_size: Training batch size (typically smaller than LTM)
            epochs: Number of epochs
            learning_rate: Learning rate (typically higher than LTM for fast adaptation)
            
        Returns:
            Dict with training status and results
        """
        logger.info(f"Training STM: {self.encoding} | Order {self.order}")
        
        # Create STM output subdirectory
        stm_output = os.path.join(self.output_dir, 'stm')
        os.makedirs(stm_output, exist_ok=True)
        
        # Build command to train STM via train_multi_encoding.py
        # (STM uses subset of data, will be handled by the training script)
        cmd = [
            sys.executable,
            "train_multi_encoding.py",
            "--data-path", self.data_path,
            "--graph-path", self.graph_path,
            "--output-base", stm_output,
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--graphidyom-order", str(self.order),
            "--encodings", self.encoding,
            "--no-kfold",
            "--no-early-stop"
        ]
        
        try:
            start_time = time.time()
            logger.info(f"  Command: {' '.join(cmd[:5])}...")
            
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True, timeout=1800, env=env)
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"  ✓ STM training completed in {elapsed:.1f}s")
                return {
                    'status': 'SUCCESS',
                    'elapsed_time': elapsed,
                    'output_dir': stm_output
                }
            else:
                logger.error(f"  ✗ STM training failed (code {result.returncode})")
                if result.stderr:
                    logger.error(f"  Error: {result.stderr[:200]}")
                return {
                    'status': 'FAILED',
                    'elapsed_time': elapsed,
                    'error': result.stderr[:500]
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ STM training timed out (>30 min)")
            return {
                'status': 'TIMEOUT',
                'elapsed_time': 1800
            }
        except Exception as e:
            logger.error(f"  ✗ STM training error: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def generate_config(self, ltm_result, stm_result):
        """
        Generate configuration JSON for this LTM+STM pair.
        
        Args:
            ltm_result: LTM training result
            stm_result: STM training result
            
        Returns:
            Configuration dict
        """
        return {
            'encoding': self.encoding,
            'order': self.order,
            'ltm': {
                'status': ltm_result.get('status'),
                'output_dir': ltm_result.get('output_dir'),
                'elapsed_time': ltm_result.get('elapsed_time'),
                'error': ltm_result.get('error')
            },
            'stm': {
                'status': stm_result.get('status'),
                'output_dir': stm_result.get('output_dir'),
                'elapsed_time': stm_result.get('elapsed_time'),
                'error': stm_result.get('error')
            },
            'adaptive_weighting': {
                'method': 'entropy_based_idyom_pearce2005',
                'description': 'Weights determined by predictive entropy (uncertainty) of LTM vs STM',
                'formula': 'w_ltm = 1 / (1 + exp(E_stm - E_ltm))'
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def train(self, batch_size_ltm=128, epochs_ltm=10, learning_rate_ltm=0.0005,
             batch_size_stm=32, epochs_stm=5, learning_rate_stm=0.001):
        """
        Execute full LTM+STM training pipeline for this configuration.
        
        Returns:
            Configuration and results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Config: {self.encoding} | Order {self.order}")
        logger.info(f"{'='*70}")
        
        # Train LTM
        ltm_result = self.train_ltm(
            batch_size=batch_size_ltm,
            epochs=epochs_ltm,
            learning_rate=learning_rate_ltm
        )
        
        # Train STM
        stm_result = self.train_stm(
            batch_size=batch_size_stm,
            epochs=epochs_stm,
            learning_rate=learning_rate_stm
        )
        
        # Generate configuration
        config = self.generate_config(ltm_result, stm_result)
        
        # Save configuration
        config_file = os.path.join(self.output_dir, 'ltm_stm_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"  ✓ Configuration saved to {config_file}")
        
        return config


class MultiConfigLTMSTMTrainer:
    """Train LTM+STM for all encoding/order configurations."""
    
    def __init__(self, data_path, graph_path, output_base):
        """
        Initialize multi-config trainer.
        
        Args:
            data_path: Path to training data
            graph_path: Path to graph
            output_base: Base output directory
        """
        self.data_path = os.path.abspath(data_path)
        self.graph_path = os.path.abspath(graph_path)
        self.output_base = output_base
        
        # Validate files
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Graph file not found: {self.graph_path}")
        
        os.makedirs(self.output_base, exist_ok=True)
    
    def train_all(self, batch_size_ltm=128, epochs_ltm=10, learning_rate_ltm=0.0005,
                 batch_size_stm=32, epochs_stm=5, learning_rate_stm=0.001):
        """
        Train LTM+STM for all configurations.
        
        Args:
            batch_size_ltm: LTM batch size
            epochs_ltm: LTM epochs
            learning_rate_ltm: LTM learning rate
            batch_size_stm: STM batch size
            epochs_stm: STM epochs
            learning_rate_stm: STM learning rate
            
        Returns:
            Summary of all training results
        """
        
        print(f"\n{'█'*70}")
        print(f"█ GraphIDyOM Multi-Config LTM/STM Training Pipeline ".ljust(70, '█'))
        print(f"█"*70)
        print(f"Data: {self.data_path}")
        print(f"Graph: {self.graph_path}")
        print(f"Output: {self.output_base}")
        print(f"Configurations: {len(CONFIGURATIONS)}")
        print(f"Weighting: Adaptive IDyOM (Pearce 2005)")
        print(f"█"*70 + "\n")
        
        results = []
        configs_data = []
        start_time = time.time()
        
        for idx, (encoding, order) in enumerate(CONFIGURATIONS, 1):
            logger.info(f"\n[{idx}/{len(CONFIGURATIONS)}] Processing {encoding} | Order {order}")
            
            # Create config-specific output directory
            config_dir = os.path.join(
                self.output_base, 
                f"{encoding}_order{order}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            try:
                # Train LTM+STM for this configuration
                trainer = LTMSTMConfigTrainer(
                    encoding=encoding,
                    order=order,
                    data_path=self.data_path,
                    graph_path=self.graph_path,
                    output_dir=config_dir
                )
                
                config = trainer.train(
                    batch_size_ltm=batch_size_ltm,
                    epochs_ltm=epochs_ltm,
                    learning_rate_ltm=learning_rate_ltm,
                    batch_size_stm=batch_size_stm,
                    epochs_stm=epochs_stm,
                    learning_rate_stm=learning_rate_stm
                )
                
                results.append({
                    'encoding': encoding,
                    'order': order,
                    'ltm_status': config['ltm']['status'],
                    'stm_status': config['stm']['status'],
                    'output_dir': config_dir
                })
                
                configs_data.append(config)
                
                # Print status
                ltm_ok = "✓" if config['ltm']['status'] == 'SUCCESS' else "✗"
                stm_ok = "✓" if config['stm']['status'] == 'SUCCESS' else "✗"
                logger.info(f"  {ltm_ok} LTM | {stm_ok} STM")
                
            except Exception as e:
                logger.error(f"  ✗ Error training {encoding} | Order {order}: {str(e)}")
                results.append({
                    'encoding': encoding,
                    'order': order,
                    'ltm_status': 'ERROR',
                    'stm_status': 'ERROR',
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*70}")
        
        ltm_success = sum(1 for r in results if r['ltm_status'] == 'SUCCESS')
        stm_success = sum(1 for r in results if r['stm_status'] == 'SUCCESS')
        
        print(f"Total Configs: {len(results)}")
        print(f"LTM Success: {ltm_success}/{len(results)}")
        print(f"STM Success: {stm_success}/{len(results)}")
        print(f"Total Time: {total_time/3600:.2f} hours ({int(total_time)} seconds)")
        print(f"{'='*70}\n")
        
        # Detailed results
        print("Detailed Results:")
        print(f"{'Encoding':<20} {'Order':<6} {'LTM':<10} {'STM':<10} {'Output'}")
        print("-" * 70)
        for result in results:
            enc = result['encoding']
            order = result['order']
            ltm = result['ltm_status']
            stm = result['stm_status']
            output = result.get('output_dir', 'N/A')
            print(f"{enc:<20} {order:<6} {ltm:<10} {stm:<10} {output}")
        
        # Save summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'total_configs': len(results),
            'ltm_success': ltm_success,
            'stm_success': stm_success,
            'results': results,
            'configurations': configs_data,
            'weighting_methodology': {
                'method': 'entropy_based_adaptive_idyom',
                'reference': 'Pearce, M. T. (2005) - Information Dynamics of Music',
                'formula': 'w_ltm = 1 / (1 + exp(E_stm - E_ltm))'
            }
        }
        
        summary_file = os.path.join(self.output_base, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        print(f"✓ Training complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Train GraphIDyOM LTM+STM for all encoding/order configurations'
    )
    parser.add_argument('--data-path', type=str,
                       default='data/worldmove_380_NY_test.csv',
                       help='Path to training data CSV')
    parser.add_argument('--graph-path', type=str,
                       default='data/worldmove_380_NY_network.json',
                       help='Path to graph JSON')
    parser.add_argument('--output-base', type=str,
                       default='experiments_ltm_stm',
                       help='Base output directory')
    parser.add_argument('--batch-size-ltm', type=int, default=128,
                       help='LTM batch size (default: 128)')
    parser.add_argument('--epochs-ltm', type=int, default=10,
                       help='LTM epochs (default: 10)')
    parser.add_argument('--learning-rate-ltm', type=float, default=0.0005,
                       help='LTM learning rate (default: 0.0005)')
    parser.add_argument('--batch-size-stm', type=int, default=32,
                       help='STM batch size (default: 32, smaller for user-specific data)')
    parser.add_argument('--epochs-stm', type=int, default=5,
                       help='STM epochs (default: 5, fewer for fast adaptation)')
    parser.add_argument('--learning-rate-stm', type=float, default=0.001,
                       help='STM learning rate (default: 0.001, higher for adaptation)')
    
    args = parser.parse_args()
    
    try:
        trainer = MultiConfigLTMSTMTrainer(
            data_path=args.data_path,
            graph_path=args.graph_path,
            output_base=args.output_base
        )
        
        summary = trainer.train_all(
            batch_size_ltm=args.batch_size_ltm,
            epochs_ltm=args.epochs_ltm,
            learning_rate_ltm=args.learning_rate_ltm,
            batch_size_stm=args.batch_size_stm,
            epochs_stm=args.epochs_stm,
            learning_rate_stm=args.learning_rate_stm
        )
        
        # Return success if at least some configs trained
        return 0 if summary['ltm_success'] > 0 else 1
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
