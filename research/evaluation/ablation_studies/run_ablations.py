"""
Ablation Study Framework - Phase 5 Scientific Validation

Compares 6 architecture variants to validate hierarchical design:
1. Baseline: LLM only (no temporal model)
2. Control: LLM + random states
3. Single-layer: LLM + single LSTM
4. Hierarchical: Fast/Medium/Slow (no observers)
5. With Observers: Full system (75 loops)
6. Dense Observers: 2x observer density (150 loops)

Author: Noodlings Project
Date: November 2025
"""

import sys
sys.path.insert(0, '../..')

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Import metrics
from noodlings.metrics.temporal_metrics import TemporalMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Ablation study coordinator for Noodlings architectures.
    
    Runs comprehensive evaluation across multiple architecture variants
    and generates comparative reports.
    """
    
    def __init__(self, test_data_path: str, checkpoint_dir: str):
        """
        Initialize ablation study.
        
        Args:
            test_data_path: Path to test data (JSON format)
            checkpoint_dir: Path to trained checkpoints
        """
        self.test_data_path = test_data_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results = {}
        
    def load_test_data(self) -> Tuple[List[List[mx.array]], Dict]:
        """
        Load test data for evaluation.
        
        Returns:
            Tuple of (conversations, scenarios) where:
            - conversations: List of affect sequences
            - scenarios: Dict of scenario_type -> affect vectors
        """
        logger.info(f"Loading test data from {self.test_data_path}")
        
        # TODO: Load actual test data from JSON
        # For now, generate synthetic test data
        conversations = []
        for _ in range(20):  # 20 test conversations
            conversation = []
            for _ in range(30):  # 30 timesteps each
                # Random affect vector: [valence, arousal, fear, sorrow, boredom]
                affect = mx.random.uniform(-1, 1, (5,))
                conversation.append(affect)
            conversations.append(conversation)
        
        # Test scenarios for PCS
        scenarios = {
            'greeting': [mx.array([0.8, 0.6, 0.1, 0.1, 0.2]) for _ in range(5)],
            'conflict': [mx.array([-0.7, 0.8, 0.7, 0.3, 0.1]) for _ in range(5)],
            'praise': [mx.array([0.9, 0.5, 0.0, 0.0, 0.1]) for _ in range(5)],
            'neutral': [mx.array([0.0, 0.3, 0.2, 0.2, 0.4]) for _ in range(5)]
        }
        
        logger.info(f"Loaded {len(conversations)} conversations, {len(scenarios)} scenario types")
        
        return conversations, scenarios
    
    def define_architectures(self) -> Dict[str, Dict]:
        """
        Define the 6 architecture variants for comparison.
        
        Returns:
            Dict mapping architecture name to config
        """
        architectures = {
            '1_baseline_llm_only': {
                'description': 'LLM only (no temporal model)',
                'checkpoint': None,  # No checkpoint needed
                'use_temporal': False,
                'use_observers': False
            },
            '2_control_random_states': {
                'description': 'LLM + random states',
                'checkpoint': None,
                'use_temporal': 'random',
                'use_observers': False
            },
            '3_single_layer': {
                'description': 'LLM + single LSTM',
                'checkpoint': self.checkpoint_dir / 'single_layer' / 'model.safetensors',
                'use_temporal': 'single',
                'use_observers': False
            },
            '4_hierarchical': {
                'description': 'Fast/Medium/Slow (no observers)',
                'checkpoint': self.checkpoint_dir / 'hierarchical' / 'model.safetensors',
                'use_temporal': 'hierarchical',
                'use_observers': False
            },
            '5_with_observers': {
                'description': 'Full system (75 observer loops)',
                'checkpoint': self.checkpoint_dir / 'with_observers' / 'model.safetensors',
                'use_temporal': 'hierarchical',
                'use_observers': 75
            },
            '6_dense_observers': {
                'description': 'Dense observers (150 loops)',
                'checkpoint': self.checkpoint_dir / 'dense_observers' / 'model.safetensors',
                'use_temporal': 'hierarchical',
                'use_observers': 150
            }
        }
        
        return architectures
    
    def load_model(self, config: Dict):
        """
        Load model based on architecture config.
        
        Args:
            config: Architecture configuration
            
        Returns:
            Model instance
        """
        # TODO: Implement actual model loading
        # For now, return a dummy model
        
        class DummyModel:
            def __init__(self, use_temporal, use_observers):
                self.use_temporal = use_temporal
                self.use_observers = use_observers
                self.h_fast = mx.zeros((1, 16))
                self.h_medium = mx.zeros((1, 16))
                self.h_slow = mx.zeros((1, 8))
                
            def __call__(self, affect):
                if self.use_temporal == False:
                    # Baseline: no temporal state
                    return mx.zeros((1, 16)), 0.0
                elif self.use_temporal == 'random':
                    # Control: random states
                    self.h_fast = mx.random.normal((1, 16))
                    self.h_medium = mx.random.normal((1, 16))
                    self.h_slow = mx.random.normal((1, 8))
                    return self.h_fast, float(mx.random.uniform(0, 1))
                else:
                    # Temporal models: some learning
                    self.h_fast = 0.9 * self.h_fast + 0.1 * mx.random.normal((1, 16))
                    self.h_medium = 0.95 * self.h_medium + 0.05 * mx.random.normal((1, 16))
                    self.h_slow = 0.99 * self.h_slow + 0.01 * mx.random.normal((1, 8))
                    surprise = float(mx.mean(mx.abs(affect)))
                    return self.h_fast, surprise
                    
            def reset_states(self):
                self.h_fast = mx.zeros((1, 16))
                self.h_medium = mx.zeros((1, 16))
                self.h_slow = mx.zeros((1, 8))
                
            def get_phenomenal_state(self):
                return mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=1)
                
            def predictor(self, state):
                return mx.random.normal(state.shape)
        
        return DummyModel(config['use_temporal'], config['use_observers'])
    
    def evaluate_architecture(
        self,
        name: str,
        config: Dict,
        test_data: List[List[mx.array]],
        test_scenarios: Dict
    ) -> Dict:
        """
        Evaluate a single architecture on all metrics.
        
        Args:
            name: Architecture name
            config: Architecture configuration
            test_data: Test conversations
            test_scenarios: Test scenarios for PCS
            
        Returns:
            Dict of metric results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {name}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*60}")
        
        # Load model
        model = self.load_model(config)
        
        # Initialize metrics calculator
        metrics = TemporalMetrics(model)
        
        # Run all metrics
        results = {
            'name': name,
            'description': config['description'],
            'checkpoint': str(config['checkpoint']) if config['checkpoint'] else None,
        }
        
        # 1. Temporal Prediction Horizon (TPH)
        logger.info("Running TPH...")
        start_time = time.time()
        tph_results = metrics.calculate_tph(test_data, horizons=[1, 5, 10, 20])
        results['TPH'] = tph_results
        results['TPH_time'] = time.time() - start_time
        logger.info(f"  TPH: {tph_results}")
        
        # 2. Surprise-Novelty Correlation (SNC)
        logger.info("Running SNC...")
        start_time = time.time()
        snc_result = metrics.calculate_snc(test_data)
        results['SNC'] = snc_result
        results['SNC_time'] = time.time() - start_time
        logger.info(f"  SNC: {snc_result:.3f}")
        
        # 3. Hierarchical Separation Index (HSI)
        logger.info("Running HSI...")
        start_time = time.time()
        hsi_results = metrics.calculate_hsi(test_data)
        results['HSI'] = hsi_results
        results['HSI_time'] = time.time() - start_time
        logger.info(f"  HSI: {hsi_results['interpretation']}")
        
        # 4. Personality Consistency Score (PCS)
        logger.info("Running PCS...")
        start_time = time.time()
        pcs_results = metrics.calculate_pcs(test_scenarios, num_trials=3)
        results['PCS'] = pcs_results
        results['PCS_time'] = time.time() - start_time
        logger.info(f"  PCS: {pcs_results['overall']:.3f} - {pcs_results['interpretation']}")
        
        # 5. Count parameters
        # TODO: Implement actual parameter counting
        results['parameters'] = 0
        
        # 6. Measure inference time
        logger.info("Measuring inference time...")
        affect = mx.random.uniform(-1, 1, (1, 5))
        start_time = time.time()
        for _ in range(100):
            model(affect)
        inference_time_ms = (time.time() - start_time) / 100 * 1000
        results['inference_time_ms'] = inference_time_ms
        logger.info(f"  Inference: {inference_time_ms:.2f}ms per turn")
        
        return results
    
    def run_all(self) -> Dict[str, Dict]:
        """
        Run ablation study on all architectures.
        
        Returns:
            Dict mapping architecture name to results
        """
        logger.info("Starting ablation study...")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        
        # Load test data
        test_data, test_scenarios = self.load_test_data()
        
        # Define architectures
        architectures = self.define_architectures()
        
        # Evaluate each architecture
        self.results = {}
        for name, config in architectures.items():
            self.results[name] = self.evaluate_architecture(
                name, config, test_data, test_scenarios
            )
        
        logger.info("\n" + "="*60)
        logger.info("Ablation study complete!")
        logger.info("="*60)
        
        return self.results
    
    def generate_report(self, output_path: str = None):
        """
        Generate markdown report with comparative results.
        
        Args:
            output_path: Where to save report (default: evaluation/reports/ablation_report.md)
        """
        if output_path is None:
            output_path = Path(__file__).parent.parent / 'reports' / 'ablation_report.md'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating report at {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("# Noodlings Ablation Study Report\n\n")
            f.write("**Date**: November 2025\n\n")
            f.write("**Objective**: Compare 6 architecture variants to validate hierarchical design.\n\n")
            f.write("---\n\n")
            
            # Summary table
            f.write("## Summary Table\n\n")
            f.write("| Architecture | TPH@10 | SNC | HSI (S/F) | PCS | Inference (ms) |\n")
            f.write("|-------------|--------|-----|-----------|-----|----------------|\n")
            
            for name, results in self.results.items():
                tph_10 = results['TPH'].get(10, float('nan'))
                snc = results['SNC']
                hsi_sf = results['HSI'].get('slow/fast', float('nan'))
                pcs = results['PCS']['overall']
                inference = results['inference_time_ms']
                
                f.write(f"| {results['description']} | {tph_10:.4f} | {snc:.3f} | {hsi_sf:.3f} | {pcs:.3f} | {inference:.2f} |\n")
            
            f.write("\n---\n\n")
            
            # Detailed results for each architecture
            f.write("## Detailed Results\n\n")
            
            for name, results in self.results.items():
                f.write(f"### {name}: {results['description']}\n\n")
                
                # TPH
                f.write("**Temporal Prediction Horizon (TPH)**\n")
                for horizon, mse in results['TPH'].items():
                    f.write(f"- {horizon}-step: MSE = {mse:.4f}\n")
                f.write("\n")
                
                # SNC
                f.write(f"**Surprise-Novelty Correlation (SNC)**: {results['SNC']:.3f}\n\n")
                
                # HSI
                f.write("**Hierarchical Separation Index (HSI)**\n")
                hsi = results['HSI']
                f.write(f"- Slow/Fast ratio: {hsi['slow/fast']:.3f}\n")
                f.write(f"- Slow/Medium ratio: {hsi['slow/medium']:.3f}\n")
                f.write(f"- Medium/Fast ratio: {hsi['medium/fast']:.3f}\n")
                f.write(f"- Interpretation: {hsi['interpretation']}\n\n")
                
                # PCS
                f.write(f"**Personality Consistency Score (PCS)**: {results['PCS']['overall']:.3f}\n")
                f.write(f"- Interpretation: {results['PCS']['interpretation']}\n\n")
                
                # Performance
                f.write(f"**Inference Time**: {results['inference_time_ms']:.2f}ms per turn\n\n")
                
                f.write("---\n\n")
            
            # Analysis
            f.write("## Analysis\n\n")
            f.write("### Key Findings\n\n")
            f.write("*TODO: Add interpretation after running on trained models*\n\n")
            f.write("### Do Observer Loops Add Value?\n\n")
            f.write("Comparing architectures 4 (hierarchical) vs 5 (with observers):\n\n")
            f.write("*TODO: Add comparison*\n\n")
            f.write("### Computational Cost vs. Performance\n\n")
            f.write("*TODO: Add cost-benefit analysis*\n\n")
            
        logger.info(f"✓ Report saved to {output_path}")
    
    def save_results(self, output_path: str = None):
        """
        Save raw results to JSON.
        
        Args:
            output_path: Where to save JSON (default: evaluation/ablation_studies/results.json)
        """
        if output_path is None:
            output_path = Path(__file__).parent / 'results.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = {}
        for name, results in self.results.items():
            json_results[name] = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[name][key] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, (np.floating, np.integer)):
                    json_results[name][key] = float(value)
                else:
                    json_results[name][key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_path}")


def main():
    """Run ablation study from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Noodlings ablation study')
    parser.add_argument('--test-data', type=str, default='../../training/data/synthetic/test.json',
                       help='Path to test data')
    parser.add_argument('--checkpoints', type=str, default='../../training/checkpoints',
                       help='Path to checkpoint directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for report')
    
    args = parser.parse_args()
    
    # Create ablation study
    study = AblationStudy(args.test_data, args.checkpoints)
    
    # Run all evaluations
    results = study.run_all()
    
    # Save results
    study.save_results()
    
    # Generate report
    study.generate_report(args.output)
    
    print("\n" + "="*60)
    print("Ablation study complete!")
    print(f"Report: evaluation/reports/ablation_report.md")
    print(f"Raw results: evaluation/ablation_studies/results.json")
    print("="*60)


if __name__ == "__main__":
    main()
