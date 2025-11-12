"""
Temporal Metrics for Noodlings - Phase 5 Scientific Validation

This module implements quantitative metrics beyond Φ for evaluating
hierarchical temporal affect models:

1. Temporal Prediction Horizon (TPH): Prediction accuracy at multiple horizons
2. Surprise-Novelty Correlation (SNC): Correlation between surprise and entropy
3. Hierarchical Separation Index (HSI): Timescale separation in layers
4. Personality Consistency Score (PCS): Response consistency across scenarios

Author: Noodlings Project
Date: November 2025
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)


class TemporalMetrics:
    """
    Temporal metrics calculator for Noodlings architecture evaluation.

    This class provides methods to quantify:
    - Temporal prediction capabilities
    - Surprise-novelty relationships
    - Hierarchical timescale separation
    - Personality consistency
    """

    def __init__(self, model):
        """
        Initialize metrics calculator.

        Args:
            model: A Noodlings model instance (ConsilienceModelPhase4 or similar)
        """
        self.model = model

    def calculate_tph(
        self,
        test_data: List[List[mx.array]],
        horizons: List[int] = [1, 5, 10, 20, 50]
    ) -> Dict[int, float]:
        """
        Calculate Temporal Prediction Horizon (TPH) metric.

        Measures the model's ability to predict future affective states at
        multiple time horizons. Lower MSE indicates better long-term prediction.

        Args:
            test_data: List of conversations, each conversation is a list of
                      5-D affect vectors [valence, arousal, fear, sorrow, boredom]
            horizons: List of prediction horizons to evaluate (in timesteps)

        Returns:
            dict: {horizon: mse} mapping each horizon to mean squared error

        Example:
            >>> metrics = TemporalMetrics(model)
            >>> tph_results = metrics.calculate_tph(test_data, [1, 5, 10])
            >>> print(f"5-step prediction MSE: {tph_results[5]:.4f}")
        """
        results = {}

        for h in horizons:
            predictions = []
            targets = []

            for conversation in test_data:
                if len(conversation) <= h:
                    continue

                # Reset model state for each conversation
                self.model.reset_states()

                for t in range(len(conversation) - h):
                    # Process conversation up to time t
                    for i in range(t + 1):
                        affect = conversation[i]
                        if affect.ndim == 1:
                            affect = affect[None, :]  # (1, 5)
                        self.model(affect)

                    # Predict h steps ahead
                    pred_affect = self._predict_ahead(conversation[:t+1], steps=h)
                    target_affect = conversation[t + h]

                    predictions.append(np.array(pred_affect))
                    targets.append(np.array(target_affect))

            if len(predictions) > 0:
                predictions = np.array(predictions)
                targets = np.array(targets)
                mse = np.mean((predictions - targets) ** 2)
                results[h] = float(mse)
            else:
                results[h] = float('nan')
                logger.warning(f"No valid predictions for horizon {h}")

        return results

    def calculate_snc(
        self,
        test_data: List[List[mx.array]]
    ) -> float:
        """
        Calculate Surprise-Novelty Correlation (SNC) metric.

        Measures the correlation between the model's internal surprise signal
        and information-theoretic novelty (Shannon entropy). High correlation
        (r > 0.7) indicates the model's surprise aligns with objective novelty.

        Args:
            test_data: List of conversations, each conversation is a list of
                      5-D affect vectors

        Returns:
            float: Pearson correlation coefficient (range: -1 to 1)
                  Values > 0.7 indicate strong alignment with novelty

        Example:
            >>> metrics = TemporalMetrics(model)
            >>> snc = metrics.calculate_snc(test_data)
            >>> print(f"Surprise-Novelty Correlation: {snc:.3f}")
        """
        surprises = []
        entropies = []

        for conversation in test_data:
            self.model.reset_states()

            for t, affect in enumerate(conversation):
                if affect.ndim == 1:
                    affect = affect[None, :]  # (1, 5)

                # Get model surprise
                _, surprise = self.model(affect)
                surprises.append(float(surprise))

                # Calculate information-theoretic entropy
                entropy = self._calculate_entropy(affect)
                entropies.append(float(entropy))

        if len(surprises) < 2:
            logger.warning("Insufficient data for correlation calculation")
            return float('nan')

        # Pearson correlation
        correlation, p_value = pearsonr(surprises, entropies)

        logger.info(f"SNC correlation: {correlation:.3f} (p={p_value:.4f})")

        return float(correlation)

    def calculate_hsi(
        self,
        test_data: List[List[mx.array]]
    ) -> Dict[str, float]:
        """
        Calculate Hierarchical Separation Index (HSI) metric.

        Measures timescale separation in the hierarchical layers by computing
        variance ratios. Proper separation means:
        - Fast layer: high variance (responds to immediate changes)
        - Medium layer: moderate variance (tracks conversation dynamics)
        - Slow layer: low variance (stable personality model)

        Args:
            test_data: List of conversations, each conversation is a list of
                      5-D affect vectors

        Returns:
            dict: {
                'slow/fast': ratio (should be << 1, ideally < 0.2),
                'slow/medium': ratio (should be < 1),
                'medium/fast': ratio (should be < 1),
                'interpretation': human-readable summary
            }

        Example:
            >>> metrics = TemporalMetrics(model)
            >>> hsi = metrics.calculate_hsi(test_data)
            >>> print(f"Slow/Fast ratio: {hsi['slow/fast']:.3f}")
            >>> print(hsi['interpretation'])
        """
        fast_deltas = []
        medium_deltas = []
        slow_deltas = []

        for conversation in test_data:
            self.model.reset_states()

            # Get state trajectory for this conversation
            states = self._get_state_trajectory(conversation)

            # Calculate deltas (rate of change) for each layer
            if len(states['fast']) > 1:
                fast_deltas.extend(np.diff(states['fast'], axis=0))
            if len(states['medium']) > 1:
                medium_deltas.extend(np.diff(states['medium'], axis=0))
            if len(states['slow']) > 1:
                slow_deltas.extend(np.diff(states['slow'], axis=0))

        # Calculate variance of deltas
        fast_var = np.var(fast_deltas) if len(fast_deltas) > 0 else 0.0
        medium_var = np.var(medium_deltas) if len(medium_deltas) > 0 else 0.0
        slow_var = np.var(slow_deltas) if len(slow_deltas) > 0 else 0.0

        # Compute ratios (avoid division by zero)
        results = {
            'slow/fast': float(slow_var / fast_var) if fast_var > 0 else float('nan'),
            'slow/medium': float(slow_var / medium_var) if medium_var > 0 else float('nan'),
            'medium/fast': float(medium_var / fast_var) if fast_var > 0 else float('nan'),
            'fast_var': float(fast_var),
            'medium_var': float(medium_var),
            'slow_var': float(slow_var)
        }

        # Interpretation
        slow_fast_ratio = results['slow/fast']
        if slow_fast_ratio < 0.1:
            interpretation = "Excellent separation: slow layer is very stable"
        elif slow_fast_ratio < 0.3:
            interpretation = "Good separation: hierarchical timescales present"
        elif slow_fast_ratio < 0.7:
            interpretation = "Moderate separation: some timescale distinction"
        else:
            interpretation = "Poor separation: layers change at similar rates"

        results['interpretation'] = interpretation

        logger.info(f"HSI ratios - Slow/Fast: {slow_fast_ratio:.3f}, {interpretation}")

        return results

    def calculate_pcs(
        self,
        test_scenarios: Dict[str, List[mx.array]],
        num_trials: int = 5
    ) -> Dict[str, float]:
        """
        Calculate Personality Consistency Score (PCS) metric.

        Measures consistency of agent responses across similar scenarios.
        High consistency (> 0.8) indicates stable personality, while low
        consistency suggests erratic behavior.

        Args:
            test_scenarios: Dict mapping scenario types to lists of affect inputs
                           e.g., {'greeting': [...], 'conflict': [...], 'praise': [...]}
            num_trials: Number of times to test each scenario (with state reset)

        Returns:
            dict: {
                'overall': average consistency score (0 to 1),
                'by_scenario': per-scenario consistency scores,
                'interpretation': human-readable summary
            }

        Example:
            >>> scenarios = {
            ...     'greeting': [happy_affect1, happy_affect2],
            ...     'conflict': [tense_affect1, tense_affect2]
            ... }
            >>> metrics = TemporalMetrics(model)
            >>> pcs = metrics.calculate_pcs(scenarios)
            >>> print(f"Consistency: {pcs['overall']:.3f}")
        """
        consistency_by_scenario = {}

        for scenario_type, utterances in test_scenarios.items():
            all_responses = []

            # Test each utterance multiple times with state reset
            for utterance in utterances:
                trial_responses = []

                for _ in range(num_trials):
                    self.model.reset_states()

                    if utterance.ndim == 1:
                        utterance = utterance[None, :]  # (1, 5)

                    # Get phenomenal state response
                    phenomenal_state, _ = self.model(utterance)

                    # Extract full state vector
                    h_fast = np.array(self.model.h_fast).flatten()
                    h_medium = np.array(self.model.h_medium).flatten()
                    h_slow = np.array(self.model.h_slow).flatten()

                    response_vector = np.concatenate([h_fast, h_medium, h_slow])
                    trial_responses.append(response_vector)

                all_responses.extend(trial_responses)

            # Calculate consistency as 1 - variance
            if len(all_responses) > 1:
                responses_array = np.array(all_responses)
                variance = np.var(responses_array, axis=0).mean()
                consistency = 1.0 - min(variance, 1.0)  # Clamp to [0, 1]
                consistency_by_scenario[scenario_type] = float(consistency)
            else:
                consistency_by_scenario[scenario_type] = float('nan')

        # Overall consistency
        valid_scores = [s for s in consistency_by_scenario.values() if not np.isnan(s)]
        overall = float(np.mean(valid_scores)) if len(valid_scores) > 0 else float('nan')

        # Interpretation
        if overall > 0.8:
            interpretation = "High consistency: stable personality traits"
        elif overall > 0.6:
            interpretation = "Moderate consistency: mostly coherent behavior"
        elif overall > 0.4:
            interpretation = "Low consistency: variable responses"
        else:
            interpretation = "Very low consistency: erratic behavior"

        results = {
            'overall': overall,
            'by_scenario': consistency_by_scenario,
            'interpretation': interpretation
        }

        logger.info(f"PCS overall: {overall:.3f}, {interpretation}")

        return results

    # Helper methods

    def _predict_ahead(
        self,
        context: List[mx.array],
        steps: int
    ) -> mx.array:
        """
        Predict affective state `steps` timesteps in the future.

        Uses the model's predictor network to forecast future states.

        Args:
            context: List of affect vectors leading up to current time
            steps: Number of timesteps to predict ahead

        Returns:
            mx.array: Predicted 5-D affect vector
        """
        # Use the predictor network to forecast
        # For now, use a simple approach: predict next state iteratively

        current_state = self.model.get_phenomenal_state()

        # Simple iterative prediction (can be improved with multi-step predictor)
        for _ in range(steps):
            # Use predictor to get next phenomenal state
            predicted_phenom = self.model.predictor(current_state)

            # Extract fast layer (first 16 dims) as affect proxy
            # This is a simplification; ideally we'd have affect decoder
            current_state = predicted_phenom

        # Map phenomenal state back to 5-D affect (simplified)
        # In practice, you might want an affect decoder network
        h_fast = current_state[:, :16]  # Fast layer = affect-responsive

        # Simple projection to 5-D affect space
        # Take first 5 dimensions as rough affect proxy (placeholder)
        # A proper decoder network would be trained for this
        affect_pred = h_fast[:, :5]

        return affect_pred[0]  # Return (5,) vector

    def _calculate_entropy(
        self,
        affect: mx.array
    ) -> float:
        """
        Calculate Shannon entropy of affect vector.

        Measures information-theoretic novelty of the input.

        Args:
            affect: 5-D affect vector

        Returns:
            float: Shannon entropy
        """
        # Discretize affect into bins for entropy calculation
        bins = self._discretize_affect(affect)

        # Calculate probability distribution
        unique, counts = np.unique(bins, return_counts=True)
        probs = counts / counts.sum()

        # Shannon entropy: -sum(p * log2(p))
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return float(entropy)

    def _discretize_affect(
        self,
        affect: mx.array,
        num_bins: int = 5
    ) -> np.ndarray:
        """
        Discretize continuous affect vector into bins.

        Args:
            affect: 5-D affect vector
            num_bins: Number of bins per dimension

        Returns:
            np.ndarray: Discretized affect codes
        """
        affect_np = np.array(affect).flatten()

        # Bin each dimension
        bins = []
        for val in affect_np:
            # Map [-1, 1] or [0, 1] to [0, num_bins-1]
            normalized = (val + 1) / 2  # Shift to [0, 1]
            bin_idx = int(normalized * num_bins)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)
            bins.append(bin_idx)

        return np.array(bins)

    def _get_state_trajectory(
        self,
        conversation: List[mx.array]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Get trajectory of hierarchical states over a conversation.

        Args:
            conversation: List of affect vectors

        Returns:
            dict: {
                'fast': list of fast layer states (16-D),
                'medium': list of medium layer states (16-D),
                'slow': list of slow layer states (8-D)
            }
        """
        fast_states = []
        medium_states = []
        slow_states = []

        self.model.reset_states()

        for affect in conversation:
            if affect.ndim == 1:
                affect = affect[None, :]  # (1, 5)

            # Process one timestep
            self.model(affect)

            # Record states
            fast_states.append(np.array(self.model.h_fast).flatten())
            medium_states.append(np.array(self.model.h_medium).flatten())
            slow_states.append(np.array(self.model.h_slow).flatten())

        return {
            'fast': fast_states,
            'medium': medium_states,
            'slow': slow_states
        }


# Convenience function for testing
def test_metrics_on_random_data():
    """
    Test metrics implementation with random data.

    Useful for verifying metrics work before running on real model.
    """
    print("Testing TemporalMetrics with random data...")

    # Create dummy model (needs to have the right interface)
    class DummyModel:
        def __init__(self):
            self.h_fast = mx.zeros((1, 16))
            self.h_medium = mx.zeros((1, 16))
            self.h_slow = mx.zeros((1, 8))

        def __call__(self, affect):
            # Random state update
            self.h_fast = mx.random.normal((1, 16))
            self.h_medium = mx.random.normal((1, 16))
            self.h_slow = mx.random.normal((1, 8))
            surprise = float(mx.random.uniform(0, 1))
            return self.h_fast, surprise

        def reset_states(self):
            self.h_fast = mx.zeros((1, 16))
            self.h_medium = mx.zeros((1, 16))
            self.h_slow = mx.zeros((1, 8))

        def get_phenomenal_state(self):
            return mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=1)

        def predictor(self, state):
            return mx.random.normal(state.shape)

    # Create test data
    test_data = []
    for _ in range(10):  # 10 conversations
        conversation = []
        for _ in range(20):  # 20 timesteps each
            affect = mx.random.uniform(-1, 1, (5,))
            conversation.append(affect)
        test_data.append(conversation)

    # Create scenarios
    test_scenarios = {
        'positive': [mx.array([0.8, 0.7, 0.1, 0.1, 0.2]) for _ in range(5)],
        'negative': [mx.array([-0.8, 0.5, 0.7, 0.6, 0.3]) for _ in range(5)]
    }

    # Test metrics
    model = DummyModel()
    metrics = TemporalMetrics(model)

    print("\n1. Testing TPH...")
    tph = metrics.calculate_tph(test_data, horizons=[1, 5, 10])
    print(f"   TPH results: {tph}")

    print("\n2. Testing SNC...")
    snc = metrics.calculate_snc(test_data)
    print(f"   SNC: {snc:.3f}")

    print("\n3. Testing HSI...")
    hsi = metrics.calculate_hsi(test_data)
    print(f"   HSI: {hsi}")

    print("\n4. Testing PCS...")
    pcs = metrics.calculate_pcs(test_scenarios)
    print(f"   PCS: {pcs}")

    print("\n✓ All metrics tested successfully!")


if __name__ == "__main__":
    test_metrics_on_random_data()
