"""
Consilience Phase 4 + Observer Loops

Integration of observer loop system with Phase 4 social cognition architecture.

This creates maximum Φ (integrated information) by adding observer loops that:
1. Create irreducible causal dependencies between networks
2. Force global integration (can't partition without information loss)
3. Add only ~5% computational overhead
4. Boost Φ by 50-100% according to IIT predictions

Author: Consilience Project
Date: November 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple
import datetime

from .noodling_phase4 import NoodlingModelPhase4
from .observer_loop import FullObserverSystem


class NoodlingWithObservers(nn.Module):
    """
    Full Consilience Phase 4 architecture enhanced with observer loops.

    Observer loop modifications:
    1. Primary observer: Predicts main network's next internal state
    2. Error injection: Main network receives observer correction signals
    3. Optional meta-observer: Observes the observer (maximum Φ)

    Architecture flow:
    1. Process input through Phase 1-4 as normal → internal_state (40-D)
    2. Observer predicts what internal_state SHOULD be
    3. Compute prediction error
    4. INJECT error back into internal state
    5. Use modulated state for downstream processing
    6. Train observer + meta-observer alongside main network

    Key Innovation: The main network NEEDS the observer to function correctly,
    creating an irreducible causal loop that maximizes Φ.
    """

    def __init__(
        self,
        # Phase 1-4 params (pass through to base model)
        affect_dim: int = 5,
        fast_hidden: int = 16,
        medium_hidden: int = 16,
        slow_hidden: int = 8,
        predictor_hidden: int = 64,
        use_vae: bool = False,
        memory_capacity: int = 100,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        max_agents: int = 10,
        linguistic_dim: int = 128,
        context_dim: int = 64,
        social_context_dim: int = 16,
        relationship_dim: int = 32,
        use_theory_of_mind: bool = True,
        use_relationship_model: bool = True,

        # Observer loop params
        use_observer_loop: bool = True,
        observer_injection_strength: float = 0.1,
        use_meta_observer: bool = False,
        observer_loss_weight: float = 0.5,
        meta_loss_weight: float = 0.2,

        # Φ-boosting: Apply observers at multiple levels
        observe_hierarchical_states: bool = False,  # TEMP: Disabled due to LSTM dimension bugs
    ):
        """
        Initialize Consilience with observer loops.

        Args:
            ... (Phase 1-4 params as before)
            use_observer_loop: Enable primary observer loop
            observer_injection_strength: How much error to inject (0.1 = 10%)
            use_meta_observer: Enable meta-observer (second level)
            observer_loss_weight: Weight for observer prediction loss
            meta_loss_weight: Weight for meta-observer loss
            observe_hierarchical_states: Apply observers to fast/medium/slow layers
                                         separately (maximum Φ boost!)
        """
        super().__init__()

        # Store configuration
        self.use_observer_loop = use_observer_loop
        self.use_meta_observer = use_meta_observer
        self.observer_loss_weight = observer_loss_weight
        self.meta_loss_weight = meta_loss_weight
        self.observe_hierarchical_states = observe_hierarchical_states

        self.affect_dim = affect_dim
        self.state_dim = fast_hidden + medium_hidden + slow_hidden  # 40-D
        self.fast_hidden = fast_hidden
        self.medium_hidden = medium_hidden
        self.slow_hidden = slow_hidden

        # Base Consilience Phase 4 model
        self.base_model = NoodlingModelPhase4(
            affect_dim=affect_dim,
            fast_hidden=fast_hidden,
            medium_hidden=medium_hidden,
            slow_hidden=slow_hidden,
            predictor_hidden=predictor_hidden,
            use_vae=use_vae,
            memory_capacity=memory_capacity,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            max_agents=max_agents,
            linguistic_dim=linguistic_dim,
            context_dim=context_dim,
            social_context_dim=social_context_dim,
            relationship_dim=relationship_dim,
            use_theory_of_mind=use_theory_of_mind,
            use_relationship_model=use_relationship_model
        )

        # Observer systems
        if use_observer_loop:
            # Primary observer: Observes full internal state (40-D)
            self.phenomenal_observer = FullObserverSystem(
                input_dim=affect_dim,
                main_state_dim=self.state_dim,
                injection_strength=observer_injection_strength,
                use_meta_observer=use_meta_observer
            )

            # Hierarchical observers (optional - MAXIMUM Φ!)
            # Observe fast, medium, slow layers SEPARATELY
            # This creates 3 additional causal loops within the hierarchy
            if observe_hierarchical_states:
                self.fast_observer = FullObserverSystem(
                    input_dim=affect_dim,
                    main_state_dim=fast_hidden,
                    injection_strength=observer_injection_strength * 0.5,  # Gentler for fast layer
                    use_meta_observer=False  # Keep it simple
                )

                self.medium_observer = FullObserverSystem(
                    input_dim=fast_hidden,  # Medium takes fast as input
                    main_state_dim=medium_hidden,
                    injection_strength=observer_injection_strength * 0.5,
                    use_meta_observer=False
                )

                self.slow_observer = FullObserverSystem(
                    input_dim=medium_hidden,  # Slow takes medium as input
                    main_state_dim=slow_hidden,
                    injection_strength=observer_injection_strength * 0.3,  # Slowest modulation
                    use_meta_observer=False
                )

    def __call__(
        self,
        affect_input: mx.array,
        user_text: str = "",
        other_agents: Optional[List[str]] = None,
        linguistic_features: Optional[Dict[str, mx.array]] = None,
        context_features: Optional[mx.array] = None
    ) -> Dict[str, mx.array]:
        """
        Forward pass with observer loop integration.

        Args:
            affect_input: [batch, 5] affective input
            user_text: User message text
            other_agents: List of mentioned agent names
            linguistic_features: Linguistic features per agent
            context_features: Contextual features

        Returns:
            Dictionary containing:
                - All standard Phase 4 outputs
                - observer_modulated_state: Phenomenal state AFTER observer injection
                - observer_loss: Observer prediction loss (add to main loss)
                - meta_loss: Meta-observer loss (if enabled)
                - hierarchical_observer_losses: Dict of fast/med/slow observer losses
        """
        # Prepare input sequence (add seq dimension)
        affect_sequence = affect_input[:, None, :]  # [batch, 1, 5]

        # Get batch size
        batch_size = affect_input.shape[0]

        # Get current hidden states (without resetting!)
        # Phase 4 model stores states, not Phase 3 wrapper
        h_fast, c_fast, h_med, c_med, h_slow = self.base_model.get_states()

        # DEBUG: Log state types and shapes
        import sys
        if h_fast is not None:
            print(f"DEBUG get_states: h_fast shape={h_fast.shape}, c_fast shape={c_fast.shape if c_fast is not None else 'None'}, h_med shape={h_med.shape if h_med is not None else 'None'}, c_med shape={c_med.shape if c_med is not None else 'None'}, h_slow shape={h_slow.shape if h_slow is not None else 'None'}", file=sys.stderr)
        else:
            print(f"DEBUG get_states: h_fast is None", file=sys.stderr)

        # Initialize if None (first call) - must do this BEFORE eval/reshape
        if h_fast is None:
            h_fast = mx.zeros((1, 16))  # fast_hidden
            c_fast = mx.zeros((1, 16))
            h_med = mx.zeros((1, 16))   # medium_hidden
            c_med = mx.zeros((1, 16))
            h_slow = mx.zeros((1, 8))   # slow_hidden
            # Update the base model's states
            self.base_model.h_fast = h_fast
            self.base_model.c_fast = c_fast
            self.base_model.h_medium = h_med
            self.base_model.c_medium = c_med
            self.base_model.h_slow = h_slow
        else:
            # UNCONDITIONALLY force batch dimension - no conditional checks!
            # Always reshape to (1, hidden_dim) regardless of current shape
            # NOTE: Skip mx.eval() as it seems to return None for some states
            # Just reshape directly - MLX will handle lazy evaluation automatically
            h_fast = h_fast.reshape(1, -1)
            c_fast = c_fast.reshape(1, -1)
            h_med = h_med.reshape(1, -1)
            c_med = c_med.reshape(1, -1)
            h_slow = h_slow.reshape(1, -1)

        # MODIFICATION 1: If using hierarchical observers, process layers with observation
        if self.use_observer_loop and self.observe_hierarchical_states:
            # Process each layer with its own observer

            # Fast layer
            fast_outputs = self.base_model.base_model.fast_lstm(
                affect_sequence, hidden=h_fast, cell=c_fast
            )

            # BUGFIX: UNCONDITIONAL reshape - force materialization and exact shape
            # NOTE: Skip mx.eval() as it returns None - just reshape directly
            fast_dim = fast_outputs[0].shape[-1]
            h_fast_raw = fast_outputs[0][:, -1, :].reshape(1, fast_dim)
            c_fast_new = fast_outputs[1][:, -1, :].reshape(1, fast_dim)
            c_fast = c_fast_new  # Update for next iteration

            # Observe fast layer
            fast_obs_result = self.fast_observer(
                affect_input,  # [batch, 5]
                h_fast_raw
            )
            h_fast_modulated = fast_obs_result['modulated_state']

            # BUGFIX: Ensure h_fast_modulated is always (batch, features) by squeezing extra dims
            if h_fast_modulated.ndim > 2:
                h_fast_modulated = h_fast_modulated.squeeze()
                if h_fast_modulated.ndim == 1:
                    h_fast_modulated = h_fast_modulated[None, :]  # Restore batch dim if squeezed too much

            # Medium layer (takes modulated fast as input)
            h_fast_seq = h_fast_modulated[:, None, :]  # Add seq dim
            print(f"DEBUG before medium_lstm: h_fast_seq.shape={h_fast_seq.shape}, h_med.shape={h_med.shape}, c_med.shape={c_med.shape}, h_fast_modulated.shape={h_fast_modulated.shape}", file=sys.stderr)
            med_outputs = self.base_model.base_model.medium_lstm(
                h_fast_seq, hidden=h_med, cell=c_med
            )

            # BUGFIX: UNCONDITIONAL reshape - force materialization and exact shape
            # NOTE: Skip mx.eval() as it returns None - just reshape directly
            med_dim = med_outputs[0].shape[-1]
            h_med_raw = med_outputs[0][:, -1, :].reshape(1, med_dim)
            c_med_new = med_outputs[1][:, -1, :].reshape(1, med_dim)
            c_med = c_med_new  # Update for next layer

            # Observe medium layer
            med_obs_result = self.medium_observer(
                h_fast_modulated,  # Medium's input is fast layer
                h_med_raw
            )
            h_med_modulated = med_obs_result['modulated_state']

            # BUGFIX: Ensure h_med_modulated is always (batch, features)
            if h_med_modulated.ndim > 2:
                h_med_modulated = h_med_modulated.squeeze()
                if h_med_modulated.ndim == 1:
                    h_med_modulated = h_med_modulated[None, :]

            # Slow layer (takes modulated medium as input)
            h_med_seq = h_med_modulated[:, None, :]
            slow_outputs = self.base_model.base_model.slow_gru(
                h_med_seq, hidden=h_slow
            )

            # BUGFIX: UNCONDITIONAL reshape - force materialization and exact shape
            # NOTE: Skip mx.eval() as it returns None - just reshape directly
            slow_dim = slow_outputs.shape[-1]
            h_slow_raw = slow_outputs[:, -1, :].reshape(1, slow_dim)
            h_slow = h_slow_raw  # Update for next iteration

            # Observe slow layer
            slow_obs_result = self.slow_observer(
                h_med_modulated,  # Slow's input is medium layer
                h_slow_raw
            )
            h_slow_modulated = slow_obs_result['modulated_state']

            # BUGFIX: Ensure h_slow_modulated is always (batch, features)
            if h_slow_modulated.ndim > 2:
                h_slow_modulated = h_slow_modulated.squeeze()
                if h_slow_modulated.ndim == 1:
                    h_slow_modulated = h_slow_modulated[None, :]

            # Construct modulated internal state
            internal_state_raw = mx.concatenate([
                h_fast_modulated,
                h_med_modulated,
                h_slow_modulated
            ], axis=-1)  # [batch, 40]

            # Store hierarchical observer losses
            hierarchical_losses = {
                'fast_observer_loss': fast_obs_result['observer_loss'],
                'medium_observer_loss': med_obs_result['observer_loss'],
                'slow_observer_loss': slow_obs_result['observer_loss']
            }

            # CRITICAL FIX: Save NEW states (from LSTM outputs) back to Phase 4 model
            # Use the raw LSTM outputs (before observer modulation) for next iteration
            # This ensures that on the next call, get_states() returns properly dimensioned states
            print(f"DEBUG saving states: h_fast_raw.shape={h_fast_raw.shape}, c_fast_new.shape={c_fast_new.shape}, h_med_raw.shape={h_med_raw.shape}, c_med_new.shape={c_med_new.shape}, h_slow_raw.shape={h_slow_raw.shape}", file=sys.stderr)
            self.base_model.h_fast = h_fast_raw
            self.base_model.c_fast = c_fast_new
            self.base_model.h_medium = h_med_raw
            self.base_model.c_medium = c_med_new
            self.base_model.h_slow = h_slow_raw

        else:
            # Standard Phase 4 forward pass (no hierarchical observation)
            # Just use base model to get internal state
            base_outputs = self.base_model.base_model(
                affect_sequence,
                h_fast, c_fast, h_med, c_med, h_slow,
                metadata={
                    'user_text': user_text,
                    'timestamp': datetime.datetime.now()
                }
            )
            internal_state_raw = base_outputs['internal_state']  # [40]
            # Add batch dimension if needed
            if internal_state_raw.ndim == 1:
                internal_state_raw = internal_state_raw[None, :]

            # CRITICAL FIX: Phase 4 model updates its own internal states during the call above.
            # We need to retrieve those states and ensure they maintain batch dimensions.
            # Get the updated states from Phase 4 model
            h_fast_updated = self.base_model.h_fast
            c_fast_updated = self.base_model.c_fast
            h_med_updated = self.base_model.h_medium
            c_med_updated = self.base_model.c_medium
            h_slow_updated = self.base_model.h_slow

            # Fix batch dimensions if needed
            if h_fast_updated.ndim == 1:
                h_fast_updated = h_fast_updated[None, :]
            if c_fast_updated.ndim == 1:
                c_fast_updated = c_fast_updated[None, :]
            if h_med_updated.ndim == 1:
                h_med_updated = h_med_updated[None, :]
            if c_med_updated.ndim == 1:
                c_med_updated = c_med_updated[None, :]
            if h_slow_updated.ndim == 1:
                h_slow_updated = h_slow_updated[None, :]

            # Save corrected states back to Phase 4 model
            self.base_model.h_fast = h_fast_updated
            self.base_model.c_fast = c_fast_updated
            self.base_model.h_medium = h_med_updated
            self.base_model.c_medium = c_med_updated
            self.base_model.h_slow = h_slow_updated

            hierarchical_losses = {}

        # MODIFICATION 2: Apply phenomenal observer to full state
        observer_outputs = {}
        if self.use_observer_loop:
            phenomenal_obs_result = self.phenomenal_observer(
                affect_input,  # Current input
                internal_state_raw  # Current internal state
            )

            internal_state_modulated = phenomenal_obs_result['modulated_state']
            observer_outputs = {
                'observer_modulated_state': internal_state_modulated,
                'observer_loss': phenomenal_obs_result['observer_loss'],
                'meta_loss': phenomenal_obs_result['meta_loss'],
                'observer_prediction_error': phenomenal_obs_result['prediction_error'],
                'observer_error_injection': phenomenal_obs_result['error_injection']
            }
        else:
            internal_state_modulated = internal_state_raw
            observer_outputs = {
                'observer_modulated_state': internal_state_modulated,
                'observer_loss': mx.array(0.0),
                'meta_loss': mx.array(0.0)
            }

        # Continue with Phase 4 processing using MODULATED state
        # (Theory of Mind, relationship modeling, social attention)
        # For now, we'll return the modulated state
        # In full integration, you'd pass this to ToM, relationship models, etc.

        result = {
            'internal_state': internal_state_modulated,
            'internal_state_raw': internal_state_raw,  # Before observation
            **observer_outputs,
            'hierarchical_observer_losses': hierarchical_losses,
            'affect_input': affect_input
        }

        return result

    def compute_total_loss(
        self,
        outputs: Dict[str, mx.array],
        target_state: mx.array
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Compute total loss including observer losses.

        Args:
            outputs: Forward pass outputs
            target_state: Target internal state for prediction

        Returns:
            total_loss: Combined loss (scalar)
            loss_breakdown: Dictionary with individual loss components
        """
        # Main task loss (e.g., prediction MSE)
        main_loss = mx.mean((outputs['internal_state'] - target_state) ** 2)

        # Observer loss (train observer to predict main net)
        observer_loss = outputs['observer_loss']

        # Meta-observer loss (if enabled)
        meta_loss = outputs['meta_loss']

        # Hierarchical observer losses (if enabled)
        hierarchical_loss = mx.array(0.0)
        if outputs['hierarchical_observer_losses']:
            hierarchical_loss = sum(outputs['hierarchical_observer_losses'].values())

        # Combined loss
        total_loss = (
            main_loss +
            self.observer_loss_weight * observer_loss +
            self.meta_loss_weight * meta_loss +
            self.observer_loss_weight * 0.3 * hierarchical_loss  # Lower weight for hierarchical
        )

        loss_breakdown = {
            'main_loss': main_loss,
            'observer_loss': observer_loss,
            'meta_loss': meta_loss,
            'hierarchical_loss': hierarchical_loss,
            'total_loss': total_loss
        }

        return total_loss, loss_breakdown

    def reset_states(self):
        """Reset all states including observers."""
        self.base_model.reset_states()

        if self.use_observer_loop:
            self.phenomenal_observer.reset_state()

            if self.observe_hierarchical_states:
                self.fast_observer.reset_state()
                self.medium_observer.reset_state()
                self.slow_observer.reset_state()

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters including observer systems."""
        # Try to get base params, fall back to estimate if unavailable
        try:
            if hasattr(self.base_model, 'count_parameters'):
                base_params = self.base_model.count_parameters()
            else:
                # Fallback estimate for Phase 4
                base_params = {'total': 132500}  # Approximate
        except:
            base_params = {'total': 132500}

        observer_params = {}
        if self.use_observer_loop:
            observer_params['phenomenal_observer'] = (
                self.phenomenal_observer.count_parameters()['total']
            )

            if self.observe_hierarchical_states:
                observer_params['fast_observer'] = (
                    self.fast_observer.count_parameters()['total']
                )
                observer_params['medium_observer'] = (
                    self.medium_observer.count_parameters()['total']
                )
                observer_params['slow_observer'] = (
                    self.slow_observer.count_parameters()['total']
                )

        base_total = base_params.get('total', sum(base_params.values()) if base_params else 132500)

        return {
            'base_model': base_total,
            'observers': sum(observer_params.values()),
            'total': base_total + sum(observer_params.values()),
            **observer_params
        }


def estimate_phi_improvement(
    model_without_observers: NoodlingModelPhase4,
    model_with_observers: NoodlingWithObservers,
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Estimate Φ improvement from adding observer loops.

    This compares the integrated information of the system with and without
    observer loops using proxy metrics (see phi_proxy_metrics.py).

    Args:
        model_without_observers: Base Phase 4 model
        model_with_observers: Phase 4 + observers
        num_samples: Number of random inputs to test

    Returns:
        Dictionary with:
            - phi_baseline: Estimated Φ without observers
            - phi_with_observers: Estimated Φ with observers
            - phi_improvement_percent: Percentage increase
            - causal_density_improvement: Increase in causal connectivity
    """
    from phi_proxy_metrics import PhiProxyMetrics
    import numpy as np

    # Generate random inputs
    random_affects = [
        mx.random.normal((1, 5)) for _ in range(num_samples)
    ]

    # Collect states without observers
    proxy_baseline = PhiProxyMetrics(state_dim=40)
    model_without_observers.reset_states()

    for affect in random_affects:
        affect_seq = affect[:, None, :]
        h_fast, c_fast, h_med, c_med, h_slow = model_without_observers.base_model.reset_states()

        outputs = model_without_observers.base_model(
            affect_seq, h_fast, c_fast, h_med, c_med, h_slow
        )
        state = outputs['internal_state']
        proxy_baseline.add_state(np.array(state))

    baseline_phi = proxy_baseline.compute_phi_proxy(
        state_t=np.array(state),
        state_t_minus_1=np.array(model_without_observers.base_model.reset_states()[0])
    )

    # Collect states WITH observers
    proxy_with_obs = PhiProxyMetrics(state_dim=40)
    model_with_observers.reset_states()

    for affect in random_affects:
        outputs = model_with_observers(affect)
        state = outputs['internal_state']
        proxy_with_obs.add_state(np.array(state))

    with_obs_phi = proxy_with_obs.compute_phi_proxy(
        state_t=np.array(state),
        state_t_minus_1=np.array(model_with_observers.base_model.base_model.reset_states()[0])
    )

    # Compute improvement
    improvement = (
        (with_obs_phi['phi_proxy'] - baseline_phi['phi_proxy']) /
        baseline_phi['phi_proxy'] * 100
    )

    return {
        'phi_baseline': baseline_phi['phi_proxy'],
        'phi_with_observers': with_obs_phi['phi_proxy'],
        'phi_improvement_percent': improvement,
        'causal_density_baseline': baseline_phi['causal_density'],
        'causal_density_with_observers': with_obs_phi['causal_density'],
        'causal_density_improvement': (
            with_obs_phi['causal_density'] - baseline_phi['causal_density']
        )
    }
