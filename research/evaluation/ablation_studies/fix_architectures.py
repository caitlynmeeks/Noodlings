#!/usr/bin/env python3
"""
Quick script to fix the broken observer architecture in all three experiment files
"""

import re

files_to_fix = [
    'test_phase_spacing.py',
    'test_modulo_12.py'
]

# The fixed __call__ method
fixed_call_method = '''    def __call__(self, affect):
        """Forward pass with observer corrections"""
        # Ensure affect has correct shape
        if affect.ndim == 1:
            affect = affect[None, None, :]
        elif affect.ndim == 2:
            affect = affect[:, None, :]

        # Fast layer
        h_fast_seq, c_fast_seq = self.fast_lstm(affect, hidden=self.h_fast, cell=self.c_fast)
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, 16)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, 16)

        # Medium layer
        h_med_seq, c_med_seq = self.medium_lstm(self.h_fast[:, None, :], hidden=self.h_medium, cell=self.c_medium)
        self.h_medium = h_med_seq[:, -1, :].reshape(1, 16)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, 16)

        # Slow layer
        h_slow_seq = self.slow_gru(self.h_medium[:, None, :], hidden=self.h_slow)
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, 8)

        # Phenomenal state (40-D)
        phenomenal_state = mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=1)

        # Predictive processing
        predicted_state = self.predictor(phenomenal_state)

        # Observer errors (included in loss for gradient flow)
        observer_errors = []
        if self.num_observers > 0:
            for observer in self.observers:
                obs_pred = observer(predicted_state)
                observer_errors.append(mx.mean((obs_pred - phenomenal_state) ** 2))

        # Total surprise (main prediction error + observer errors)
        main_error = mx.mean((predicted_state - phenomenal_state) ** 2)
        observer_error = mx.mean(mx.array(observer_errors)) if observer_errors else mx.array(0.0)
        surprise = float(main_error + 0.1 * observer_error)

        return phenomenal_state, surprise'''

for filename in files_to_fix:
    print(f"Fixing {filename}...")

    with open(filename, 'r') as f:
        content = f.read()

    # Fix 1: Remove prev_state initialization from __init__
    content = re.sub(
        r'\s+# Store previous state for surprise calculation\s+self\.prev_state = mx\.zeros\(\(1, joint_dim\)\)',
        '',
        content
    )

    # Fix 2: Remove prev_state from reset_states
    content = re.sub(
        r'\s+self\.prev_state = mx\.zeros\(\(1, 40\)\)',
        '',
        content
    )

    # Fix 3: Replace the entire __call__ method
    # Find the __call__ method and replace it
    pattern = r'(    def __call__\(self, affect\):.*?)(    def \w+|class \w+|\ndef \w+)'

    def replacer(match):
        return fixed_call_method + '\n\n' + match.group(2)

    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    # Fix 4: Change optimizer from Adam to AdamW
    content = re.sub(
        r'optimizer = optim\.Adam\(learning_rate=1e-4\)',
        'optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-5)',
        content
    )

    with open(filename, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {filename}")

print("\n✓ All files fixed!")
