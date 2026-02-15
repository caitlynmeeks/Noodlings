# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Node Definitions - Templates for all Neural Canvas node types.
#
#   Provides factory functions to create preconfigured nodes.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.neural_canvas.node_definitions
# PURPOSE:  Node Definitions
# LAYER:    Studio / Neural Canvas
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   create_node_from_type(), get_node_icon(), get_node_color()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Optional, Dict, Any
from .neural_node import NeuralNode, NodeType, Port, DataType, WeightInfo


# Registry for user-defined custom nodes (registered via scripting API)
_custom_node_registry: Dict[str, Dict[str, Any]] = {}


# Node type definitions with default parameters and ports
NODE_DEFINITIONS: Dict[NodeType, Dict[str, Any]] = {
    NodeType.INPUT: {
        'name': 'Affect Input',
        'description': 'Network entry point (5-D affect vector)',
        'how_it_works': '''INPUT NODE - Network Entry Point

This node provides the starting data for your neural network.

HOW IT WORKS:
- Outputs a 5-dimensional affect vector
- The 5 dimensions are: Valence, Arousal, Dominance, Boredom, Sorrow
- Values typically range from -1 to +1 or 0 to 1

IN TUTORIALS:
This feeds emotional state into the network. The network
learns to process and transform these values over time.

OUTPUT: affect (5-D) - The emotional input vector''',
        'params': {'output_dim': 5},
        'inputs': {},
        'outputs': {
            'affect': Port('affect', DataType.AFFECT, shape=(5,), label='Affect (5-D)')
        },
        'weights': {},
        'color': '#2A4A2A',  # Deep forest green (darker, richer)
        'icon': '🔵'
    },

    NodeType.OUTPUT: {
        'name': 'Network Output',
        'description': 'Network exit point',
        'how_it_works': '''OUTPUT NODE - Network Exit Point

This is where processed data leaves the network.

HOW IT WORKS:
- Receives the final processed state
- Marks the end of the computational graph
- Data here is available to external systems

IN CHARMNETWORK:
The 40-D phenomenal state arrives here after
being assembled from fast/medium/slow timescales.
This state can then drive behavior, expressions,
or other downstream systems.

WHAT CONNECTS HERE:
Usually the output of STATE_CONCAT or a final
processing layer. This represents the network's
complete internal representation.''',
        'params': {},
        'inputs': {
            'state': Port('state', DataType.PHENOMENAL_STATE, shape=(40,), label='State (40-D)')
        },
        'outputs': {},
        'weights': {},
        'color': '#2A4A2A',  # Deep forest green (matches INPUT)
        'icon': '🟢'
    },

    NodeType.LSTM: {
        'name': 'LSTM Layer',
        'description': 'Long Short-Term Memory recurrent layer',
        'how_it_works': '''LSTM - Long Short-Term Memory

LSTMs remember things over time! They're the "memory" of neural networks.

THE KEY INSIGHT:
Unlike simple neurons, LSTMs have GATES that control:
- What to FORGET from memory
- What to ADD to memory
- What to OUTPUT

INPUTS:
- x: Current input (what's happening NOW)
- h: Hidden state (short-term memory)
- c: Cell state (long-term memory)

OUTPUTS:
- h_out: Updated hidden state (what to remember short-term)
- c_out: Updated cell state (what to remember long-term)

WHY LSTM?
Regular neurons are stateless - they can't remember.
LSTMs solve the "vanishing gradient problem" that
makes regular RNNs forget after a few steps.

USE CASES:
- Processing sequences (text, audio, time series)
- Tracking emotional state over conversations
- Any task requiring memory of past inputs

PARAMS:
- input_dim: Size of each input
- hidden_dim: Size of memory (bigger = more capacity)''',
        'params': {
            'input_dim': 5,
            'hidden_dim': 16,
            'dropout': 0.0,
            'bias': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input'),
            'h': Port('h', DataType.HIDDEN_STATE, required=False, label='Hidden'),
            'c': Port('c', DataType.CELL_STATE, required=False, label='Cell')
        },
        'outputs': {
            'h_out': Port('h_out', DataType.HIDDEN_STATE, label='Hidden'),
            'c_out': Port('c_out', DataType.CELL_STATE, label='Cell')
        },
        'weights': {},  # Computed dynamically based on params
        'color': '#4A2A4A',  # Deeper plum (darker, more saturated)
        'icon': '📦'
    },

    NodeType.GRU: {
        'name': 'GRU Layer',
        'description': 'Gated Recurrent Unit layer',
        'how_it_works': '''GRU - Gated Recurrent Unit

Like LSTM but simpler! Combines forget and input gates.

THE DIFFERENCE:
- LSTM has 3 gates (forget, input, output) + cell state
- GRU has 2 gates (reset, update) + NO cell state

WHEN TO USE GRU VS LSTM:
- GRU: Faster, fewer parameters, often works just as well
- LSTM: Better for very long sequences, more expressive

INPUTS:
- x: Current input
- h: Hidden state (memory)

OUTPUTS:
- h_out: Updated hidden state

THE GATES:
- Reset gate: How much past memory to forget
- Update gate: How much to update with new info

PARAMS:
- input_dim: Size of each input
- hidden_dim: Size of memory''',
        'params': {
            'input_dim': 16,
            'hidden_dim': 8,
            'dropout': 0.0,
            'bias': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input'),
            'h': Port('h', DataType.HIDDEN_STATE, required=False, label='Hidden')
        },
        'outputs': {
            'h_out': Port('h_out', DataType.HIDDEN_STATE, label='Hidden')
        },
        'weights': {},
        'color': '#4A2A4A',  # Deeper plum (matches LSTM)
        'icon': '⚙️'
    },

    NodeType.RNN: {
        'name': 'RNN Layer',
        'description': 'Simple recurrent neural network layer',
        'params': {
            'input_dim': 5,
            'hidden_dim': 16,
            'activation': 'tanh'
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True),
            'h': Port('h', DataType.HIDDEN_STATE, required=False)
        },
        'outputs': {
            'h_out': Port('h_out', DataType.HIDDEN_STATE)
        },
        'weights': {},
        'color': '#673AB7',  # Deep purple
        'icon': '🔄'
    },

    NodeType.LINEAR: {
        'name': 'Linear Layer',
        'description': 'Fully connected linear transformation',
        'how_it_works': '''LINEAR LAYER - The Building Block

This is the fundamental neuron! Every input connects to every output.

THE MATH:
  output = (input * weights) + bias

Think of it as:
  output = w1*x1 + w2*x2 + ... + wn*xn + bias

WHAT THE WEIGHTS DO:
- Each weight controls how much one input affects the output
- Positive weight = input increases output
- Negative weight = input decreases output
- The bias shifts the whole result up or down

EXAMPLE (AND gate with 2 inputs):
- weights = [1.0, 1.0], bias = -1.5
- Input [0,0]: 0+0-1.5 = -1.5
- Input [1,1]: 1+1-1.5 = +0.5

PARAMS:
- in_features: How many inputs
- out_features: How many outputs
- bias: Add a learnable offset (usually True)''',
        'params': {
            'in_features': 16,
            'out_features': 32,
            'bias': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#9C27B0',  # Purple
        'icon': '➡️'
    },

    NodeType.CONV1D: {
        'name': 'Conv1D Layer',
        'description': '1D convolutional layer',
        'params': {
            'in_channels': 1,
            'out_channels': 16,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#E91E63',  # Pink
        'icon': '🌊'
    },

    NodeType.ATTENTION: {
        'name': 'Attention',
        'description': 'Scaled dot-product attention mechanism',
        'params': {
            'embed_dim': 64,
            'dropout': 0.1
        },
        'inputs': {
            'query': Port('query', DataType.TENSOR, required=True),
            'key': Port('key', DataType.TENSOR, required=True),
            'value': Port('value', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR),
            'weights': Port('weights', DataType.TENSOR)
        },
        'weights': {},
        'color': '#F44336',  # Red
        'icon': '👁️'
    },

    NodeType.MULTI_HEAD_ATTENTION: {
        'name': 'Multi-Head Attention',
        'description': 'Multi-head attention (Transformer core)',
        'how_it_works': '''MULTI-HEAD ATTENTION - The Transformer's Secret Sauce

This is what makes GPT, Claude, and all modern AI tick!

THE KEY INSIGHT:
Instead of processing tokens one-by-one (like LSTM), attention
lets EVERY token look at EVERY other token simultaneously.

HOW IT WORKS:
1. Each token creates three vectors:
   - Query (Q): "What am I looking for?"
   - Key (K): "What do I contain?"
   - Value (V): "What information do I carry?"

2. Attention scores = Q dot K (how well does my query match your key?)

3. Softmax makes scores into probabilities (who should I pay attention to?)

4. Output = weighted sum of Values (blend information from relevant tokens)

MULTI-HEAD = MULTIPLE PERSPECTIVES:
Instead of one attention, we run several in parallel:
- Head 1 might focus on grammar
- Head 2 might focus on meaning
- Head 3 might focus on position
Then combine all perspectives!

PARAMS:
- embed_dim: Size of input embeddings
- num_heads: How many parallel attention heads
- dropout: Regularization during training

WHY THIS MATTERS:
"The cat sat on the mat because it was tired"
What does "it" refer to? Attention learns to connect "it" to "cat"
by giving high attention weight to that relationship.''',
        'params': {
            'embed_dim': 64,
            'num_heads': 4,
            'dropout': 0.0
        },
        'inputs': {
            'query': Port('query', DataType.TENSOR, required=True, label='Query (Q)'),
            'key': Port('key', DataType.TENSOR, required=True, label='Key (K)'),
            'value': Port('value', DataType.TENSOR, required=True, label='Value (V)')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output'),
            'attn_weights': Port('attn_weights', DataType.TENSOR, label='Attention Weights')
        },
        'weights': {},
        'color': '#FF5722',  # Deep orange
        'icon': '👁️'
    },

    NodeType.TRANSFORMER_BLOCK: {
        'name': 'Transformer Block',
        'description': 'Complete transformer encoder block (self-attention + FFN)',
        'how_it_works': '''TRANSFORMER BLOCK - The Production Workhorse

A complete transformer encoder layer, ready to use.

WHAT'S INSIDE (you'd need 7+ nodes to build this manually):
1. Multi-Head Self-Attention
2. Add & Normalize (residual connection + layer norm)
3. Feed-Forward Network (Linear -> GELU -> Linear)
4. Add & Normalize again

THE FLOW:
  x -> Self-Attention -> Add(x) -> LayerNorm -> FFN -> Add -> LayerNorm -> out

WHY USE THIS vs EXPLODED VIEW:
- FASTER: Optimized PyTorch implementation
- SIMPLER: One node instead of many
- PRODUCTION: This is what real models use

Use the exploded tutorial canvas to LEARN how it works,
then use this node when you need SPEED.

PARAMS:
- embed_dim: Size of embeddings (must match input)
- num_heads: Parallel attention heads
- ff_dim: Hidden size of feed-forward network (usually 4x embed_dim)
- dropout: Regularization (0.0 for inference)

TYPICAL VALUES:
- GPT-2 small: embed_dim=768, num_heads=12, ff_dim=3072
- Our tutorials: embed_dim=64, num_heads=4, ff_dim=256''',
        'params': {
            'embed_dim': 64,
            'num_heads': 4,
            'ff_dim': 256,
            'dropout': 0.0
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input Sequence')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output'),
            'attn_weights': Port('attn_weights', DataType.TENSOR, label='Attention Weights')
        },
        'weights': {},
        'color': '#E65100',  # Dark orange (transformer family)
        'icon': '🤖'
    },

    NodeType.POSITIONAL_ENCODING: {
        'name': 'Positional Encoding',
        'description': 'Add position information to embeddings',
        'how_it_works': '''POSITIONAL ENCODING - Teaching Position to Attention

THE PROBLEM:
Attention treats all tokens equally - it has no sense of ORDER!
"The cat ate the fish" and "The fish ate the cat" would look the same.

THE SOLUTION:
Add unique position signals to each token's embedding.
Token 1 gets a different "position fingerprint" than token 5.

HOW IT WORKS (Sinusoidal Encoding):
For each position, we add a pattern of sine and cosine waves:
  - Low frequencies: change slowly across positions
  - High frequencies: change rapidly

WHY SINE/COSINE?
1. Unique pattern for each position
2. Relative positions can be computed (pos 5 - pos 3 = consistent pattern)
3. Generalizes to longer sequences than seen in training

THE MATH:
  PE(pos, 2i) = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Where:
- pos = position in sequence (0, 1, 2, ...)
- i = dimension index
- d = embedding dimension

PARAMS:
- max_seq_len: Maximum sequence length to support
- embed_dim: Must match your embedding size''',
        'params': {
            'max_seq_len': 512,
            'embed_dim': 64
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Embeddings')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Embeddings + Position')
        },
        'weights': {},
        'color': '#FF9800',  # Orange (transformer family)
        'icon': '📍'
    },

    NodeType.TANH: {
        'name': 'Tanh',
        'description': 'Hyperbolic tangent activation',
        'how_it_works': '''TANH - Hyperbolic Tangent

Squashes values into the range -1 to +1.

THE MATH:
  tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

THE SHAPE:
  -infinity -> -1
  -2 -> -0.96
  -1 -> -0.76
   0 -> 0  (centered at zero!)
  +1 -> 0.76
  +2 -> 0.96
  +infinity -> +1

VS SIGMOID:
- Sigmoid: 0 to 1 (always positive)
- Tanh: -1 to +1 (centered at zero)

WHY TANH?
- Zero-centered = easier training
- Great for hidden layers in LSTMs/GRUs
- Natural for bipolar outputs

USE CASES:
- LSTM/GRU gate activations
- When you need negative values''',
        'params': {},
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#3A3A3A',  # Dark charcoal (neutral, subtle)
        'icon': '〰️'
    },

    NodeType.RELU: {
        'name': 'ReLU',
        'description': 'Rectified Linear Unit activation',
        'how_it_works': '''RELU - Rectified Linear Unit

The simplest and most popular activation function!

THE MATH:
  relu(x) = max(0, x)

That's it! If negative -> 0, if positive -> unchanged.

THE SHAPE:
  -100 -> 0
  -1 -> 0
   0 -> 0
  +1 -> 1
  +100 -> 100

WHY RELU?
- Super fast to compute
- No vanishing gradient (for positive values)
- Sparse activation = efficient networks

THE "DYING RELU" PROBLEM:
If a neuron always outputs negative values, it
gets stuck at 0 forever and stops learning.
Solutions: Leaky ReLU, GELU, or careful initialization.

USE CASES:
- Hidden layers in feedforward networks
- CNN layers
- Anywhere you need fast, simple activation''',
        'params': {},
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#3A3A3A',  # Dark charcoal
        'icon': '⚡'
    },

    NodeType.GELU: {
        'name': 'GELU',
        'description': 'Gaussian Error Linear Unit activation',
        'params': {},
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#3A3A3A',  # Dark charcoal
        'icon': '🌀'
    },

    NodeType.SIGMOID: {
        'name': 'Sigmoid',
        'description': 'Sigmoid activation function',
        'how_it_works': '''SIGMOID - The Classic Activation

Squashes ANY value into the range 0 to 1.

THE MATH:
  sigmoid(x) = 1 / (1 + e^(-x))

THE SHAPE:
  -infinity -> 0
  -5 -> 0.007
  -1 -> 0.27
   0 -> 0.5  (the midpoint!)
  +1 -> 0.73
  +5 -> 0.993
  +infinity -> 1

WHY USE IT?
- Perfect for binary decisions (0 = no, 1 = yes)
- Smooth gradient = network can learn
- Output is always positive and bounded

IN THE AND GATE:
- Linear layer outputs can be any value
- Sigmoid converts to 0-1 probability
- We then threshold at 0.5 to decide ON/OFF

HISTORICAL NOTE:
One of the oldest activation functions, inspired by
how real neurons fire (or don't). Modern networks
often prefer ReLU, but sigmoid is perfect for
binary classification outputs.''',
        'params': {},
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#3A3A3A',  # Dark charcoal
        'icon': '📈'
    },

    NodeType.SOFTMAX: {
        'name': 'Softmax',
        'description': 'Softmax activation',
        'params': {
            'dim': -1
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#8BC34A',  # Light green
        'icon': '🎲'
    },

    NodeType.LAYER_NORM: {
        'name': 'Layer Normalization',
        'description': 'Layer normalization',
        'params': {
            'normalized_shape': (16,),
            'eps': 1e-5,
            'elementwise_affine': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#4CAF50',  # Green
        'icon': '📏'
    },

    NodeType.BATCH_NORM: {
        'name': 'Batch Normalization',
        'description': 'Batch normalization',
        'params': {
            'num_features': 16,
            'eps': 1e-5,
            'momentum': 0.1
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#009688',  # Teal
        'icon': '📊'
    },

    NodeType.DROPOUT: {
        'name': 'Dropout',
        'description': 'Dropout regularization',
        'params': {
            'p': 0.5
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#00BCD4',  # Cyan
        'icon': '💧'
    },

    NodeType.STATE_CONCAT: {
        'name': 'State Concatenation',
        'description': 'Concatenate multiple hidden states into phenomenal state',
        'how_it_works': '''STATE CONCAT - Assemble Phenomenal State

Combines hidden states from multiple timescales into one.

THE ARCHITECTURE:
CharmNetwork processes affect at three speeds:
- Fast LSTM (16-D): Immediate reactions (seconds)
- Medium LSTM (16-D): Short-term mood (minutes)
- Slow GRU (8-D): Long-term disposition (hours/days)

THE CONCATENATION:
  Fast [16] + Medium [16] + Slow [8] = State [40]

WHY THIS MATTERS:
The combined 40-D "phenomenal state" captures
the full temporal richness of emotional experience.
Like how your mood right now is shaped by:
- What just happened (fast)
- How your day has been (medium)
- Your general life situation (slow)

OUTPUT:
A single 40-D vector encoding all timescales.''',
        'params': {},
        'inputs': {
            'fast': Port('fast', DataType.HIDDEN_STATE, shape=(16,), label='Fast (16-D)'),
            'medium': Port('medium', DataType.HIDDEN_STATE, shape=(16,), label='Medium (16-D)'),
            'slow': Port('slow', DataType.HIDDEN_STATE, shape=(8,), label='Slow (8-D)')
        },
        'outputs': {
            'state': Port('state', DataType.PHENOMENAL_STATE, shape=(40,), label='Phenomenal State')
        },
        'weights': {},
        'color': '#2A4A4A',  # Deeper teal (darker, sophisticated)
        'icon': '➕'
    },

    NodeType.STATE_SPLIT: {
        'name': 'State Split',
        'description': 'Split phenomenal state into components',
        'params': {},
        'inputs': {
            'state': Port('state', DataType.PHENOMENAL_STATE, shape=(40,))
        },
        'outputs': {
            'fast': Port('fast', DataType.HIDDEN_STATE, shape=(16,)),
            'medium': Port('medium', DataType.HIDDEN_STATE, shape=(16,)),
            'slow': Port('slow', DataType.HIDDEN_STATE, shape=(8,))
        },
        'weights': {},
        'color': '#03A9F4',  # Light blue
        'icon': '➖'
    },

    NodeType.AFFECT_HEAD: {
        'name': 'Affect Head',
        'description': 'Maps phenomenal state to 5-D continuous affect',
        'how_it_works': '''AFFECT HEAD - Emotion Decoder

Decodes the phenomenal state into 5 continuous affect dimensions.

THE 5 OUTPUTS:
- Valence (-1 to +1): Unpleasant <-> Pleasant
- Arousal (0 to 1): Calm <-> Excited
- Dominance (0 to 1): Submissive <-> Dominant
- Sorrow (0 to 1): Contentment <-> Grief
- Boredom (0 to 1): Engaged <-> Disinterested

THE ARCHITECTURE:
  State [40-D] -> Linear [32] -> ReLU -> Linear [5]

WHY NOT DISCRETE EMOTIONS?
Discrete labels (happy, sad, angry) are human shortcuts.
Continuous affect captures the rich gradient of experience.
"Bittersweet" isn't a label - it's a point in affect space.

HOW IT'S TRAINED:
The network learns to predict these values from
sequences of emotional experiences. Over time,
the phenomenal state naturally organizes to
encode affect in a way the head can decode.

PARAMS:
- state_dim: Input dimensions (40)
- affect_dim: Output dimensions (5)
- hidden_dim: Intermediate layer size (32)''',
        'params': {
            'state_dim': 40,
            'affect_dim': 5,
            'hidden_dim': 32
        },
        'inputs': {
            'state': Port('state', DataType.PHENOMENAL_STATE, shape=(40,), label='State (40-D)')
        },
        'outputs': {
            'valence': Port('valence', DataType.SCALAR, label='Valence'),
            'arousal': Port('arousal', DataType.SCALAR, label='Arousal'),
            'fear': Port('fear', DataType.SCALAR, label='Fear'),
            'sorrow': Port('sorrow', DataType.SCALAR, label='Sorrow'),
            'boredom': Port('boredom', DataType.SCALAR, label='Boredom')
        },
        'weights': {},
        'color': '#4A3A2A',  # Deeper tobacco brown (darker, warmer)
        'icon': '🎭'
    },

    NodeType.QUANTUM_MICROTUBULE: {
        'name': 'Quantum Microtubule',
        'description': 'Penrose-Hameroff quantum consciousness layer',
        'params': {
            'input_dim': 16,
            'hidden_dim': 16,
            'collapse_threshold': 0.5,
            'coherence_time': 10,
            'entanglement_range': 3,
            'noise_scale': 0.1,
            'use_collapse': True,
            'use_entanglement': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True),
            'mt_state': Port('mt_state', DataType.HIDDEN_STATE, required=False)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR),
            'new_mt_state': Port('new_mt_state', DataType.HIDDEN_STATE)
        },
        'weights': {},
        'color': '#4A2A3A',  # Deeper burgundy (darker, mysterious)
        'icon': '⚛️'
    },

    NodeType.IBM_QUANTUM: {
        'name': 'IBM Quantum',
        'description': 'Real quantum computation via IBM Quantum cloud',
        'params': {
            'num_qubits': 4,
            'shots': 100,
            'backend': 'simulator',
            'entanglement_type': 'full'
        },
        'inputs': {
            'classical_state': Port('classical_state', DataType.TENSOR, required=True)
        },
        'outputs': {
            'quantum_result': Port('quantum_result', DataType.TENSOR)
        },
        'weights': {},
        'color': '#4A2A3A',  # Deeper burgundy (matches quantum family)
        'icon': '🔮'
    },

    NodeType.ENTROPY_INJECTION: {
        'name': 'Entropy Injection',
        'description': 'Inject true quantum randomness (TrueRNG)',
        'params': {
            'noise_scale': 0.1,
            'use_hardware_rng': True,
            'distribution': 'avalanche'
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#FF5722',  # Deep orange
        'icon': '🎲'
    },

    NodeType.CHECKPOINT: {
        'name': 'Checkpoint',
        'description': 'Trained weight checkpoint (.npz file)',
        'params': {
            'checkpoint_path': '',
            'total_params': 0,
            'trained_epochs': 0,
            'final_loss': 0.0
        },
        'inputs': {},  # No inputs - this is a data source
        'outputs': {
            'weights': Port('weights', DataType.TENSOR)  # Provides weights to network
        },
        'weights': {},
        'color': '#795548',  # Brown (asset color)
        'icon': '💾'
    },

    # Tutorial/Interactive nodes
    NodeType.NUMBER_INPUT: {
        'name': 'Number Input',
        'description': 'Interactive scalar input with slider (0.0 to 1.0)',
        'how_it_works': '''NUMBER INPUT - Interactive Slider

A hands-on input for learning! Drag the slider to change the value.

HOW TO USE:
- Drag the slider left/right to change the value
- The value is sent to connected nodes
- Perfect for testing how networks respond to inputs

IN TUTORIALS:
This lets you manually set input values (like 0 or 1)
to test logic gates and see how the network responds.

PARAMS:
- value: Current slider position
- min_value: Minimum value (usually 0.0)
- max_value: Maximum value (usually 1.0)
- step: How much each drag increments (1.0 for binary)

TIP:
Set step=1.0 for binary (0/1) inputs like logic gates.
Set step=0.1 for finer control over continuous values.''',
        'params': {
            'value': 0.5,  # Current value
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.1
        },
        'inputs': {},  # No inputs - this is a source node
        'outputs': {
            'value': Port('value', DataType.SCALAR, shape=(1,), label='Value')
        },
        'weights': {},
        'color': '#4A6A4A',  # Forest green (input family)
        'icon': '🎚️'
    },

    NodeType.THRESHOLD_OUTPUT: {
        'name': 'Threshold Output',
        'description': 'ON/OFF display based on threshold comparison',
        'how_it_works': '''THRESHOLD OUTPUT - Visual Binary Display

Shows ON or OFF based on whether the input exceeds a threshold.

THE LOGIC:
  if (input >= threshold): show "ON"  (green)
  else: show "OFF" (dim)

IN TUTORIALS:
This visualizes the final decision of your network.
- Green ON = network says "yes"
- Dim OFF = network says "no"

PARAMS:
- threshold: The cutoff value (default 0.5)
- show_value: Also display the numeric value

EXAMPLE WITH AND GATE:
- After sigmoid, values range 0-1
- Threshold at 0.5 converts to binary decision
- Input [1,1] -> sigmoid(0.5) = 0.62 -> ON
- Input [0,0] -> sigmoid(-1.5) = 0.18 -> OFF

This is how neural networks make decisions!''',
        'params': {
            'threshold': 0.5,  # Activation threshold
            'show_value': True  # Show numeric value alongside ON/OFF
        },
        'inputs': {
            'value': Port('value', DataType.SCALAR, shape=(1,), required=True, label='Value')
        },
        'outputs': {},  # No outputs - this is a sink node
        'weights': {},
        'color': '#6A4A4A',  # Muted red-brown (output family)
        'icon': '💡'
    },

    NodeType.CONCAT: {
        'name': 'Concat',
        'description': 'Concatenate two inputs into one tensor',
        'how_it_works': '''CONCAT - Combine Inputs

Joins two tensors into one longer tensor.

THE OPERATION:
  Input A: [1, 2]
  Input B: [3, 4]
  Output:  [1, 2, 3, 4]

WHY CONCAT?
Networks often need multiple inputs combined:
- Two separate signals -> one input to a layer
- Features from different sources merged

IN THE AND GATE:
- Input A is a scalar [0] or [1]
- Input B is a scalar [0] or [1]
- Concat produces [A, B] - a 2-element vector
- This feeds into the Linear layer

INPUTS:
- a: First tensor to concatenate
- b: Second tensor to concatenate

OUTPUT:
- out: Combined tensor [a... + b...]

The order matters! A comes first, then B.''',
        'params': {},
        'inputs': {
            'a': Port('a', DataType.TENSOR, required=True, label='Input A'),
            'b': Port('b', DataType.TENSOR, required=True, label='Input B')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Combined')
        },
        'weights': {},
        'color': '#4A4A6A',  # Muted blue (utility)
        'icon': '⊕'
    },

    NodeType.STACK: {
        'name': 'Stack (Sequence)',
        'description': 'Stack tensors along sequence dimension for transformers',
        'how_it_works': '''STACK - Build a Sequence

Stacks tensors along a NEW sequence dimension.
This is what transformers need!

THE DIFFERENCE FROM CONCAT:
  CONCAT: [1,2] + [3,4] = [1,2,3,4]  (one long vector)
  STACK:  [1,2] + [3,4] = [[1,2], [3,4]]  (2 tokens of 2D each)

WHY STACK FOR TRANSFORMERS?
Attention operates on SEQUENCES of tokens:
- Input shape: (batch, seq_len, embed_dim)
- Each token is a separate vector
- Attention compares tokens TO EACH OTHER

EXAMPLE:
  Token 1 embedding: [0.1, 0.2, ..., 0.16]  (16D)
  Token 2 embedding: [0.3, 0.4, ..., 0.16]  (16D)
  Token 3 embedding: [0.5, 0.6, ..., 0.16]  (16D)

  STACK output: (1, 3, 16)  <- 3 tokens of 16D each
  This feeds properly into MULTI_HEAD_ATTENTION!

USE THIS when building sequences for transformers.
Use CONCAT when you just want to merge features.''',
        'params': {},
        'inputs': {
            'a': Port('a', DataType.TENSOR, required=True, label='Token A'),
            'b': Port('b', DataType.TENSOR, required=True, label='Token B')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Sequence')
        },
        'weights': {},
        'color': '#5A4A6A',  # Purple-blue (transformer utility)
        'icon': '≡'
    },

    NodeType.PULSE_INPUT: {
        'name': 'Pulse Input',
        'description': 'Button that sends 1.0 on click, 0.0 otherwise. Click to pulse!',
        'params': {
            'pulse_active': False,  # True when button just clicked
            'pulse_duration': 1,  # How many steps the pulse lasts
        },
        'inputs': {},  # No inputs - this is a source node
        'outputs': {
            'pulse': Port('pulse', DataType.SCALAR, shape=(1,), label='Pulse')
        },
        'weights': {},
        'color': '#6A4A6A',  # Purple-ish (input family variant)
        'icon': '⚡'
    },

    NodeType.OUTPUT_CHART: {
        'name': 'Output Chart',
        'description': 'Time series line chart showing values over time',
        'params': {
            'history_length': 50,  # How many steps to show
            'min_value': 0.0,  # Y-axis minimum
            'max_value': 1.0,  # Y-axis maximum
            'auto_scale': True,  # Auto-adjust Y axis
        },
        'inputs': {
            'value': Port('value', DataType.SCALAR, shape=(1,), required=True, label='Value')
        },
        'outputs': {},  # No outputs - this is a sink node
        'weights': {},
        'color': '#4A6A6A',  # Teal-ish (output family variant)
        'icon': '📈'
    },

    NodeType.COUNTER_OUTPUT: {
        'name': 'Counter Output',
        'description': 'Displays a rounded integer count',
        'params': {
            'scale': 1.0,  # Multiply value by this before display
            'offset': 0.0,  # Add this after scaling
        },
        'inputs': {
            'value': Port('value', DataType.SCALAR, shape=(1,), required=True, label='Value')
        },
        'outputs': {},  # No outputs - this is a sink node
        'weights': {},
        'color': '#6A6A4A',  # Olive-ish (output family variant)
        'icon': '🔢'
    },

    NodeType.TEXT_INPUT: {
        'name': 'Text Input',
        'description': 'Single line text entry for testing affect responses',
        'params': {
            'text': '',  # Current text
            'max_length': 200,  # Maximum characters
        },
        'inputs': {},  # No inputs - this is a source node
        'outputs': {
            'text': Port('text', DataType.TENSOR, label='Text')
        },
        'weights': {},
        'color': '#4A5A4A',  # Muted green (input family)
        'icon': '📝'
    },

    NodeType.SIMPLE_EMBED: {
        'name': 'Simple Embed',
        'description': 'Converts text to affect-like vector using keyword heuristics',
        'params': {
            'output_dim': 8,  # Embedding dimension
        },
        'inputs': {
            'text': Port('text', DataType.TENSOR, required=True, label='Text')
        },
        'outputs': {
            'embedding': Port('embedding', DataType.TENSOR, label='Embedding')
        },
        'weights': {},
        'color': '#5A4A5A',  # Muted purple (transform)
        'icon': '🔤'
    },

    NodeType.AFFECT_VIS: {
        'name': 'Affect Visualizer',
        'description': 'Pentagon/radar chart showing 5D affect (valence, arousal, dominance, sorrow, boredom)',
        'params': {
            'show_labels': True,  # Show dimension labels
            'show_values': True,  # Show numeric values
        },
        'inputs': {
            'affect': Port('affect', DataType.AFFECT, shape=(5,), required=True, label='Affect (5-D)')
        },
        'outputs': {},  # No outputs - this is a sink node
        'weights': {},
        'color': '#5A5A4A',  # Warm gray (output family)
        'icon': '🎭'
    },

    NodeType.ATTENTION_VIS: {
        'name': 'Attention Visualizer',
        'description': 'Heatmap showing attention weights between tokens',
        'how_it_works': '''ATTENTION VISUALIZER - See What the Model Sees

Shows which tokens are paying attention to which other tokens.

THE HEATMAP:
- Rows = Query tokens (who's asking)
- Columns = Key tokens (who's being asked about)
- Brightness = Attention weight (how much attention)

INTERPRETING IT:
- Bright diagonal = tokens attending to themselves (common)
- Bright off-diagonal = interesting relationships!
- "it" attending to "cat" = pronoun resolution
- "sat" attending to "cat" = subject-verb agreement

WHY THIS MATTERS:
This visualization reveals what the model has LEARNED.
You can literally see it understanding language structure!

USE IN TUTORIALS:
1. Feed different sentences
2. Watch attention patterns change
3. See how the model "thinks"

PARAMS:
- show_values: Display numeric weights on cells
- colormap: Color scheme (viridis, hot, cool)''',
        'params': {
            'show_values': False,  # Show numeric values on cells
            'colormap': 'viridis',  # viridis, hot, cool
            'token_labels': [],  # Labels for axes
        },
        'inputs': {
            'weights': Port('weights', DataType.TENSOR, required=True, label='Attention Weights')
        },
        'outputs': {},  # No outputs - this is a sink node
        'weights': {},
        'color': '#FF7043',  # Light deep orange (transformer family)
        'icon': '🔥'
    },

    NodeType.TOKEN_INPUT: {
        'name': 'Token Input',
        'description': 'Select a token from vocabulary (for generation tutorials)',
        'params': {
            'token_id': 0,  # Current token ID
            'vocab_size': 100,  # Size of vocabulary
            'vocab': ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast', 'happy', 'sad'],  # Sample vocab
        },
        'inputs': {},  # No inputs - this is a source node
        'outputs': {
            'token_id': Port('token_id', DataType.SCALAR, shape=(1,), label='Token ID')
        },
        'weights': {},
        'color': '#4A5A5A',  # Teal-ish (input family)
        'icon': '🔤'
    },

    NodeType.EMBEDDING: {
        'name': 'Embedding',
        'description': 'Token embedding lookup table - converts token IDs to dense vectors',
        'params': {
            'vocab_size': 100,  # Number of tokens
            'embed_dim': 16,  # Embedding dimension
        },
        'inputs': {
            'token_id': Port('token_id', DataType.SCALAR, required=True, label='Token ID')
        },
        'outputs': {
            'embedding': Port('embedding', DataType.TENSOR, label='Embedding')
        },
        'weights': {
            'weight': {'shape': None, 'path': None, 'trainable': True}
        },
        'color': '#5A5A6A',  # Blue-gray (transform)
        'icon': '📊'
    },

    NodeType.SAMPLING: {
        'name': 'Sampling',
        'description': 'Temperature-controlled sampling from logits distribution',
        'params': {
            'temperature': 1.0,  # Higher = more random, lower = more deterministic
            'top_k': 0,  # If > 0, only sample from top k tokens
            'top_p': 1.0,  # Nucleus sampling threshold
        },
        'inputs': {
            'logits': Port('logits', DataType.TENSOR, required=True, label='Logits')
        },
        'outputs': {
            'token_id': Port('token_id', DataType.SCALAR, label='Sampled Token'),
            'probs': Port('probs', DataType.TENSOR, label='Probabilities')
        },
        'weights': {},
        'color': '#6A5A5A',  # Warm gray (stochastic)
        'icon': '🎲'
    },

    NodeType.TOKEN_OUTPUT: {
        'name': 'Token Output',
        'description': 'Displays sampled token as text from vocabulary',
        'params': {
            'vocab': ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast', 'happy', 'sad'],  # Sample vocab
        },
        'inputs': {
            'token_id': Port('token_id', DataType.SCALAR, required=True, label='Token ID')
        },
        'outputs': {},  # No outputs - this is a sink node
        'weights': {},
        'color': '#5A6A5A',  # Green-gray (output)
        'icon': '💬'
    },

    NodeType.PROB_VIS: {
        'name': 'Probability Visualizer',
        'description': 'Bar chart showing probability distribution over tokens',
        'params': {
            'top_k': 10,  # Show top k probabilities
            'vocab': ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast', 'happy', 'sad'],  # Labels
        },
        'inputs': {
            'probs': Port('probs', DataType.TENSOR, required=True, label='Probabilities')
        },
        'outputs': {},  # No outputs - this is a sink node
        'weights': {},
        'color': '#5A5A5A',  # Neutral gray (output)
        'icon': '📊'
    },

    # Annotation nodes
    NodeType.COMMENT: {
        'name': 'Comment',
        'description': 'Floating explanatory text (functionally inert)',
        'params': {
            'text': 'Add your comment here...',  # The comment text
            'width': 320,  # Display width in pixels
            'height': None,  # Auto-calculated if None, or explicit height
            'show_on_start': False,  # If True, auto-popup when file loads (only one per canvas)
        },
        'inputs': {},  # No inputs - purely decorative
        'outputs': {},  # No outputs - purely decorative
        'weights': {},
        'color': '#5c4a3d',  # Coffee brown (annotation)
        'icon': '💬'
    },

    # Math/Signal nodes
    NodeType.SINE: {
        'name': 'Sine',
        'description': 'Sine wave: output = sin(input * frequency * 2pi)',
        'how_it_works': '''SINE - Smooth Wave Generator

Creates smooth oscillating values from any input.

THE MATH:
  out = amplitude * sin(x * frequency * 2 * pi + phase)

THE WAVE SHAPE:
  input 0.00 -> 0.0   (start at zero)
  input 0.25 -> 1.0   (peak)
  input 0.50 -> 0.0   (back to zero)
  input 0.75 -> -1.0  (trough)
  input 1.00 -> 0.0   (full cycle)

PARAMS:
- frequency: How many cycles per unit input (1.0 = 1 cycle)
- amplitude: How tall the waves are (1.0 = -1 to +1)
- phase: Shifts the wave (0.25 = start at peak)

USE CASES:
- Animations (connect to TIME node!)
- Audio synthesis
- Smooth periodic signals
- Creating plasma/demoscene effects

IN THE DEMOSCENE:
Time feeds into Sine to create smooth animation.
Different frequencies create layered effects.''',
        'params': {
            'frequency': 1.0,  # Oscillation frequency
            'amplitude': 1.0,  # Output scale
            'phase': 0.0,  # Phase offset (0-1)
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#2E7D32',  # Green (math)
        'icon': '~'
    },

    NodeType.NOISE: {
        'name': 'Noise',
        'description': 'Random noise generator (uniform, gaussian, or perlin)',
        'how_it_works': '''NOISE - Random Value Generator

Outputs random values for organic variation.

NOISE TYPES:
- uniform: Even distribution between -scale and +scale
- gaussian: Bell curve centered at 0
- perlin: Smooth coherent noise (coming soon)

THE OUTPUT:
Each execution produces a new random value.
Great for adding organic variation to animations.

PARAMS:
- noise_type: Which distribution to use
- scale: How large the random values can be
- seed: For reproducible randomness (0 = truly random)

USE CASES:
- Adding chaos to animations
- Simulating natural variation
- Breaking up repetitive patterns
- Audio texture

IN THE DEMOSCENE:
Small amounts of noise add organic feel to
otherwise mathematical patterns.''',
        'params': {
            'noise_type': 'uniform',  # uniform, gaussian, perlin
            'scale': 1.0,  # Output scale
            'seed': 0,  # Random seed (0 = random each run)
        },
        'inputs': {
            'shape': Port('shape', DataType.TENSOR, required=False, label='Shape Hint')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Noise')
        },
        'weights': {},
        'color': '#5D4037',  # Brown (random/entropy)
        'icon': '?'
    },

    NodeType.TIME: {
        'name': 'Time',
        'description': 'Outputs elapsed time in seconds (for animations)',
        'how_it_works': '''TIME - Animation Clock

Outputs how many seconds have passed since start.

THE OUTPUT:
- Starts at 0.0 when canvas begins executing
- Increases continuously (0.0, 0.1, 0.2, ...)
- Never stops (unless you set loop_duration)

PARAMS:
- scale: Time multiplier (2.0 = twice as fast)
- loop_duration: Reset time every N seconds (0 = no loop)

TYPICAL PATTERNS:
- TIME -> SINE = smooth oscillation
- TIME -> shader uniform = animation
- TIME * 2 = double speed animation

IN THE DEMOSCENE:
Time is the heartbeat of the animation.
Everything flows from this single value.

DEMOSCENE HISTORY:
In classic demos, time was often derived from
music BPM to sync visuals with sound.''',
        'params': {
            'scale': 1.0,  # Time multiplier
            'loop_duration': 0.0,  # If > 0, loops time at this duration
        },
        'inputs': {},  # No inputs - source node
        'outputs': {
            'time': Port('time', DataType.SCALAR, shape=(1,), label='Time')
        },
        'weights': {},
        'color': '#1565C0',  # Blue (time)
        'icon': 't'
    },

    NodeType.MULTIPLY: {
        'name': 'Multiply',
        'description': 'Element-wise multiplication: out = a * b',
        'how_it_works': '''MULTIPLY - Scale Values

Multiplies two values together element-wise.

THE MATH:
  out = a * b

EXAMPLES:
  2 * 3 = 6
  [1, 2] * [3, 4] = [3, 8]
  signal * 0.5 = signal at half strength

USE CASES:
- Scaling signals (multiply by constant)
- Modulation (multiply two signals together)
- Ring modulation in audio
- Amplitude control''',
        'params': {},
        'inputs': {
            'a': Port('a', DataType.TENSOR, required=True, label='A'),
            'b': Port('b', DataType.TENSOR, required=True, label='B')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='A * B')
        },
        'weights': {},
        'color': '#4A4A4A',  # Gray (math operator)
        'icon': 'x'
    },

    NodeType.ADD: {
        'name': 'Add',
        'description': 'Element-wise addition: out = a + b',
        'how_it_works': '''ADD - Combine Values

Adds two values together element-wise.

THE MATH:
  out = a + b

EXAMPLES:
  2 + 3 = 5
  [1, 2] + [3, 4] = [4, 6]
  slow_wave + fast_wave = combined signal

USE CASES:
- Mixing signals together
- Adding offset/bias to values
- Layering multiple waves
- Combining features

IN THE DEMOSCENE:
Add is used to layer multiple sine waves,
creating complex patterns from simple parts.

SIGNAL MIXING:
When you add waves together, they interfere:
- Same phase = amplify
- Opposite phase = cancel''',
        'params': {},
        'inputs': {
            'a': Port('a', DataType.TENSOR, required=True, label='A'),
            'b': Port('b', DataType.TENSOR, required=True, label='B')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='A + B')
        },
        'weights': {},
        'color': '#4A4A4A',  # Gray (math operator)
        'icon': '+'
    },

    # Audio nodes
    NodeType.OSCILLATOR: {
        'name': 'Oscillator',
        'description': 'Audio waveform generator (sine, saw, square, triangle)',
        'params': {
            'waveform': 'sine',  # sine, saw, square, triangle
            'frequency': 440.0,  # Hz (A4 = 440)
            'sample_rate': 44100,  # Samples per second
            'duration': 0.1,  # Seconds of audio per execution
        },
        'inputs': {
            'freq_mod': Port('freq_mod', DataType.SCALAR, required=False, label='Freq Mod'),
            'amp_mod': Port('amp_mod', DataType.SCALAR, required=False, label='Amp Mod')
        },
        'outputs': {
            'audio': Port('audio', DataType.TENSOR, label='Audio Buffer')
        },
        'weights': {},
        'color': '#7B1FA2',  # Purple (audio)
        'icon': 'osc'
    },

    NodeType.AUDIO_OUTPUT: {
        'name': 'Audio Output',
        'description': 'Play audio buffer to speakers',
        'params': {
            'volume': 0.5,  # 0-1 output volume
            'channel': 'mono',  # mono, stereo
        },
        'inputs': {
            'audio': Port('audio', DataType.TENSOR, required=True, label='Audio')
        },
        'outputs': {},  # No outputs - sink node
        'weights': {},
        'color': '#C62828',  # Red (output/speaker)
        'icon': 'spk'
    },

    NodeType.AUDIO_FILE: {
        'name': 'Audio File',
        'description': 'Load audio file (wav/mp3) as playable buffer',
        'how_it_works': '''AUDIO FILE - Load Sound Effects

Loads an audio file from disk for playback.

SUPPORTED FORMATS:
- WAV (recommended - no compression)
- MP3
- OGG

PARAMS:
- file_path: Path to audio file
- loop: Whether to loop playback

OUTPUT:
- audio: Audio buffer ready for playback

USE WITH:
- AUDIO_TRIGGER: Play when condition met
- AUDIO_OUTPUT: Direct playback control''',
        'params': {
            'file_path': '',  # Path to audio file
            'loop': False,
        },
        'inputs': {},
        'outputs': {
            'audio': Port('audio', DataType.TENSOR, label='Audio Buffer')
        },
        'weights': {},
        'color': '#C62828',  # Red (audio)
        'icon': 'wav'
    },

    NodeType.AUDIO_TRIGGER: {
        'name': 'Audio Trigger',
        'description': 'Play audio when input crosses threshold',
        'how_it_works': '''AUDIO TRIGGER - Conditional Sound Playback

Plays a sound when a condition is met.

HOW IT WORKS:
1. Monitors the 'trigger' input value
2. When value crosses threshold (going UP), plays audio_on
3. When value crosses threshold (going DOWN), plays audio_off

PERFECT FOR:
- AND gate tutorials (ding when ON, buzz when OFF)
- Sound effects triggered by neural network output
- Interactive feedback sounds

PARAMS:
- threshold: Value at which to trigger (default 0.5)
- audio_on_path: Sound to play when triggered ON
- audio_off_path: Sound to play when triggered OFF
- volume: Playback volume (0-1)

INPUTS:
- trigger: Value to monitor
- audio_on: Alternative audio input for ON sound
- audio_off: Alternative audio input for OFF sound''',
        'params': {
            'threshold': 0.5,
            'audio_on_path': '',  # Path to "on" sound
            'audio_off_path': '',  # Path to "off" sound
            'volume': 0.7,
        },
        'inputs': {
            'trigger': Port('trigger', DataType.SCALAR, required=True, label='Trigger Value'),
            'audio_on': Port('audio_on', DataType.TENSOR, required=False, label='Audio (ON)'),
            'audio_off': Port('audio_off', DataType.TENSOR, required=False, label='Audio (OFF)'),
        },
        'outputs': {},  # No outputs - sink node
        'weights': {},
        'color': '#C62828',  # Red (audio)
        'icon': 'trg'
    },

    # Scripting nodes
    NodeType.SCRIPTED_NODE: {
        'name': 'Script Node',
        'description': 'Custom JavaScript logic for user-defined behavior',
        'how_it_works': '''SCRIPT NODE - Custom Logic

Write your own node behavior in JavaScript!

HOW IT WORKS:
The 'script' param contains JavaScript that runs each execution.
Your script receives inputs and must return outputs.

SCRIPT TEMPLATE:
  // inputs.a, inputs.b, etc. are your input values
  // Return an object with output values
  return {
      out: inputs.a + inputs.b,
      flag: inputs.a > 0.5
  };

AVAILABLE IN SCRIPT:
- inputs: Object with all input port values
- params: Object with node params (excluding script)
- Math: Standard JavaScript Math object

INPUTS/OUTPUTS:
Configure num_inputs and num_outputs params.
Inputs are named: a, b, c, d...
Outputs are named: out, out2, out3, out4...

EXAMPLE - Custom activation:
  var x = inputs.a;
  var leaky = x > 0 ? x : 0.01 * x;
  return { out: leaky };

EXAMPLE - Conditional routing:
  if (inputs.a > params.threshold) {
      return { out: inputs.b, flag: 1.0 };
  } else {
      return { out: inputs.c, flag: 0.0 };
  }''',
        'params': {
            'script': '''// Custom node logic
// inputs.a, inputs.b, etc. are your inputs
// Return an object with output values
return {
    out: inputs.a * 2.0
};''',
            'num_inputs': 1,  # Number of input ports (1-4)
            'num_outputs': 1,  # Number of output ports (1-4)
            'input_names': 'a',  # Comma-separated custom names
            'output_names': 'out',  # Comma-separated custom names
        },
        'inputs': {
            'a': Port('a', DataType.TENSOR, required=False, label='Input A'),
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#FF8F00',  # Amber (scripting)
        'icon': 'js'
    },

    # Visual nodes
    NodeType.SHADER_VIS: {
        'name': 'Shader Display',
        'description': 'GLSL fragment shader visualization with neural network inputs as uniforms',
        'how_it_works': '''SHADER DISPLAY - GLSL Visualization

Renders real-time graphics driven by neural network signals!

WHAT IT DOES:
- Runs a GLSL fragment shader every frame
- Your neural network values become "uniforms"
- Creates stunning visual effects

BUILT-IN UNIFORMS:
- time: Elapsed seconds (auto)
- resolution: Width/height in pixels (auto)
- u_value: YOUR neural network input!

THE DEMOSCENE CONNECTION:
In the 80s/90s, hackers created stunning visuals
with minimal code. This is that tradition continued!

Classic effects you can create:
- Plasma (sine waves + colors)
- Tunnel (polar coordinates)
- Fire (noise + feedback)
- Fractals (iteration)

HOW TO USE:
1. Connect a signal to the 'value' input
2. Edit shader_code in the Inspector
3. Use u_value in your shader math!

EXAMPLE SHADER:
  float v = sin(uv.x * 10.0 + u_value * 6.28);
  gl_FragColor = vec4(v, 0, 0, 1);

This creates red stripes that shift based on
your neural network output!''',
        'params': {
            'shader_code': '''#version 330 core
out vec4 FragColor;
uniform float time;
uniform vec2 resolution;
uniform float u_value;  // from neural network

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    float v = sin(uv.x * 10.0 + time + u_value * 3.14159);
    FragColor = vec4(v * 0.5 + 0.5, 0.0, 0.0, 1.0);
}''',
            'width': 256,
            'height': 256,
            'preset': 'custom',  # custom, plasma, tunnel, starfield
        },
        'inputs': {
            'value': Port('value', DataType.SCALAR, required=False, label='Uniform Value')
        },
        'outputs': {},  # No outputs - display sink
        'weights': {},
        'color': '#00695C',  # Teal (shader/visual)
        'icon': 'fx'
    },
}


def create_node_from_type(node_type: NodeType, name: Optional[str] = None) -> NeuralNode:
    """
    Create a node from its type definition.

    Args:
        node_type: Type of node to create
        name: Override default name (optional)

    Returns:
        Preconfigured NeuralNode instance
    """
    if node_type not in NODE_DEFINITIONS:
        raise ValueError(f"Unknown node type: {node_type}")

    definition = NODE_DEFINITIONS[node_type]

    node = NeuralNode.create_with_uuid(
        node_type=node_type,
        name=name or definition['name']
    )

    # Copy parameters
    node.params = definition['params'].copy()

    # Copy ports (deep copy to avoid shared references)
    node.inputs = {
        name: Port(
            name=port.name,
            data_type=port.data_type,
            shape=port.shape,
            required=port.required,
            label=port.label  # Preserve label
        )
        for name, port in definition['inputs'].items()
    }

    node.outputs = {
        name: Port(
            name=port.name,
            data_type=port.data_type,
            shape=port.shape,
            label=port.label  # Preserve label
        )
        for name, port in definition['outputs'].items()
    }

    # Copy metadata
    node.color = definition.get('color')
    node.description = definition['description']

    # Compute weights if applicable (for LSTM, GRU, Linear, etc.)
    _compute_weights_for_node(node)

    return node


def _compute_weights_for_node(node: NeuralNode):
    """
    Compute weight shapes for a node based on its type and parameters.

    Modifies node.weights in place.
    """
    if node.type == NodeType.LSTM:
        input_dim = node.params.get('input_dim', 5)
        hidden_dim = node.params.get('hidden_dim', 16)
        has_bias = node.params.get('bias', True)

        # LSTM has 4 gates (input, forget, cell, output)
        # Weight shape: (4 * hidden_dim, input_dim) and (4 * hidden_dim, hidden_dim)
        node.weights['weight_ih'] = WeightInfo('weight_ih', (4 * hidden_dim, input_dim))
        node.weights['weight_hh'] = WeightInfo('weight_hh', (4 * hidden_dim, hidden_dim))

        if has_bias:
            node.weights['bias_ih'] = WeightInfo('bias_ih', (4 * hidden_dim,))
            node.weights['bias_hh'] = WeightInfo('bias_hh', (4 * hidden_dim,))

    elif node.type == NodeType.GRU:
        input_dim = node.params.get('input_dim', 16)
        hidden_dim = node.params.get('hidden_dim', 8)
        has_bias = node.params.get('bias', True)

        # GRU has 3 gates (reset, update, new)
        node.weights['weight_ih'] = WeightInfo('weight_ih', (3 * hidden_dim, input_dim))
        node.weights['weight_hh'] = WeightInfo('weight_hh', (3 * hidden_dim, hidden_dim))

        if has_bias:
            node.weights['bias_ih'] = WeightInfo('bias_ih', (3 * hidden_dim,))
            node.weights['bias_hh'] = WeightInfo('bias_hh', (3 * hidden_dim,))

    elif node.type == NodeType.LINEAR:
        in_features = node.params.get('in_features', 16)
        out_features = node.params.get('out_features', 32)
        has_bias = node.params.get('bias', True)

        node.weights['weight'] = WeightInfo('weight', (out_features, in_features))

        if has_bias:
            node.weights['bias'] = WeightInfo('bias', (out_features,))

    elif node.type == NodeType.AFFECT_HEAD:
        state_dim = node.params.get('state_dim', 40)
        affect_dim = node.params.get('affect_dim', 5)
        hidden_dim = node.params.get('hidden_dim', 32)

        # Two-layer MLP: state → hidden → affect
        node.weights['fc1_weight'] = WeightInfo('fc1_weight', (hidden_dim, state_dim))
        node.weights['fc1_bias'] = WeightInfo('fc1_bias', (hidden_dim,))
        node.weights['fc2_weight'] = WeightInfo('fc2_weight', (affect_dim, hidden_dim))
        node.weights['fc2_bias'] = WeightInfo('fc2_bias', (affect_dim,))


def get_node_icon(node_type: NodeType) -> str:
    """Get emoji icon for node type."""
    return NODE_DEFINITIONS.get(node_type, {}).get('icon', '⚙️')


def get_node_color(node_type: NodeType) -> str:
    """Get color for node type."""
    return NODE_DEFINITIONS.get(node_type, {}).get('color', '#757575')

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
