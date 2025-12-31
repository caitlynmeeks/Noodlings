# Neural Canvas Tutorial Projects

**For Next Session - Pedagogical NN Toys**

Design spec for learner-focused neural network tutorials in NoodleStudio's Neural Canvas.
Target audience: Steve DiPaola, university students, newcomers to neural networks.

---

## Philosophy

Each tutorial should:
1. **Teach ONE concept clearly** - No concept overload
2. **Be runnable immediately** - Test mode shows results
3. **Build on previous tutorials** - Progressive complexity
4. **Connect to affect/cognition** - Tie back to Noodlings mission
5. **Be elegant** - Minimal nodes, maximum insight

---

## Tutorial Progression

### Level 1: The Single Neuron

#### Tutorial 1.1: "The AND Gate"
**Concept:** How a single neuron computes

```
[NUMBER_INPUT a] ──┐
                   ├──► [LINEAR 1] ──► [SIGMOID] ──► [THRESHOLD_OUTPUT]
[NUMBER_INPUT b] ──┘
```

**What it teaches:**
- Inputs have weights
- Bias shifts the decision boundary
- Sigmoid squashes to 0-1
- Threshold makes binary decision

**Missing nodes needed:**
- `NUMBER_INPUT` - Single scalar with slider (0.0 to 1.0)
- `THRESHOLD_OUTPUT` - Shows ON/OFF based on threshold (default 0.5)

**User interaction:**
- Adjust sliders for inputs a and b
- Watch output light turn on/off
- Experiment: What weights make AND work? (both high = on)

---

#### Tutorial 1.2: "The OR Gate"
**Concept:** Different weights, same architecture

Same topology as AND, but different weights. User adjusts weights to discover OR logic.

**Key insight:** The neuron's "personality" (AND vs OR) comes from its weights, not its structure.

---

#### Tutorial 1.3: "The XOR Problem" (The Aha Moment)
**Concept:** Why depth matters

```
[NUMBER_INPUT a] ──┬──► [LINEAR 2] ──► [TANH] ──► [LINEAR 1] ──► [SIGMOID] ──► [OUTPUT]
                   │         │
[NUMBER_INPUT b] ──┴─────────┘
```

**What it teaches:**
- XOR cannot be solved by single layer (user tries, fails)
- Hidden layer creates "features" (intermediate representations)
- Adding depth = adding computational power
- The "aha moment" when it suddenly works

**User interaction:**
1. First: Try single layer (it fails no matter what weights)
2. Then: Add hidden layer
3. Observe: Hidden neurons learn "a AND NOT b" and "b AND NOT a"

**Missing nodes needed:**
- `DECISION_BOUNDARY_VIS` - 2D plot showing the classification space (optional but powerful)

---

### Level 2: Memory and Time

#### Tutorial 2.1: "The Echo Chamber"
**Concept:** RNN remembers previous input

```
[NUMBER_INPUT] ──► [RNN 4] ──► [LINEAR 1] ──► [OUTPUT_CHART]
                     │
                     └── (self-loop hidden state)
```

**What it teaches:**
- Hidden state persists between steps
- Network "remembers" what it saw before
- Output depends on history, not just current input

**User interaction:**
- Enter a sequence: 1, 0, 0, 0, 0...
- Watch output slowly decay (echo fades)
- Reset states - echo disappears

**Missing nodes needed:**
- `OUTPUT_CHART` - Line chart showing output over last N steps (time series visualization)

---

#### Tutorial 2.2: "Learning to Count"
**Concept:** LSTM maintains long-term memory

```
[PULSE_INPUT] ──► [LSTM 8] ──► [LINEAR 1] ──► [COUNTER_OUTPUT]
```

**What it teaches:**
- LSTM gates control information flow
- Cell state is "long-term memory"
- Can count pulses over long sequences
- Contrast with RNN (forgets after ~10 steps)

**User interaction:**
- Click pulse button repeatedly
- Watch counter increment
- Compare RNN (forgets) vs LSTM (remembers)

**Missing nodes needed:**
- `PULSE_INPUT` - Button that sends 1.0 on click, 0.0 otherwise
- `COUNTER_OUTPUT` - Displays rounded integer count

---

#### Tutorial 2.3: "The Delay Line"
**Concept:** RNN can shift/delay signals

```
[NUMBER_INPUT] ──► [RNN 4] ──► [LINEAR 1] ──► [OUTPUT_CHART]
```

Trained to output input from N steps ago.

**What it teaches:**
- Temporal processing = accessing past
- Different hidden units learn different delays
- Foundation for understanding attention later

---

### Level 3: Affect and Emotion

#### Tutorial 3.1: "The Mood Ring"
**Concept:** Mapping input to continuous affect

```
[TEXT_INPUT] ──► [SIMPLE_EMBED] ──► [LINEAR 8] ──► [TANH] ──► [LINEAR 5] ──► [AFFECT_VIS]
```

**What it teaches:**
- Continuous affect (not discrete emotions)
- 5 dimensions: valence, arousal, dominance, sorrow, boredom
- Neural networks as function approximators
- Our affect model!

**User interaction:**
- Type words: "happy", "sad", "angry", "bored"
- Watch 5D affect visualization respond
- Experiment with combinations

**Missing nodes needed:**
- `TEXT_INPUT` - Single line text entry
- `SIMPLE_EMBED` - Character/word lookup table (simple bag-of-words initially)
- `AFFECT_VIS` - Pentagon/radar chart showing 5D affect

---

#### Tutorial 3.2: "Mood Persistence" (The CharmNetwork Lite)
**Concept:** Affect has temporal dynamics

```
[AFFECT_INPUT] ──► [LSTM 16] ──► [AFFECT_HEAD] ──► [AFFECT_VIS]
```

**What it teaches:**
- Mood doesn't jump instantly
- Previous affect influences current response
- This is why we use LSTMs for affect!
- Foundation for understanding CharmNetwork

**User interaction:**
- Input sequence of affects (happy, happy, sad)
- Watch output smooth the transitions
- Observe: sudden sad doesn't immediately override prior happiness

---

#### Tutorial 3.3: "The Full CharmNetwork"
**Concept:** Multi-timescale integration

```
                    ┌── [Fast LSTM 16] ──────────────┐
[AFFECT_INPUT] ─────┼── [Medium LSTM 16] ────────────┼──► [STATE_CONCAT] ──► [AFFECT_HEAD] ──► [AFFECT_VIS]
                    └── [Slow GRU 8] ────────────────┘
```

**What it teaches:**
- Different timescales capture different patterns
- Fast: immediate reaction
- Medium: conversation context
- Slow: long-term mood
- This is the actual production CharmNetwork!

**User interaction:**
- Long affect sequence with patterns
- Watch how different layers respond
- Toggle layers on/off to see contribution

---

### Level 4: Generation and Prediction

#### Tutorial 4.1: "The Sequence Predictor"
**Concept:** Predicting next in sequence

```
[SEQUENCE_INPUT] ──► [LSTM 16] ──► [LINEAR vocab_size] ──► [SOFTMAX] ──► [TOKEN_OUTPUT]
```

Given: A, B, C
Predict: D (or whatever pattern)

**What it teaches:**
- Output is probability distribution
- Softmax converts to probabilities
- Foundation for language models

**Missing nodes needed:**
- `SEQUENCE_INPUT` - Input tokens one at a time
- `TOKEN_OUTPUT` - Shows predicted token + probability

---

#### Tutorial 4.2: "Tiny Shakespeare" (Stretch Goal)
**Concept:** Character-level language model

```
[CHAR_INPUT] ──► [EMBEDDING 16] ──► [LSTM 64] ──► [LINEAR 26] ──► [SOFTMAX] ──► [SAMPLING] ──► [TEXT_OUTPUT]
                                                                                     │
                                                                                     └── (loop back)
```

**What it teaches:**
- Embeddings map discrete to continuous
- Generation loop
- Temperature in sampling
- "Creativity" as controlled randomness

**Missing nodes needed:**
- `CHAR_INPUT` - Character input (or auto-feed from sequence)
- `EMBEDDING` - Learnable lookup table
- `SAMPLING` - Sample from distribution with temperature
- `TEXT_OUTPUT` - Accumulating text display

---

### Level 5: Understanding and Compression

#### Tutorial 5.1: "The Bottleneck" (Autoencoder)
**Concept:** Compression reveals structure

```
[AFFECT_INPUT] ──► [LINEAR 5→3] ──► [TANH] ──► [LINEAR 3→5] ──► [RECONSTRUCTION_VIS]
                         │
                         └── [LATENT_VIS] (shows 3D space)
```

**What it teaches:**
- Bottleneck forces compression
- Network learns what's "essential"
- Latent space is meaningful
- Reconstruction error as signal

**Missing nodes needed:**
- `RECONSTRUCTION_VIS` - Side-by-side input vs output
- `LATENT_VIS` - Visualization of bottleneck activations

---

## Missing Node Types Summary

### Essential (Priority 1 - needed for basic tutorials)

| Node | Description | Inputs | Outputs |
|------|-------------|--------|---------|
| `NUMBER_INPUT` | Single scalar with slider | - | `value: SCALAR` |
| `PULSE_INPUT` | Button sends 1.0 on click | - | `pulse: SCALAR` |
| `THRESHOLD_OUTPUT` | ON/OFF display at threshold | `value: SCALAR` | - |
| `OUTPUT_CHART` | Time series line chart | `value: SCALAR` | - |
| `AFFECT_VIS` | Pentagon/radar for 5D affect | `affect: AFFECT` | - |

### Important (Priority 2 - for richer tutorials)

| Node | Description | Inputs | Outputs |
|------|-------------|--------|---------|
| `TEXT_INPUT` | Single line text entry | - | `embedding: TENSOR` |
| `SIMPLE_EMBED` | Bag-of-words encoder | `text: TEXT` | `embedding: TENSOR` |
| `EMBEDDING` | Learnable lookup table | `token_id: SCALAR` | `vector: TENSOR` |
| `COUNTER_OUTPUT` | Integer display | `value: SCALAR` | - |
| `SEQUENCE_INPUT` | Token sequence feeder | - | `token: SCALAR` |

### Nice to Have (Priority 3 - for advanced tutorials)

| Node | Description | Inputs | Outputs |
|------|-------------|--------|---------|
| `SAMPLING` | Sample from distribution | `probs: TENSOR`, `temperature: SCALAR` | `token: SCALAR` |
| `TEXT_OUTPUT` | Accumulating text display | `token: SCALAR` | - |
| `DECISION_BOUNDARY_VIS` | 2D classification plot | `model reference` | - |
| `LATENT_VIS` | N-D space visualization | `latent: TENSOR` | - |
| `RECONSTRUCTION_VIS` | Input vs output comparison | `input, output: TENSOR` | - |

### Flow Control (Priority 2 - for sequences)

| Node | Description | Inputs | Outputs |
|------|-------------|--------|---------|
| `LOOP` | Iterate input over time | `sequence: TENSOR` | `item: TENSOR`, `done: SCALAR` |
| `DELAY` | Time-shift by N steps | `value: TENSOR` | `delayed: TENSOR` |
| `ACCUMULATOR` | Sum/concat over time | `value: TENSOR` | `accumulated: TENSOR` |

### Math Operations (Priority 2 - for flexibility)

| Node | Description | Inputs | Outputs |
|------|-------------|--------|---------|
| `ADD` | Element-wise addition | `a, b: TENSOR` | `sum: TENSOR` |
| `MULTIPLY` | Element-wise multiply | `a, b: TENSOR` | `product: TENSOR` |
| `CONCAT` | Concatenate tensors | `a, b: TENSOR` | `concat: TENSOR` |
| `SPLIT` | Split tensor at index | `tensor: TENSOR` | `a, b: TENSOR` |
| `RESHAPE` | Change tensor shape | `tensor: TENSOR` | `reshaped: TENSOR` |

---

## Implementation Order

### Phase 1: Basic Logic (Tutorials 1.1-1.3)
1. Add `NUMBER_INPUT` node
2. Add `THRESHOLD_OUTPUT` node
3. Create AND gate tutorial .nncanvas
4. Create OR gate tutorial
5. Create XOR tutorial

### Phase 2: Memory (Tutorials 2.1-2.3)
1. Add `PULSE_INPUT` node
2. Add `OUTPUT_CHART` node
3. Add `COUNTER_OUTPUT` node
4. Create Echo Chamber tutorial
5. Create Counting tutorial

### Phase 3: Affect (Tutorials 3.1-3.3)
1. Add `AFFECT_VIS` node (pentagon chart)
2. Add `TEXT_INPUT` node
3. Add `SIMPLE_EMBED` node
4. Create Mood Ring tutorial
5. Create CharmNetwork Lite tutorial

### Phase 4: Generation (Tutorials 4.1-4.2)
1. Add `EMBEDDING` node
2. Add `SAMPLING` node
3. Add flow control nodes
4. Create Sequence Predictor tutorial

---

## File Organization

```
facet_assemblies/
  charm_networks/
    default.nncanvas              # Production CharmNetwork
    tutorials/
      01_and_gate.nncanvas
      02_or_gate.nncanvas
      03_xor_problem.nncanvas
      04_echo_chamber.nncanvas
      05_counting.nncanvas
      06_delay_line.nncanvas
      07_mood_ring.nncanvas
      08_mood_persistence.nncanvas
      09_charm_network_full.nncanvas
      10_sequence_predictor.nncanvas
```

---

## UI Enhancements Needed

### Tutorial Mode
- **Tutorial Selector** - Dropdown to load tutorial canvases
- **Tutorial Sidebar** - Step-by-step instructions panel
- **Hints System** - Progressive hints if user gets stuck
- **Success Detection** - Recognize when user achieves goal

### Visualization Enhancements
- **Live Value Display** - Already implemented (green badges)
- **Interactive Sliders** - For NUMBER_INPUT nodes
- **Animation** - Show data flowing through network
- **Gradient Visualization** - Show which weights matter (for training tutorials later)

---

## Training Mode (Future)

Not in scope for this phase, but planned:

1. **LOSS** nodes - MSE, CrossEntropy
2. **OPTIMIZER** nodes - SGD, Adam
3. **TRAINING_LOOP** meta-node - Batch training
4. **GRADIENT_VIS** - Show gradients on wires
5. **LEARNING_CURVE** - Plot loss over time

Users would:
1. Design network in canvas
2. Connect LOSS node
3. Click "Train"
4. Watch loss decrease
5. Test trained model

---

## Key Design Decisions

### Why not Keras/TensorFlow style?
We want **visual understanding**, not code generation. Users see data flow, not API calls.

### Why continuous affect instead of emotion labels?
Because that's what makes Noodlings different. We teach the better model.

### Why tutorials in canvas files, not hardcoded?
Tutorials are shareable. Users can:
- Modify tutorials
- Save their versions
- Share discoveries
- Build on each other's work

### Why PyTorch for test mode?
Cross-platform. Windows users exist.

---

## Success Metrics

A tutorial is successful if a newcomer can:
1. **Complete it in under 5 minutes** without help
2. **Explain the concept** to someone else afterward
3. **Modify it** to do something slightly different
4. **Connect it** to the next tutorial's concept

---

## Next Steps for Fresh Session

1. **Start with Phase 1** - Basic logic gates
2. **Implement NUMBER_INPUT and THRESHOLD_OUTPUT** nodes
3. **Create tutorial canvases** for AND, OR, XOR
4. **Add tutorial loading UI** to canvas panel
5. **Write brief instructions** for each tutorial
6. **Test with fresh eyes** - Does it make sense?

---

*Ordnung muss sein, but also: play is learning.*
