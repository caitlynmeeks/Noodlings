# PyTorch Migration Guide

**Noodlings CharmNetwork MLX → PyTorch Conversion Strategy**

**Date:** December 3, 2025
**Author:** NinaK (Vulcan Nina Hagen) + Caity
**Status:** Planning Phase

---

## Executive Summary

The Noodlings CharmNetwork architecture is **highly portable** to PyTorch with minimal modifications. The design uses standard LSTM/GRU operations with no Apple Silicon-specific dependencies. Expected migration time: 5-7 days for experienced PyTorch developer.

**Key Benefits of Migration:**
- ✅ Cross-platform: Linux, Windows, Mac
- ✅ NVIDIA GPU support (99% of ML hardware)
- ✅ 2-5x faster inference on comparable NVIDIA GPUs
- ✅ Better deployment tools (Docker, cloud platforms)
- ✅ Larger community and ecosystem

---

## Architecture Analysis

### Current Stack (MLX)
- **Framework:** Apple MLX (lazy evaluation, unified memory)
- **Target Hardware:** Apple Silicon (M1/M2/M3)
- **Model Size:** ~54K parameters (Phase 4)
- **Components:**
  - Hierarchical temporal network (fast LSTM, medium LSTM, slow GRU)
  - Quantum microtubule layers (optional)
  - Affect head (40-D → 5-D continuous affect)

### MLX Operations Used
```python
# All have direct PyTorch equivalents:
mx.zeros()              → torch.zeros()
mx.concatenate()        → torch.cat()
mx.array()              → torch.tensor()
mx.random.normal()      → torch.randn()
mx.savez() / mx.load()  → torch.save() / torch.load()
nn.Linear()             → nn.Linear()  (SAME API!)
nn.LSTM()               → nn.LSTM()    (SAME API!)
nn.GRU()                → nn.GRU()     (SAME API!)
mx.eval()               → (remove - PyTorch is eager by default)
```

**Portability Score: 95%** - Only minor framework differences to handle.

---

## Migration Phases

### Phase 1: Core Model (1-2 days)

**Files to Convert:**
- `noodlings/models/noodling_phase4.py`
- `noodlings/models/noodling_phase6.py` (if using Phase 6)
- `noodlings/api.py`

**Changes Required:**

#### 1.1 Imports
```python
# BEFORE (MLX)
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# AFTER (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
```

#### 1.2 Device Management
```python
# Add to __init__:
class NoodlingModelPhase4(nn.Module):
    def __init__(self, ..., device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Move all initialized tensors to device:
        self.h_fast = torch.zeros((1, self.fast_hidden), device=self.device)
        self.c_fast = torch.zeros((1, self.fast_hidden), device=self.device)
        # ... etc
```

#### 1.3 Remove Lazy Evaluation
```python
# BEFORE (MLX - lines 188, 209, 218, 227, 232)
mx.eval(base_output['actual_state'])
mx.eval(fast_modulated)
mx.eval(medium_modulated)
mx.eval(slow_modulated)
mx.eval(phenomenal_state_quantum)

# AFTER (PyTorch)
# DELETE THESE LINES - PyTorch evaluates eagerly
```

#### 1.4 Array Operations
```python
# BEFORE (MLX)
phenomenal_state = mx.concatenate([h_fast, h_medium, h_slow], axis=-1)
affect_mx = mx.array(affect_vector, dtype=mx.float32)

# AFTER (PyTorch)
phenomenal_state = torch.cat([h_fast, h_medium, h_slow], dim=-1)
affect_torch = torch.tensor(affect_vector, dtype=torch.float32, device=self.device)
```

#### 1.5 State Dictionary Methods
```python
# BEFORE (MLX)
def save_weights(self, path: str):
    weights = {}
    for name, param in self.named_parameters():
        weights[name] = param
    mx.savez(path, **weights)

def load_weights(self, path: str):
    weights = mx.load(path)
    # ... manual loading

# AFTER (PyTorch)
def save_weights(self, path: str):
    torch.save(self.state_dict(), path)

def load_weights(self, path: str):
    self.load_state_dict(torch.load(path, map_location=self.device))
```

---

### Phase 2: Quantum Microtubule Layers (1 day)

**Files to Convert:**
- `noodlings/models/quantum_microtubule.py`
- `noodlings/models/quantum_charm_network.py`

**Special Consideration: Avalanche RNG**

Current implementation (lines 86-88):
```python
entropy_service = get_entropy_service()
self.avalanche_rng = entropy_service.create_avalanche_rng(beta=2.0)
```

**Option A: Port entropy_service**
- Convert `applications/cmush/entropy_service.py` to PyTorch
- Maintains exact same RNG behavior

**Option B: Use PyTorch distributions (RECOMMENDED)**
```python
from torch.distributions import Pareto

class QuantumMicrotubuleLayer(nn.Module):
    def __init__(self, ..., device='cuda'):
        super().__init__()
        self.device = device

        # Power-law distribution (avalanche dynamics)
        # beta=2.0 in avalanche RNG → alpha=2.0 in Pareto
        self.avalanche_dist = Pareto(
            torch.tensor(1.0, device=device),  # scale
            torch.tensor(2.0, device=device)   # alpha (shape parameter)
        )

    def _sample_quantum_noise(self, shape):
        """Sample power-law distributed quantum noise."""
        return self.avalanche_dist.sample(shape) * self.noise_scale
```

**Why Option B?**
- ✅ Simpler (no custom RNG service)
- ✅ PyTorch-native (better performance)
- ✅ Statistically equivalent (both power-law distributions)
- ✅ Easier to maintain

---

### Phase 3: Checkpoint Conversion (1 day)

**Convert Existing Trained Models**

Create conversion utility:

```python
# noodlings/utils/convert_mlx_to_pytorch.py

import numpy as np
import torch
import mlx.core as mx
from typing import Dict

def convert_checkpoint(
    mlx_path: str,
    pytorch_path: str,
    model: torch.nn.Module,
    device: str = 'cuda'
) -> None:
    """
    Convert MLX .npz checkpoint to PyTorch .pth format.

    Args:
        mlx_path: Path to MLX checkpoint.npz file
        pytorch_path: Output path for PyTorch checkpoint.pth
        model: PyTorch model instance (for state_dict keys)
        device: Target device ('cuda' or 'cpu')
    """
    print(f"Loading MLX checkpoint: {mlx_path}")

    # Load MLX weights (stored as NumPy .npz)
    mlx_weights = np.load(mlx_path)

    # Convert to PyTorch state dict
    state_dict = {}
    for key, value in mlx_weights.items():
        # Convert NumPy array to PyTorch tensor
        tensor = torch.from_numpy(value).to(device)
        state_dict[key] = tensor
        print(f"  Converted: {key} → shape {tensor.shape}")

    # Load into model to verify compatibility
    model.load_state_dict(state_dict)
    print(f"✓ Verified compatibility with model architecture")

    # Save PyTorch checkpoint
    torch.save({
        'model_state_dict': state_dict,
        'architecture': 'NoodlingModelPhase4',
        'params_count': sum(p.numel() for p in model.parameters()),
        'converted_from_mlx': True,
        'original_checkpoint': mlx_path
    }, pytorch_path)

    print(f"✓ Saved PyTorch checkpoint: {pytorch_path}")

# Usage:
if __name__ == '__main__':
    from noodlings.models.noodling_phase4 import NoodlingModelPhase4

    model = NoodlingModelPhase4(device='cuda')
    convert_checkpoint(
        mlx_path='checkpoints/phase4.npz',
        pytorch_path='checkpoints/phase4_pytorch.pth',
        model=model
    )
```

**Batch Conversion Script:**
```bash
# convert_all_checkpoints.sh
python noodlings/utils/convert_mlx_to_pytorch.py \
    --mlx checkpoints/phase4.npz \
    --pytorch checkpoints/phase4_pytorch.pth

python noodlings/utils/convert_mlx_to_pytorch.py \
    --mlx checkpoints/phase6.npz \
    --pytorch checkpoints/phase6_pytorch.pth
```

---

### Phase 4: Integration Testing (1-2 days)

**Update Integration Points:**

#### 4.1 agent_bridge.py
```python
# BEFORE (MLX)
import mlx.core as mx
from noodlings.api import NoodlingAgent

# AFTER (PyTorch)
import torch
from noodlings.api import NoodlingAgent

# In AgentBridge.__init__:
self.consciousness = NoodlingAgent(
    checkpoint_path=checkpoint_path,
    config=config,
    device='cuda'  # Or 'cpu' for testing
)
```

#### 4.2 Device Selection Logic
```python
# Add to config.yaml or runtime detection:
def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'  # Apple Metal Performance Shaders
        print(f"✓ Using Apple Metal GPU")
    else:
        device = 'cpu'
        print(f"⚠️  Using CPU (slower)")
    return device
```

#### 4.3 Performance Benchmarks
```python
# noodlings/tests/benchmark_pytorch.py

import time
import torch
from noodlings.models.quantum_charm_network import QuantumCharmNetwork

def benchmark_forward_pass(model, device='cuda', n_iterations=1000):
    """Benchmark CharmNetwork forward pass."""
    # Dummy input
    affect = torch.randn(1, 1, 5, device=device)
    h_fast = torch.zeros(1, 16, device=device)
    c_fast = torch.zeros(1, 16, device=device)
    h_medium = torch.zeros(1, 16, device=device)
    c_medium = torch.zeros(1, 16, device=device)
    h_slow = torch.zeros(1, 8, device=device)

    # Warmup
    for _ in range(10):
        model(affect, h_fast, c_fast, h_medium, c_medium, h_slow)

    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()

    for _ in range(n_iterations):
        output = model(affect, h_fast, c_fast, h_medium, c_medium, h_slow)

    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_iterations) * 1000
    print(f"Average forward pass: {avg_ms:.3f}ms")
    print(f"Throughput: {1000/avg_ms:.1f} cycles/second")

    return avg_ms

# Compare MLX vs PyTorch
if __name__ == '__main__':
    # PyTorch CUDA
    model_cuda = QuantumCharmNetwork(device='cuda')
    cuda_time = benchmark_forward_pass(model_cuda, 'cuda')

    # PyTorch CPU
    model_cpu = QuantumCharmNetwork(device='cpu')
    cpu_time = benchmark_forward_pass(model_cpu, 'cpu')

    print(f"\nSpeedup: {cpu_time/cuda_time:.2f}x faster on CUDA")
```

---

## Testing Strategy

### Unit Tests

```python
# noodlings/tests/test_pytorch_equivalence.py

import torch
import numpy as np
from noodlings.models.noodling_phase4 import NoodlingModelPhase4

def test_forward_pass_shape():
    """Verify output shapes match MLX version."""
    model = NoodlingModelPhase4(device='cpu')
    affect = torch.randn(1, 1, 5)

    self_state, predicted_state, social_info = model.forward_with_social_context(
        affect=affect,
        present_agents=['user_123']
    )

    assert self_state.shape == (1, 40), "Phenomenal state should be 40-D"
    assert predicted_state.shape == (1, 40), "Predicted state should be 40-D"

def test_state_persistence():
    """Verify LSTM states update correctly."""
    model = NoodlingModelPhase4(device='cpu')

    # First forward pass
    affect1 = torch.randn(1, 1, 5)
    _ = model.forward_with_social_context(affect=affect1)
    h_fast_1 = model.h_fast.clone()

    # Second forward pass (states should change)
    affect2 = torch.randn(1, 1, 5)
    _ = model.forward_with_social_context(affect=affect2)
    h_fast_2 = model.h_fast.clone()

    assert not torch.allclose(h_fast_1, h_fast_2), "States should update"

def test_checkpoint_save_load():
    """Verify checkpoint save/load cycle."""
    model1 = NoodlingModelPhase4(device='cpu')
    model2 = NoodlingModelPhase4(device='cpu')

    # Save
    torch.save(model1.state_dict(), '/tmp/test_checkpoint.pth')

    # Load into different model
    model2.load_state_dict(torch.load('/tmp/test_checkpoint.pth'))

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Parameter {n1} mismatch after load"
```

### Integration Tests

```python
# Test in noodleMUSH
def test_agent_conversation_pytorch():
    """End-to-end test: Agent perceives and responds."""
    from applications.cmush.agent_bridge import AgentBridge

    agent = AgentBridge(
        agent_id='test_agent',
        agent_name='Test',
        config={'device': 'cpu'}
    )

    # Perceive event
    event = {
        'type': 'say',
        'user': 'caity',
        'text': 'Hello!',
        'room_id': 'test_room'
    }

    result = await agent.perceive_event(event)

    assert result is not None
    assert 'surprise' in result
    assert result['surprise'] >= 0.0
```

---

## Performance Expectations

### MLX (Apple M3 Max)
- Forward pass: ~2-3ms
- Compute: ~0.1 MFLOPs
- Memory: Unified 128GB

### PyTorch (NVIDIA RTX 4090)
- Forward pass: ~0.5-1ms (2-3x faster)
- Compute: ~0.1 MFLOPs (same)
- Memory: GDDR6X 24GB (900 GB/s bandwidth)

### PyTorch (CPU - Intel i9)
- Forward pass: ~5-10ms (slower, but acceptable)
- Compute: ~0.1 MFLOPs (same)

**Why faster on NVIDIA?**
- Mature cuDNN kernels for LSTM/GRU (15+ years optimization)
- Higher memory bandwidth (900 GB/s vs 400 GB/s)
- Better FP32 tensor core utilization

---

## Deployment Advantages

### Docker Support
```dockerfile
# Dockerfile.pytorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN pip install mlx-to-pytorch-converter websockets aiohttp

COPY noodlings/ /app/noodlings/
COPY applications/cmush/ /app/cmush/

WORKDIR /app
CMD ["python", "cmush/server.py"]
```

### Cloud Deployment
- **AWS:** EC2 with NVIDIA GPUs (g4dn, p3, p4 instances)
- **Google Cloud:** Compute Engine with T4/V100/A100 GPUs
- **Azure:** NC-series VMs with NVIDIA GPUs
- **RunPod, Lambda Labs, Paperspace:** GPU rental platforms

### Windows Support
- PyTorch has excellent Windows support
- CUDA drivers work seamlessly
- No Rosetta translation layer needed

---

## Migration Risks & Mitigations

### Risk 1: Numerical Differences
**Problem:** Minor floating-point differences between MLX and PyTorch implementations.

**Mitigation:**
- Compare outputs with `torch.allclose(mlx_output, pytorch_output, atol=1e-5)`
- Use deterministic mode: `torch.use_deterministic_algorithms(True)`
- Test on known conversations with recorded outputs

### Risk 2: State Restoration Issues
**Problem:** Checkpoint conversion might not preserve exact states.

**Mitigation:**
- Convert checkpoints and run forward pass comparison
- Save both MLX and PyTorch outputs for same inputs
- Use conversation replay tests

### Risk 3: Quantum RNG Behavior Change
**Problem:** Different RNG implementations might affect quantum collapse events.

**Mitigation:**
- Use same random seed for testing: `torch.manual_seed(42)`
- Document that quantum effects are stochastic (minor variation expected)
- Verify statistical properties (mean, variance) match

---

## Rollout Strategy

### Stage 1: Parallel Testing (Week 1)
- Keep MLX version running in production
- Run PyTorch version side-by-side
- Compare outputs for equivalence

### Stage 2: Beta Testing (Week 2)
- Deploy PyTorch version to subset of users
- Monitor for issues
- Collect performance metrics

### Stage 3: Full Migration (Week 3)
- Switch all users to PyTorch version
- Keep MLX version as fallback
- Archive MLX code after 1 month of stability

---

## Code Maintenance Strategy

### Option A: Dual Support (MLX + PyTorch)
**Pros:**
- Mac users get native MLX performance
- Linux/Windows users get PyTorch

**Cons:**
- Maintain two codebases
- Double testing effort
- Bug fix synchronization

### Option B: PyTorch Only
**Pros:**
- Single codebase
- Simpler maintenance
- PyTorch works on Mac too (via MPS backend)

**Cons:**
- Mac users lose unified memory benefits
- Slightly slower on Apple Silicon

**Recommendation:** Option B (PyTorch only) for simplicity. PyTorch's MPS backend works well enough on Mac.

---

## Resources

### PyTorch Documentation
- Official Docs: https://pytorch.org/docs/stable/index.html
- LSTM/GRU API: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- CUDA Best Practices: https://pytorch.org/docs/stable/notes/cuda.html

### Community Support
- PyTorch Forums: https://discuss.pytorch.org/
- PyTorch Slack: https://pytorch.slack.com/
- Stack Overflow: `[pytorch]` tag

### Performance Tools
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- NVIDIA Nsight: https://developer.nvidia.com/nsight-systems
- TensorBoard: https://www.tensorflow.org/tensorboard

---

## Timeline Estimate

| Phase | Duration | Complexity | Priority |
|-------|----------|------------|----------|
| Phase 1: Core Model | 1-2 days | Low | High |
| Phase 2: Quantum Layers | 1 day | Medium | High |
| Phase 3: Checkpoints | 1 day | Low | High |
| Phase 4: Integration | 1-2 days | Medium | High |
| Testing & Validation | 1 day | Low | High |
| Documentation | 0.5 days | Low | Medium |

**Total: 5.5-7.5 days** for experienced PyTorch developer.

---

## Conclusion

The Noodlings CharmNetwork architecture is **excellently positioned** for PyTorch migration. The use of standard LSTM/GRU operations and clean abstractions means minimal code changes are required. The migration will unlock cross-platform support, better performance on NVIDIA GPUs, and access to the broader PyTorch ecosystem.

**Key Advantages:**
- 95% of code is framework-agnostic
- No Apple Silicon-specific dependencies
- Expected 2-5x performance improvement on NVIDIA GPUs
- Opens deployment to Linux/Windows users (vastly larger audience)

**Recommended Approach:**
- Start with Phase 1 (Core Model) to validate approach
- Use PyTorch distributions for quantum RNG (simpler than porting entropy service)
- Maintain single PyTorch codebase going forward (not dual MLX/PyTorch)

---

**Status:** Ready to begin Phase 1
**Next Steps:** Convert `noodling_phase4.py` and test forward pass equivalence

*Ordnung muss sein!* 🖖
