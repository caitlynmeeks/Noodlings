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
#   Flow Control Facets - Logic gates and timing controls
#
#   Special facet types that control execution flow rather th...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.flow_control_facets
# PURPOSE:  flow control facets facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FlowControlMode, FlowControlOutput, TickerGateFacet, ConditionalBranchFacet, RateLimiterFacet
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum


class FlowControlMode(Enum):
    """Flow control operating modes."""
    PASS = "pass"           # Input passes through
    BLOCK = "block"         # Input blocked
    BYPASS = "bypass"       # Input routed to bypass output


@dataclass
class FlowControlOutput:
    """Output from flow control facet."""
    mode: FlowControlMode
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class TickerGateFacet:
    """
    Execute downstream every N cycles.

    Inputs: in
    Outputs: out (fires every N), bypassed (fires on other cycles)
    """

    def __init__(
        self,
        facet_id: str,
        interval: int = 5,
        initial_delay: int = 0,
        mode: str = "modulo"
    ):
        """
        Initialize ticker gate.

        Args:
            facet_id: Unique identifier
            interval: Fire every N cycles
            initial_delay: Wait N cycles before starting
            mode: "modulo" (cycle % interval == 0) or "countdown"
        """
        self.facet_id = facet_id
        self.interval = interval
        self.initial_delay = initial_delay
        self.mode = mode

        self.current_count = 0
        self.total_fires = 0
        self.total_bypasses = 0
        self.paused = False

    def process(self, inputs: Dict[str, Any], cycle: int) -> Dict[str, Any]:
        """
        Process input and decide whether to pass or bypass.

        Args:
            inputs: Input values
            cycle: Current cognitive cycle

        Returns:
            Dict with 'out' or 'bypassed' populated
        """
        if self.paused:
            self.total_bypasses += 1
            return {
                'bypassed': inputs.get('in'),
                'metadata': {'reason': 'paused'}
            }

        # Handle initial delay
        if cycle < self.initial_delay:
            self.total_bypasses += 1
            return {
                'bypassed': inputs.get('in'),
                'metadata': {'reason': 'initial_delay'}
            }

        # Check if should fire
        should_fire = False

        if self.mode == "modulo":
            should_fire = (cycle % self.interval) == 0
        elif self.mode == "countdown":
            self.current_count += 1
            if self.current_count >= self.interval:
                should_fire = True
                self.current_count = 0

        if should_fire:
            self.total_fires += 1
            return {
                'out': inputs.get('in'),
                'metadata': {
                    'cycle': cycle,
                    'fire_count': self.total_fires
                }
            }
        else:
            self.total_bypasses += 1
            return {
                'bypassed': inputs.get('in'),
                'metadata': {'cycle': cycle}
            }

    def pause(self):
        """Pause ticker (all inputs bypassed)."""
        self.paused = True

    def resume(self):
        """Resume ticker."""
        self.paused = False

    def reset(self):
        """Reset counters."""
        self.current_count = 0
        self.total_fires = 0
        self.total_bypasses = 0

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage (0 for flow control - no LLM)."""
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.total_fires + self.total_bypasses,
            'avg_tokens': 0
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = self.total_fires + self.total_bypasses
        return {
            'execution_count': total,
            'total_tokens': 0,
            'avg_tokens': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'last_tokens': 0,
            'last_time': 0.0,
            'fires': self.total_fires,
            'bypasses': self.total_bypasses
        }


class ConditionalBranchFacet:
    """
    Route input based on boolean condition.

    Inputs: in, condition_variables (dict)
    Outputs: true_out, false_out
    """

    def __init__(
        self,
        facet_id: str,
        condition: str,
        variables: List[str]
    ):
        """
        Initialize conditional branch.

        Args:
            facet_id: Unique identifier
            condition: Python expression (e.g., "surprise > 0.5")
            variables: List of variable names used in condition
        """
        self.facet_id = facet_id
        self.condition = condition
        self.variables = variables

        self.true_count = 0
        self.false_count = 0

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate condition and route to appropriate output.

        Args:
            inputs: Input values (must include all variables)

        Returns:
            Dict with 'true_out' or 'false_out' populated
        """
        # Build evaluation context
        eval_context = {}
        for var in self.variables:
            if var in inputs:
                eval_context[var] = inputs[var]
            else:
                raise ValueError(f"Variable '{var}' not found in inputs")

        # Evaluate condition
        try:
            result = eval(self.condition, {"__builtins__": {}}, eval_context)
        except Exception as e:
            raise RuntimeError(f"Condition evaluation failed: {e}")

        # Route to appropriate output
        if result:
            self.true_count += 1
            return {
                'true_out': inputs.get('in'),
                'metadata': {
                    'condition': self.condition,
                    'result': True,
                    'true_count': self.true_count
                }
            }
        else:
            self.false_count += 1
            return {
                'false_out': inputs.get('in'),
                'metadata': {
                    'condition': self.condition,
                    'result': False,
                    'false_count': self.false_count
                }
            }

    def set_condition(self, condition: str):
        """Update condition expression."""
        self.condition = condition

    def evaluate(self, variables: Dict[str, Any]) -> bool:
        """Manually evaluate condition with given variables."""
        return eval(self.condition, {"__builtins__": {}}, variables)

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage (0 for flow control)."""
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.true_count + self.false_count,
            'avg_tokens': 0
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.true_count + self.false_count,
            'total_tokens': 0,
            'avg_tokens': 0,
            'true_count': self.true_count,
            'false_count': self.false_count
        }


class RateLimiterFacet:
    """
    Throttle execution by time.

    Inputs: in
    Outputs: out (throttled), rate_limited (blocked inputs)
    """

    def __init__(
        self,
        facet_id: str,
        min_interval: float = 3.0,
        mode: str = "throttle"
    ):
        """
        Initialize rate limiter.

        Args:
            facet_id: Unique identifier
            min_interval: Minimum seconds between executions
            mode: "throttle" (immediate first, then block) or "debounce" (delay until quiet)
        """
        self.facet_id = facet_id
        self.min_interval = min_interval
        self.mode = mode

        self.last_execution_time = 0.0
        self.pass_count = 0
        self.block_count = 0

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check rate limit and route accordingly.

        Args:
            inputs: Input values

        Returns:
            Dict with 'out' or 'rate_limited' populated
        """
        current_time = time.time()
        elapsed = current_time - self.last_execution_time

        if self.mode == "throttle":
            if elapsed >= self.min_interval or self.last_execution_time == 0:
                # Pass through
                self.last_execution_time = current_time
                self.pass_count += 1
                return {
                    'out': inputs.get('in'),
                    'metadata': {
                        'elapsed': elapsed,
                        'pass_count': self.pass_count
                    }
                }
            else:
                # Rate limited
                self.block_count += 1
                return {
                    'rate_limited': inputs.get('in'),
                    'metadata': {
                        'elapsed': elapsed,
                        'required': self.min_interval,
                        'block_count': self.block_count
                    }
                }

        elif self.mode == "debounce":
            # TODO: Implement debounce logic
            raise NotImplementedError("Debounce mode not yet implemented")

    def set_interval(self, seconds: float):
        """Update minimum interval."""
        self.min_interval = seconds

    def clear(self):
        """Allow immediate execution on next call."""
        self.last_execution_time = 0.0

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage (0 for flow control)."""
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.pass_count + self.block_count,
            'avg_tokens': 0
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.pass_count + self.block_count,
            'total_tokens': 0,
            'avg_tokens': 0,
            'pass_count': self.pass_count,
            'block_count': self.block_count
        }


class CacheFacet:
    """
    Cache outputs to avoid recomputation.

    Inputs: in, invalidate_signal (optional)
    Outputs: out, cache_hit (boolean)
    """

    def __init__(
        self,
        facet_id: str,
        ttl: int = 10,
        invalidate_on: Optional[List[str]] = None
    ):
        """
        Initialize cache facet.

        Args:
            facet_id: Unique identifier
            ttl: Time-to-live in cycles
            invalidate_on: List of event names that invalidate cache
        """
        self.facet_id = facet_id
        self.ttl = ttl
        self.invalidate_on = invalidate_on or []

        self.cached_value = None
        self.cached_cycle = None
        self.hit_count = 0
        self.miss_count = 0

    def process(self, inputs: Dict[str, Any], cycle: int) -> Dict[str, Any]:
        """
        Return cached value if valid, otherwise pass through.

        Args:
            inputs: Input values
            cycle: Current cycle

        Returns:
            Dict with 'out' and 'cache_hit'
        """
        # Check for explicit invalidation signal
        if inputs.get('invalidate_signal'):
            self.invalidate()

        # Check cache validity
        if self.cached_value is not None and self.cached_cycle is not None:
            age = cycle - self.cached_cycle
            if age < self.ttl:
                # Cache hit
                self.hit_count += 1
                return {
                    'out': self.cached_value,
                    'cache_hit': True,
                    'metadata': {
                        'age': age,
                        'hit_count': self.hit_count
                    }
                }

        # Cache miss - pass through and cache result
        self.cached_value = inputs.get('in')
        self.cached_cycle = cycle
        self.miss_count += 1

        return {
            'out': self.cached_value,
            'cache_hit': False,
            'metadata': {
                'miss_count': self.miss_count
            }
        }

    def invalidate(self):
        """Invalidate cached value."""
        self.cached_value = None
        self.cached_cycle = None

    def set_ttl(self, cycles: int):
        """Update TTL."""
        self.ttl = cycles

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage (0 for cache)."""
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.hit_count + self.miss_count,
            'avg_tokens': 0
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.hit_count + self.miss_count,
            'total_tokens': 0,
            'avg_tokens': 0,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': (
                self.hit_count / (self.hit_count + self.miss_count)
                if (self.hit_count + self.miss_count) > 0 else 0
            )
        }


class AccumulatorFacet:
    """
    Accumulate inputs over time window.

    Inputs: in
    Outputs: accumulated (list), window_full (boolean)
    """

    def __init__(
        self,
        facet_id: str,
        window_size: int = 5,
        trigger_mode: str = "full"
    ):
        """
        Initialize accumulator.

        Args:
            facet_id: Unique identifier
            window_size: Number of inputs to accumulate
            trigger_mode: "full" (fire when full), "always" (fire every time), "threshold"
        """
        self.facet_id = facet_id
        self.window_size = window_size
        self.trigger_mode = trigger_mode

        self.buffer = []
        self.total_accumulated = 0

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accumulate input and fire when conditions met.

        Args:
            inputs: Input values

        Returns:
            Dict with 'accumulated' and 'window_full'
        """
        # Add to buffer
        self.buffer.append(inputs.get('in'))
        self.total_accumulated += 1

        # Trim buffer if exceeded
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # Check trigger condition
        window_full = len(self.buffer) >= self.window_size

        if self.trigger_mode == "full":
            should_fire = window_full
        elif self.trigger_mode == "always":
            should_fire = True
        else:
            should_fire = False

        return {
            'accumulated': self.buffer.copy() if should_fire else [],
            'window_full': window_full,
            'metadata': {
                'buffer_size': len(self.buffer),
                'total_accumulated': self.total_accumulated
            }
        }

    def clear(self):
        """Clear buffer."""
        self.buffer = []

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage (0 for accumulator)."""
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.total_accumulated,
            'avg_tokens': 0
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.total_accumulated,
            'total_tokens': 0,
            'avg_tokens': 0,
            'buffer_size': len(self.buffer),
            'window_size': self.window_size
        }


if __name__ == "__main__":
    """Test flow control facets."""

    print("=== Testing Flow Control Facets ===\n")

    # Test 1: Ticker Gate
    print("Test 1: TickerGateFacet (interval=3)")
    print("-" * 40)
    ticker = TickerGateFacet("ticker", interval=3)

    for cycle in range(10):
        result = ticker.process({'in': f"cycle_{cycle}"}, cycle)
        if 'out' in result:
            print(f"Cycle {cycle}: FIRED -> {result['out']}")
        else:
            print(f"Cycle {cycle}: bypassed")

    print(f"Stats: fires={ticker.total_fires}, bypasses={ticker.total_bypasses}\n")

    # Test 2: Conditional Branch
    print("\nTest 2: ConditionalBranchFacet (surprise > 0.5)")
    print("-" * 40)
    branch = ConditionalBranchFacet(
        "branch",
        condition="surprise > 0.5",
        variables=["surprise"]
    )

    test_surprises = [0.2, 0.7, 0.4, 0.9, 0.1]
    for i, surprise_val in enumerate(test_surprises):
        inputs = {'in': f"input_{i}", 'surprise': surprise_val}
        result = branch.process(inputs)
        if 'true_out' in result:
            print(f"Surprise {surprise_val}: TRUE -> {result['true_out']}")
        else:
            print(f"Surprise {surprise_val}: FALSE -> {result['false_out']}")

    print(f"Stats: true={branch.true_count}, false={branch.false_count}\n")

    # Test 3: Rate Limiter
    print("\nTest 3: RateLimiterFacet (min_interval=0.5s)")
    print("-" * 40)
    limiter = RateLimiterFacet("limiter", min_interval=0.5)

    for i in range(5):
        result = limiter.process({'in': f"request_{i}"})
        if 'out' in result:
            print(f"Request {i}: PASSED")
        else:
            print(f"Request {i}: RATE LIMITED")
        time.sleep(0.3)  # Faster than limit

    print(f"Stats: pass={limiter.pass_count}, block={limiter.block_count}\n")

    # Test 4: Cache
    print("\nTest 4: CacheFacet (ttl=3 cycles)")
    print("-" * 40)
    cache = CacheFacet("cache", ttl=3)

    for cycle in range(8):
        result = cache.process({'in': f"expensive_computation_{cycle}"}, cycle)
        print(f"Cycle {cycle}: value={result['out']}, cache_hit={result['cache_hit']}")

    print(f"Stats: hits={cache.hit_count}, misses={cache.miss_count}\n")

    # Test 5: Accumulator
    print("\nTest 5: AccumulatorFacet (window=3, mode=full)")
    print("-" * 40)
    accumulator = AccumulatorFacet("accumulator", window_size=3, trigger_mode="full")

    for i in range(7):
        result = accumulator.process({'in': i})
        print(f"Input {i}: accumulated={result['accumulated']}, full={result['window_full']}")

    print("\n=== All tests complete ===")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
