"""
Performance Operation Tracker

Tracks all operations (LLM calls, MLX forward passes, memory ops) with timestamps
for real-time visibility into what agents are doing and where bottlenecks are.

Author: NoodleMUSH Project
Date: November 2025
"""

import time
import logging
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks operations with timing for performance analysis.

    Maintains a rolling buffer of recent operations per agent.
    """

    def __init__(self, max_operations_per_agent: int = 100):
        """
        Initialize performance tracker.

        Args:
            max_operations_per_agent: Max operations to keep in buffer per agent
        """
        self.max_operations = max_operations_per_agent

        # agent_id -> deque of operation dicts
        self.operation_logs: Dict[str, deque] = {}

        # Global operation counter for sequencing
        self.operation_counter = 0

        logger.info(f"PerformanceTracker initialized (buffer size: {max_operations_per_agent})")

    @contextmanager
    def track_operation(
        self,
        agent_id: str,
        operation_type: str,
        details: Optional[Dict] = None
    ):
        """
        Context manager to track an operation's duration.

        Args:
            agent_id: Agent performing the operation
            operation_type: Type of operation (e.g., "llm_text_to_affect", "mlx_forward")
            details: Optional additional details about the operation

        Usage:
            with tracker.track_operation(agent_id, "llm_generate_response"):
                response = await llm.generate(...)
        """
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()

        # Operation metadata
        op_id = self.operation_counter
        self.operation_counter += 1

        try:
            yield  # Execute the operation

            # Operation completed successfully
            duration_ms = (time.time() - start_time) * 1000

            operation = {
                'id': op_id,
                'agent_id': agent_id,
                'type': operation_type,
                'timestamp': start_timestamp,
                'duration_ms': round(duration_ms, 2),
                'status': 'success',
                'details': details or {}
            }

            self._log_operation(agent_id, operation)

        except Exception as e:
            # Operation failed
            duration_ms = (time.time() - start_time) * 1000

            operation = {
                'id': op_id,
                'agent_id': agent_id,
                'type': operation_type,
                'timestamp': start_timestamp,
                'duration_ms': round(duration_ms, 2),
                'status': 'error',
                'error': str(e),
                'details': details or {}
            }

            self._log_operation(agent_id, operation)

            # Re-raise the exception
            raise

    def log_instant_event(
        self,
        agent_id: str,
        event_type: str,
        details: Optional[Dict] = None
    ):
        """
        Log an instantaneous event (not duration-based).

        Args:
            agent_id: Agent performing the event
            event_type: Type of event (e.g., "received_stimulus", "surprise_spike")
            details: Optional additional details
        """
        timestamp = datetime.now().isoformat()

        event = {
            'id': self.operation_counter,
            'agent_id': agent_id,
            'type': event_type,
            'timestamp': timestamp,
            'duration_ms': 0,
            'status': 'event',
            'details': details or {}
        }

        self.operation_counter += 1
        self._log_operation(agent_id, event)

    def _log_operation(self, agent_id: str, operation: Dict):
        """
        Add operation to agent's log buffer.

        Args:
            agent_id: Agent ID
            operation: Operation dict
        """
        # Initialize buffer for agent if needed
        if agent_id not in self.operation_logs:
            self.operation_logs[agent_id] = deque(maxlen=self.max_operations)

        # Add to buffer (automatically evicts oldest if full)
        self.operation_logs[agent_id].append(operation)

        # Log to console for debugging
        logger.debug(
            f"[{operation['timestamp']}] {agent_id}: {operation['type']} "
            f"({operation['duration_ms']}ms) - {operation['status']}"
        )

    def get_recent_operations(
        self,
        agent_id: str,
        last_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Get recent operations for an agent.

        Args:
            agent_id: Agent ID
            last_n: Number of recent operations to return (None = all)

        Returns:
            List of operation dicts
        """
        if agent_id not in self.operation_logs:
            return []

        operations = list(self.operation_logs[agent_id])

        if last_n is not None:
            return operations[-last_n:]

        return operations

    def get_all_agent_operations(self, last_n: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Get recent operations for all agents.

        Args:
            last_n: Number of recent operations per agent (None = all)

        Returns:
            Dict mapping agent_id -> list of operations
        """
        return {
            agent_id: self.get_recent_operations(agent_id, last_n)
            for agent_id in self.operation_logs.keys()
        }

    def get_operation_stats(self, agent_id: str) -> Dict:
        """
        Get statistics about an agent's operations.

        Args:
            agent_id: Agent ID

        Returns:
            Dict with stats (avg duration by type, etc.)
        """
        if agent_id not in self.operation_logs:
            return {'total_operations': 0}

        operations = list(self.operation_logs[agent_id])

        # Group by operation type
        by_type = {}
        for op in operations:
            op_type = op['type']
            if op_type not in by_type:
                by_type[op_type] = {
                    'count': 0,
                    'total_ms': 0,
                    'avg_ms': 0,
                    'min_ms': float('inf'),
                    'max_ms': 0
                }

            stats = by_type[op_type]
            stats['count'] += 1
            stats['total_ms'] += op['duration_ms']
            stats['min_ms'] = min(stats['min_ms'], op['duration_ms'])
            stats['max_ms'] = max(stats['max_ms'], op['duration_ms'])

        # Calculate averages
        for op_type in by_type:
            stats = by_type[op_type]
            stats['avg_ms'] = round(stats['total_ms'] / stats['count'], 2)

        return {
            'agent_id': agent_id,
            'total_operations': len(operations),
            'by_type': by_type
        }

    def clear_agent_log(self, agent_id: str):
        """Clear operation log for an agent."""
        if agent_id in self.operation_logs:
            self.operation_logs[agent_id].clear()


# Global tracker instance
_global_tracker = None


def get_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


# Testing
if __name__ == "__main__":
    import asyncio

    async def test_tracker():
        tracker = get_tracker()

        # Test timed operation
        with tracker.track_operation("agent_test", "llm_call", {"model": "qwen"}):
            await asyncio.sleep(0.1)  # Simulate LLM call

        # Test instant event
        tracker.log_instant_event("agent_test", "surprise_spike", {"surprise": 0.85})

        # Get recent operations
        ops = tracker.get_recent_operations("agent_test")
        print(f"\nRecent operations: {len(ops)}")
        for op in ops:
            print(f"  [{op['timestamp']}] {op['type']}: {op['duration_ms']}ms")

        # Get stats
        stats = tracker.get_operation_stats("agent_test")
        print(f"\nStats: {stats}")

    asyncio.run(test_tracker())
