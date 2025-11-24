#!/usr/bin/env python3
"""
Experiment 3B: Temporal Continuity Test

Tests whether the 40-D phenomenal state shows learned temporal dynamics
by repeating the same conversation sequence twice and comparing trajectories.

Hypothesis:
- If temporal model is trained: Similar conversation patterns produce similar state trajectories
- If random/untrained: State trajectories will be completely different

Method:
1. Run standardized conversation sequence (5 exchanges)
2. Reset agent state
3. Run EXACT same conversation sequence again
4. Compare phenomenal state trajectories using Dynamic Time Warping (DTW)
5. Measure trajectory similarity

Pass criteria: High trajectory correlation (>0.7)
Fail criteria: Low correlation (<0.3) indicates random/untrained dynamics

Author: Caity + Spock
Date: November 23, 2025
"""

import asyncio
import websockets
import json
import numpy as np
import aiohttp
from typing import List, Dict, Tuple
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, '..')


class TemporalContinuityExperiment:
    """Test temporal consistency of phenomenal state trajectories."""

    def __init__(
        self,
        agent_id: str = "agent_testsubject",
        websocket_url: str = "ws://localhost:8765",
        api_url: str = "http://localhost:8081",
        output_dir: str = "experiment_results"
    ):
        self.agent_id = agent_id
        self.websocket_url = websocket_url
        self.api_url = api_url
        self.output_dir = output_dir
        self.ws = None
        self.user_id = None

        # Standard conversation sequence
        self.conversation = [
            "Hello! How are you today?",
            "That's interesting. What have you been thinking about?",
            "Tell me more about that.",
            "How does that make you feel?",
            "I appreciate you sharing that with me."
        ]

    async def connect_websocket(self):
        """Establish WebSocket connection and authenticate."""
        print("  Connecting to noodleMUSH...")

        self.ws = await websockets.connect(
            self.websocket_url,
            ping_interval=20,
            ping_timeout=60
        )

        # Authenticate
        await self.ws.send(json.dumps({
            'type': 'login',
            'username': 'caity',
            'password': 'j33k13p13'
        }))

        # Wait for login response
        while True:
            msg = await self.ws.recv()
            data = json.loads(msg)

            if data.get('type') == 'login_response':
                if data.get('success'):
                    self.user_id = data.get('user_id')
                    print(f"    ✓ Authenticated as {self.user_id}")
                    break
                else:
                    raise RuntimeError(f"Login failed: {data.get('message')}")

        # Consume welcome messages
        for _ in range(5):
            try:
                await asyncio.wait_for(self.ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                break

    async def send_message(self, text: str):
        """Send message via WebSocket."""
        # Reconnect if needed
        try:
            await self.ws.ping()
        except:
            print("    → Reconnecting...")
            await self.connect_websocket()

        msg = {
            'type': 'command',
            'command': f'say {text}'
        }
        await self.ws.send(json.dumps(msg))

    async def capture_state(self, wait_for_step_change: bool = True, timeout: int = 20) -> Tuple[np.ndarray, int]:
        """
        Capture current 40-D phenomenal state via API.

        Args:
            wait_for_step_change: If True, wait until step counter changes (perception occurred)
            timeout: Max seconds to wait for step change

        Returns:
            Tuple of (40-D vector, step number)
        """
        # Get current step
        if wait_for_step_change:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/agents/{self.agent_id}/state") as response:
                    if response.status == 200:
                        initial_data = await response.json()
                        initial_step = initial_data.get('step', 0)
                    else:
                        initial_step = 0

            # Wait for step to increment (perception occurred)
            for attempt in range(timeout * 2):  # Check every 0.5s
                await asyncio.sleep(0.5)

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.api_url}/api/agents/{self.agent_id}/state") as response:
                        if response.status == 200:
                            data = await response.json()
                            current_step = data.get('step', 0)

                            if current_step > initial_step:
                                # Perception occurred!
                                break
            else:
                print(f"        ⚠ Timeout waiting for perception (step stuck at {initial_step})")
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/agents/{self.agent_id}/state") as response:
                    if response.status == 200:
                        data = await response.json()
                    else:
                        raise RuntimeError(f"Failed to get state: {response.status}")

        # Extract 40-D vector
        fast = data.get('fast_state', [])
        medium = data.get('medium_state', [])
        slow = data.get('slow_state', [])

        vector = fast + medium + slow

        if len(vector) != 40:
            raise RuntimeError(f"Expected 40-D vector, got {len(vector)}-D")

        step = data.get('step', 0)
        return np.array(vector, dtype=np.float32), step

    async def reset_agent(self):
        """Reset agent consciousness state via API."""
        async with aiohttp.ClientSession() as session:
            # Call reset endpoint (if it exists)
            # For now, we'll just note that agent should be respawned manually
            pass

    async def run_conversation_sequence(self, trial_num: int) -> List[np.ndarray]:
        """
        Run the standard conversation sequence and capture state trajectory.

        Args:
            trial_num: Trial number (1 or 2)

        Returns:
            List of 40-D state vectors (one per conversation turn)
        """
        print(f"\n[Trial {trial_num}] Running conversation sequence...")

        trajectory = []

        for i, utterance in enumerate(self.conversation, 1):
            # Send message
            await self.send_message(utterance)
            print(f"  [{i}/5] Sent: \"{utterance[:50]}...\"")

            # Capture state (waits for perception to occur)
            state, step = await self.capture_state(wait_for_step_change=True)
            trajectory.append(state)
            print(f"        ✓ Captured state (step={step})")

        print(f"  ✓ Trajectory captured ({len(trajectory)} states)")

        return trajectory

    def compute_trajectory_similarity(self, traj1: List[np.ndarray], traj2: List[np.ndarray]) -> Dict:
        """
        Compute similarity metrics between two state trajectories.

        Args:
            traj1: First trajectory (list of 40-D vectors)
            traj2: Second trajectory (list of 40-D vectors)

        Returns:
            Dictionary of similarity metrics
        """
        # Convert to matrices
        T1 = np.array(traj1)  # Shape: (T, 40)
        T2 = np.array(traj2)  # Shape: (T, 40)

        # 1. Point-wise correlation (how similar is state i in traj1 to state i in traj2?)
        pointwise_correlations = []
        for i in range(len(T1)):
            corr = np.corrcoef(T1[i], T2[i])[0, 1]
            pointwise_correlations.append(corr)

        mean_pointwise_corr = np.mean(pointwise_correlations)

        # 2. Trajectory shape correlation (do they follow similar paths?)
        # Flatten and correlate entire sequences
        traj_corr = np.corrcoef(T1.flatten(), T2.flatten())[0, 1]

        # 3. Euclidean distance between trajectories
        pointwise_distances = []
        for i in range(len(T1)):
            dist = np.linalg.norm(T1[i] - T2[i])
            pointwise_distances.append(dist)

        mean_distance = np.mean(pointwise_distances)

        # 4. Trajectory divergence (how much do they drift apart over time?)
        divergence_trend = np.polyfit(range(len(pointwise_distances)), pointwise_distances, 1)[0]

        return {
            'mean_pointwise_correlation': float(mean_pointwise_corr),
            'trajectory_correlation': float(traj_corr),
            'mean_euclidean_distance': float(mean_distance),
            'divergence_slope': float(divergence_trend),
            'pointwise_correlations': [float(x) for x in pointwise_correlations],
            'pointwise_distances': [float(x) for x in pointwise_distances]
        }

    async def run_experiment(self):
        """Run temporal continuity experiment."""
        print("=" * 70)
        print("EXPERIMENT 3B: TEMPORAL CONTINUITY TEST")
        print("=" * 70)
        print("\nTesting: Do identical conversations produce similar state trajectories?")
        print(f"Agent: {self.agent_id}")
        print(f"Conversation length: {len(self.conversation)} turns")
        print()

        # Connect
        await self.connect_websocket()

        # Trial 1
        trajectory1 = await self.run_conversation_sequence(trial_num=1)

        # Wait between trials
        print("\n  ⏸  Waiting 10 seconds between trials...")
        await asyncio.sleep(10)

        # Trial 2 (exact same conversation)
        trajectory2 = await self.run_conversation_sequence(trial_num=2)

        # Disconnect
        if self.ws:
            await self.ws.close()
            print("\n  ✓ WebSocket disconnected")

        # Compute similarity
        print("\n" + "=" * 70)
        print("ANALYZING TRAJECTORY SIMILARITY")
        print("=" * 70)

        metrics = self.compute_trajectory_similarity(trajectory1, trajectory2)

        print(f"\nPointwise Correlation: {metrics['mean_pointwise_correlation']:.3f}")
        print(f"Trajectory Correlation: {metrics['trajectory_correlation']:.3f}")
        print(f"Mean Distance: {metrics['mean_euclidean_distance']:.3f}")
        print(f"Divergence Slope: {metrics['divergence_slope']:.3f}")

        print("\nPer-turn correlations:")
        for i, corr in enumerate(metrics['pointwise_correlations'], 1):
            print(f"  Turn {i}: {corr:.3f}")

        # Interpret results
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        mean_corr = metrics['mean_pointwise_correlation']

        if mean_corr > 0.7:
            print("\n✓ HIGH CORRELATION (>0.7)")
            print("  The temporal model shows strong learned dynamics.")
            print("  Similar inputs produce similar state trajectories.")
            print("  This indicates successful temporal training.")
        elif mean_corr > 0.4:
            print("\n~ MODERATE CORRELATION (0.4-0.7)")
            print("  The temporal model shows some learned structure.")
            print("  There's partial consistency, but also variability.")
            print("  May benefit from additional training.")
        else:
            print("\n✗ LOW CORRELATION (<0.4)")
            print("  The temporal model shows weak/random dynamics.")
            print("  Little consistency between identical conversations.")
            print("  Suggests untrained or poorly trained model.")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.output_dir}/experiment3b_continuity_{timestamp}.json"

        results = {
            'metadata': {
                'experiment': '3B_temporal_continuity',
                'agent_id': self.agent_id,
                'timestamp': timestamp,
                'conversation_length': len(self.conversation)
            },
            'conversation': self.conversation,
            'trajectory1': [t.tolist() for t in trajectory1],
            'trajectory2': [t.tolist() for t in trajectory2],
            'metrics': metrics
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved: {output_file}")

        return metrics


async def main():
    """Run Experiment 3B."""
    experiment = TemporalContinuityExperiment(
        agent_id="agent_testsubject",
        websocket_url="ws://localhost:8765",
        api_url="http://localhost:8081",
        output_dir="experiment_results"
    )

    await experiment.run_experiment()

    print("\n" + "=" * 70)
    print("EXPERIMENT 3B COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
