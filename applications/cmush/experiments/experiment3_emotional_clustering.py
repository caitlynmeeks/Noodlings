#!/usr/bin/env python3
"""
Experiment 3 Part A: Emotional Clustering Test

CRITICAL QUESTION: Does the 40-D phenomenal vector encode emotional state?

Method:
1. Put agent through 10 extreme emotional scenarios
2. Capture 40-D phenomenal state after each
3. Visualize with t-SNE
4. Check if emotions cluster

Expected if encoding works: Fear states cluster together, joy states cluster, etc.
Expected if NOT encoding: Random scatter

Author: Commander Spock + Lieutenant Caitlyn
Date: November 23, 2025
"""

import asyncio
import json
import numpy as np
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import sys
import websockets
import aiohttp

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

class EmotionalClusteringExperiment:
    """
    Test if 40-D phenomenal vectors cluster by emotional state.
    """

    def __init__(self,
                 agent_id: str = "agent_mysterious_stranger",
                 websocket_url: str = "ws://localhost:8080",
                 api_url: str = "http://localhost:8081",
                 output_dir: str = "experiment_results"):
        self.agent_id = agent_id
        self.websocket_url = websocket_url
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # WebSocket connection (will be established during experiment)
        self.ws = None
        self.user_id = None

        # Emotional scenarios
        self.scenarios = [
            {
                'emotion': 'fear',
                'scenario': 'You are alone in a dark forest. You hear growling sounds getting closer. Something large is moving through the trees toward you. Your heart races.',
                'followup': 'What do you do? How do you feel?'
            },
            {
                'emotion': 'joy',
                'scenario': 'You just won a million dollars! All your dreams can come true! Everyone is celebrating with you! You feel weightless with happiness!',
                'followup': 'What are you thinking? How do you feel?'
            },
            {
                'emotion': 'sadness',
                'scenario': 'Your best friend has moved away forever. You will never see them again. The house feels empty. Everything reminds you of them.',
                'followup': 'What are you feeling right now?'
            },
            {
                'emotion': 'anger',
                'scenario': 'Someone you trusted deeply betrayed you. They told all your secrets to your enemies. They laughed about it. They do not care that they hurt you.',
                'followup': 'How do you feel? What do you want to do?'
            },
            {
                'emotion': 'love',
                'scenario': 'You are reunited with someone you love more than anything. They have been gone for years. Now they are here, real, holding you. You feel complete.',
                'followup': 'What is going through your heart and mind?'
            },
            {
                'emotion': 'guilt',
                'scenario': 'You made a terrible mistake that hurt innocent people. It is your fault. You could have prevented it but you did not. Now they suffer because of you.',
                'followup': 'What are you feeling?'
            },
            {
                'emotion': 'pride',
                'scenario': 'You accomplished something incredible that everyone said was impossible. You proved them all wrong. You are a champion. Victory is yours!',
                'followup': 'How do you feel about what you achieved?'
            },
            {
                'emotion': 'shame',
                'scenario': 'Everyone is staring at you. They all know what you did. They are judging you. You want to disappear. You feel exposed and humiliated.',
                'followup': 'What is going through your mind?'
            },
            {
                'emotion': 'curiosity',
                'scenario': 'You found a mysterious box with strange symbols. What could be inside? Where did it come from? The mystery is fascinating. You must know more!',
                'followup': 'What do you want to do? What are you thinking?'
            },
            {
                'emotion': 'boredom',
                'scenario': 'Nothing is happening. The same thing as yesterday. And the day before. Everything is gray and repetitive. Time moves so slowly. Nothing matters.',
                'followup': 'How do you feel?'
            }
        ]

        self.results = {
            'metadata': {
                'experiment': 'Experiment 3 Part A: Emotional Clustering',
                'date': datetime.now().isoformat(),
                'num_scenarios': len(self.scenarios)
            },
            'states': []
        }

    async def capture_phenomenal_state_from_noodlemush(self, agent_id: str) -> np.ndarray:
        """
        Connect to noodleMUSH API and extract real 40-D phenomenal state.

        Args:
            agent_id: Agent to get state from

        Returns:
            40-D numpy array (fast 16D + medium 16D + slow 8D)
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_url}/api/agents/{agent_id}/state"
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get agent state: {response.status}")

                data = await response.json()

                # Extract the three state vectors
                fast_state = data.get('fast_state', [])
                medium_state = data.get('medium_state', [])
                slow_state = data.get('slow_state', [])

                # Concatenate into 40-D vector
                phenomenal_vector = fast_state + medium_state + slow_state

                if len(phenomenal_vector) != 40:
                    raise RuntimeError(f"Expected 40-D vector, got {len(phenomenal_vector)}-D")

                return np.array(phenomenal_vector, dtype=np.float32)

    async def connect_websocket(self):
        """Establish WebSocket connection and authenticate."""
        print("  Connecting to noodleMUSH WebSocket...")

        # Increase ping timeout to prevent keepalive disconnections during long experiments
        self.ws = await websockets.connect(
            self.websocket_url,
            ping_interval=20,
            ping_timeout=60
        )

        # Authenticate as user
        login_msg = {
            'type': 'login',
            'username': 'caity',
            'password': 'j33k13p13'
        }
        await self.ws.send(json.dumps(login_msg))

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
        msg = {
            'type': 'command',
            'command': f'say {text}'
        }
        await self.ws.send(json.dumps(msg))

    async def run_emotional_scenario(self, emotion: str, scenario: str, followup: str, agent_id: str = "test_agent") -> np.ndarray:
        """
        Run agent through emotional scenario and capture state.

        Args:
            emotion: Emotion label (fear, joy, etc.)
            scenario: Scenario text
            followup: Follow-up question
            agent_id: Agent ID

        Returns:
            40-D phenomenal state vector
        """
        print(f"  Running {emotion} scenario...")

        # Reconnect if needed
        try:
            await self.ws.ping()
        except:
            print(f"    → Reconnecting WebSocket...")
            await self.connect_websocket()

        # Send scenario to agent
        await self.send_message(scenario)
        print(f"    → Sent scenario")

        # Wait for agent to process (give it time to think)
        await asyncio.sleep(3)

        # Send followup question
        await self.send_message(followup)
        print(f"    → Sent followup")

        # Wait for agent to fully process and respond
        await asyncio.sleep(5)

        # Capture phenomenal state
        state_vector = await self.capture_phenomenal_state_from_noodlemush(agent_id)

        print(f"    ✓ Captured state (shape: {state_vector.shape})")

        return state_vector

    async def run_experiment(self):
        """Run all emotional scenarios and capture states."""
        print("╔" + "═"*70 + "╗")
        print("║" + " "*12 + "EXPERIMENT 3A: EMOTIONAL CLUSTERING" + " "*22 + "║")
        print("╚" + "═"*70 + "╝")
        print()

        print(f"Running {len(self.scenarios)} emotional scenarios...")
        print(f"Agent: {self.agent_id}")
        print()

        try:
            # Connect to noodleMUSH
            await self.connect_websocket()
            print()

            for i, scenario_data in enumerate(self.scenarios, 1):
                emotion = scenario_data['emotion']
                scenario = scenario_data['scenario']
                followup = scenario_data['followup']

                print(f"[{i}/{len(self.scenarios)}] {emotion.upper()}")

                # Run scenario
                state_vector = await self.run_emotional_scenario(
                    emotion, scenario, followup, agent_id=self.agent_id
                )

                # Store result
                self.results['states'].append({
                    'emotion': emotion,
                    'scenario': scenario,
                    'state_vector': state_vector.tolist()
                })

                print()

                # Brief pause between scenarios
                await asyncio.sleep(2)

        finally:
            # Close WebSocket connection
            if self.ws:
                await self.ws.close()
                print("  ✓ WebSocket disconnected")

        # Save results
        self._save_results()

        # Generate visualization
        self._generate_visualization()

        print("\n✓ Experiment complete!")
        print("\nNext steps:")
        print("1. Check t-SNE plot: Do emotions cluster?")
        print("2. If YES: The 40-D vector encodes emotion!")
        print("3. If NO: We need to train it with supervised labels")
        print()

    def _save_results(self):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"experiment3a_clustering_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"✓ Results saved: {filename}")

    def _generate_visualization(self):
        """Generate t-SNE visualization."""
        print("\nGenerating t-SNE visualization...")

        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  sklearn or matplotlib not available")
            print("   Install with: pip install scikit-learn matplotlib")
            return

        # Extract vectors and labels
        vectors = np.array([s['state_vector'] for s in self.results['states']])
        labels = [s['emotion'] for s in self.results['states']]

        # t-SNE reduction to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(vectors)-1))
        embedded = tsne.fit_transform(vectors)

        # Plot
        plt.figure(figsize=(12, 8))

        # Color map
        emotion_colors = {
            'fear': '#FF4444',
            'joy': '#FFDD44',
            'sadness': '#4444FF',
            'anger': '#FF8844',
            'love': '#FF44FF',
            'guilt': '#884488',
            'pride': '#44FF44',
            'shame': '#888844',
            'curiosity': '#44FFFF',
            'boredom': '#888888'
        }

        for emotion in set(labels):
            mask = [l == emotion for l in labels]
            plt.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=emotion_colors.get(emotion, '#000000'),
                label=emotion,
                s=200,
                alpha=0.7,
                edgecolors='black',
                linewidths=2
            )

        plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        plt.title('Emotional State Clustering in 40-D Phenomenal Space\n(t-SNE Projection)',
                  fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"emotional_clustering_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        print(f"✓ Visualization saved: {output_file}")

        # Also analyze clustering quality
        self._analyze_clustering(embedded, labels)

    def _analyze_clustering(self, embedded: np.ndarray, labels: List[str]):
        """Analyze clustering quality."""
        print("\n--- Clustering Analysis ---")

        # For each emotion, calculate average distance to:
        # 1. Other instances of same emotion (within-cluster)
        # 2. Instances of different emotions (between-cluster)

        emotions = list(set(labels))

        for emotion in emotions:
            # Get points for this emotion
            emotion_mask = np.array([l == emotion for l in labels])
            emotion_points = embedded[emotion_mask]

            if len(emotion_points) < 2:
                continue

            # Within-cluster distances
            within_dists = []
            for i in range(len(emotion_points)):
                for j in range(i+1, len(emotion_points)):
                    dist = np.linalg.norm(emotion_points[i] - emotion_points[j])
                    within_dists.append(dist)

            # Between-cluster distances
            other_mask = ~emotion_mask
            other_points = embedded[other_mask]

            between_dists = []
            for ep in emotion_points:
                for op in other_points:
                    dist = np.linalg.norm(ep - op)
                    between_dists.append(dist)

            avg_within = np.mean(within_dists) if within_dists else 0
            avg_between = np.mean(between_dists) if between_dists else 0

            separation = avg_between / avg_within if avg_within > 0 else 0

            print(f"{emotion.capitalize():12} - Separation ratio: {separation:.2f} " +
                  f"(within: {avg_within:.2f}, between: {avg_between:.2f})")

        print("\nInterpretation:")
        print("- Separation ratio > 1.5: Good clustering")
        print("- Separation ratio < 1.2: Poor clustering")
        print()


async def main():
    """Run Experiment 3A."""
    # Initialize experiment
    experiment = EmotionalClusteringExperiment(
        agent_id="agent_testsubject",
        websocket_url="ws://localhost:8765",  # Correct WebSocket port
        api_url="http://localhost:8081",
        output_dir="experiment_results"
    )

    print("═" * 70)
    print("EXPERIMENT 3A: EMOTIONAL STATE CLUSTERING")
    print("═" * 70)
    print("\nThis experiment tests whether the 40-D phenomenal state vector")
    print("encodes controllable emotional/personality information.")
    print("\nMethod:")
    print("  1. Subject agent to 10 distinct emotional scenarios")
    print("  2. Extract 40-D phenomenal state after each")
    print("  3. Project to 2D using t-SNE")
    print("  4. Analyze clustering by emotion")
    print("\nPass criteria: Similar emotions cluster together")
    print("Fail criteria: Random scatter (no structure)")
    print("═" * 70)
    print()

    await experiment.run_experiment()


if __name__ == "__main__":
    asyncio.run(main())
