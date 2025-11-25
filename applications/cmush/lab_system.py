"""
Lab System - Double-Blind Affect Testing

Provides infrastructure for validating the affect prediction model
through casual, ADHD-friendly A/B testing during natural conversations.

Architecture:
- LabTestSession: Manages a test session (50 trials)
- Dual cognition: Runs each response twice (real affect vs random affect)
- Blind presentation: Randomizes order (A/B)
- User choice: Records preference and applies chosen response

Author: Noodlings Project
Date: November 24, 2025
"""

import random
import time
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LabTestSession:
    """
    Manages a double-blind affect test session.

    Flow:
    1. User sends message
    2. System saves agent state
    3. Generate response with REAL affect
    4. Restore state
    5. Generate response with RANDOM affect
    6. Restore state
    7. Present blind comparison (A/B randomized)
    8. User chooses preferred response
    9. Apply chosen response to world
    10. Record result
    """

    def __init__(
        self,
        player_id: str,
        target_trials: int = 50,
        experiment_name: str = "affect_test"
    ):
        """
        Initialize lab test session.

        Args:
            player_id: User running the test
            target_trials: Number of trials to complete
            experiment_name: Name for this experiment (for file naming)
        """
        self.player_id = player_id
        self.target_trials = target_trials
        self.experiment_name = experiment_name

        # Session state
        self.trials_completed = 0
        self.results = []
        self.current_trial = None
        self.awaiting_choice = False

        # Statistics
        self.real_preferred_count = 0
        self.random_preferred_count = 0
        self.equal_count = 0

        # Timestamp
        self.start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Lab test session started: {self.session_id}, target={target_trials}")

    async def intercept_message(
        self,
        message: str,
        agent,
        world,
        broadcast_fn
    ) -> bool:
        """
        Intercept user message and run dual cognition if not awaiting choice.

        Args:
            message: User message text
            agent: Target agent for testing
            world: World state manager
            broadcast_fn: Function to broadcast messages to player

        Returns:
            True if intercepted (lab active), False otherwise
        """
        if self.awaiting_choice:
            # Don't intercept - waiting for @lab choose command
            return False

        # Run dual cognition
        await self._run_dual_cognition(message, agent, world, broadcast_fn)
        return True

    async def _run_dual_cognition(
        self,
        message: str,
        agent,
        world,
        broadcast_fn
    ):
        """
        Core logic: Run cognition twice with different affect vectors.

        Steps:
        1. Save original agent state
        2. Generate response with REAL affect
        3. Restore state
        4. Generate response with RANDOM affect
        5. Restore state
        6. Randomize presentation order
        7. Present blind comparison
        """
        logger.info(f"[Lab] Starting dual cognition for trial {self.trials_completed + 1}, message='{message}'")

        # Save original state
        original_state = agent.save_state_snapshot()
        logger.info(f"[Lab] Saved original state snapshot")

        # Trial 1: Real affect (model prediction)
        agent.restore_state_snapshot(original_state)
        logger.info(f"[Lab] Running Trial 1: REAL affect")
        response_real = await self._generate_response(agent, message, use_real_affect=True)

        # Trial 2: Random affect
        agent.restore_state_snapshot(original_state)
        logger.info(f"[Lab] Running Trial 2: RANDOM affect")
        response_random = await self._generate_response(agent, message, use_real_affect=False)

        # Restore to original state
        agent.restore_state_snapshot(original_state)
        logger.info(f"[Lab] Restored original state")

        # Randomize presentation order
        real_is_A = random.choice([True, False])

        # Store trial data
        self.current_trial = {
            'message': message,
            'response_A': response_real if real_is_A else response_random,
            'response_B': response_random if real_is_A else response_real,
            'real_is_A': real_is_A,
            'timestamp': time.time()
        }

        self.awaiting_choice = True

        # Present comparison to user
        await self._present_comparison(broadcast_fn)

    async def _generate_response(
        self,
        agent,
        message: str,
        use_real_affect: bool
    ) -> str:
        """
        Generate a single response with specified affect mode.

        Args:
            agent: Agent instance
            message: User message
            use_real_affect: If True, use predicted affect; if False, use random

        Returns:
            Generated response text
        """
        if not use_real_affect:
            # Override affect with random vector (PAD model + sorrow + boredom)
            random_affect = {
                'valence': random.uniform(-1.0, 1.0),
                'arousal': random.uniform(0.0, 1.0),
                'dominance': random.uniform(0.0, 1.0),
                'sorrow': random.uniform(0.0, 1.0),
                'boredom': random.uniform(0.0, 1.0)
            }
            agent.set_affect_override(random_affect)
            logger.debug(f"[Lab] Set random affect: valence={random_affect['valence']:.2f}, arousal={random_affect['arousal']:.2f}, dominance={random_affect['dominance']:.2f}")

        # Generate response (this calls the agent's normal response generation)
        # Note: The actual implementation will depend on agent interface
        # For now, we'll assume there's a generate_response_text() method
        try:
            response = await agent.generate_response_text(message)
        except AttributeError:
            # Fallback: try alternate method names
            if hasattr(agent, 'generate_speech'):
                response = await agent.generate_speech(message)
            else:
                response = "[Error: Could not generate response]"
                logger.error(f"[Lab] Agent missing response generation method")

        # Clear affect override
        if not use_real_affect:
            agent.clear_affect_override()

        return response

    async def _present_comparison(self, broadcast_fn):
        """
        Present blind comparison to user.

        Args:
            broadcast_fn: Function to send messages to player
        """
        trial = self.current_trial

        # Build comparison message
        lines = [
            "",
            "[Dual cognition running...]",
            "",
            "Response A:",
            trial['response_A'],
            "",
            "Response B:",
            trial['response_B'],
            "",
            "Which response is more coherent/satisfying?",
            "Type: @lab choose A  OR  @lab choose B  OR  @lab choose equal",
            ""
        ]

        message = "\n".join(lines)
        await broadcast_fn(message)

    async def record_choice(
        self,
        choice: str,
        world,
        broadcast_fn
    ):
        """
        Record user's choice and apply chosen response to world.

        Args:
            choice: 'A', 'B', or 'equal'
            world: World state manager
            broadcast_fn: Function to broadcast messages
        """
        if not self.awaiting_choice:
            await broadcast_fn("No trial in progress. Start lab mode with: @lab start [trials]")
            return

        trial = self.current_trial

        # Normalize choice
        choice = choice.upper()
        if choice not in ['A', 'B', 'EQUAL']:
            await broadcast_fn("Invalid choice. Use: @lab choose A, @lab choose B, or @lab choose equal")
            return

        # Determine if real was preferred
        if choice == 'EQUAL':
            real_preferred = None
        elif choice == 'A':
            real_preferred = trial['real_is_A']
        else:  # B
            real_preferred = not trial['real_is_A']

        # Save result
        result = {
            'trial': self.trials_completed + 1,
            'choice': choice,
            'real_preferred': real_preferred,
            'message': trial['message'],
            'response_real': trial['response_A'] if trial['real_is_A'] else trial['response_B'],
            'response_random': trial['response_B'] if trial['real_is_A'] else trial['response_A'],
            'timestamp': datetime.now().isoformat()
        }

        self.results.append(result)
        self.trials_completed += 1

        # Update statistics
        if real_preferred is True:
            self.real_preferred_count += 1
        elif real_preferred is False:
            self.random_preferred_count += 1
        else:
            self.equal_count += 1

        # Apply chosen response to world
        chosen_response = trial['response_A'] if choice == 'A' else trial['response_B']
        await broadcast_fn(f"\n{chosen_response}\n")

        # Show progress (pass choice since we're about to clear current_trial)
        await self._show_progress(broadcast_fn, choice)

        # Clear trial state
        self.current_trial = None
        self.awaiting_choice = False

        # Check if complete
        if self.trials_completed >= self.target_trials:
            await self._finish_test(broadcast_fn)

    async def _show_progress(self, broadcast_fn, last_choice: str = None):
        """
        Show progress bar and statistics.

        Args:
            broadcast_fn: Function to broadcast messages
            last_choice: The choice that was just made (A/B/EQUAL)
        """
        percent = int((self.trials_completed / self.target_trials) * 100)
        filled = int(percent / 5)  # 20 blocks for 100%
        bar = "[" + ("=" * filled) + (" " * (20 - filled)) + "]"

        # Calculate win rate
        if self.trials_completed > 0:
            win_rate = (self.real_preferred_count / self.trials_completed) * 100
        else:
            win_rate = 0.0

        choice_display = last_choice if last_choice else 'N/A'
        message = f"\nTrial {self.trials_completed}/{self.target_trials} recorded (chose {choice_display})\nProgress: {bar} {percent}%\nWin rate: {win_rate:.1f}%\n"

        await broadcast_fn(message)

    async def _finish_test(self, broadcast_fn):
        """
        Complete test and save results.

        Args:
            broadcast_fn: Function to broadcast messages
        """
        # Calculate final statistics
        duration = time.time() - self.start_time
        win_rate = (self.real_preferred_count / self.trials_completed) * 100 if self.trials_completed > 0 else 0.0

        # Build summary
        lines = [
            "",
            "LAB TEST COMPLETE!",
            f"Total Trials: {self.trials_completed}",
            f"Real Affect Preferred:   {self.real_preferred_count} ({(self.real_preferred_count/self.trials_completed)*100:.1f}%)",
            f"Random Affect Preferred: {self.random_preferred_count} ({(self.random_preferred_count/self.trials_completed)*100:.1f}%)",
            f"No Preference:           {self.equal_count}",
            "",
        ]

        # Verdict
        if win_rate >= 70:
            lines.append("EXCELLENT: Real affect improves responses!")
        elif win_rate >= 65:
            lines.append("STRONG: Real affect provides clear benefit!")
        elif win_rate >= 60:
            lines.append("GOOD: Real affect shows improvement!")
        elif win_rate >= 55:
            lines.append("MODEST: Real affect provides some benefit!")
        elif 45 <= win_rate < 55:
            lines.append("NEUTRAL: No clear benefit from affect!")
        else:
            lines.append("WARNING: Random affect performed better - investigate!")

        # Save results
        output_path = self._save_results(duration)
        lines.append(f"\nResults saved to: {output_path}")
        lines.append("")

        message = "\n".join(lines)
        await broadcast_fn(message)

    def _save_results(self, duration: float) -> str:
        """
        Save results to JSON file.

        Args:
            duration: Session duration in seconds

        Returns:
            Path to saved file
        """
        # Prepare data structure
        data = {
            'player': self.player_id,
            'experiment': self.experiment_name,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'trials': self.trials_completed,
            'real_preferred': self.real_preferred_count,
            'random_preferred': self.random_preferred_count,
            'equal': self.equal_count,
            'win_rate': (self.real_preferred_count / self.trials_completed) * 100 if self.trials_completed > 0 else 0.0,
            'results': self.results
        }

        # Create experiments directory
        output_dir = Path('experiments')
        output_dir.mkdir(exist_ok=True)

        # Save file
        filename = f"lab_test_{self.session_id}.json"
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"[Lab] Results saved: {output_path}")
        return str(output_path)

    def get_status(self) -> Dict[str, Any]:
        """
        Get current session status.

        Returns:
            Status dict with progress and statistics
        """
        if self.trials_completed > 0:
            win_rate = (self.real_preferred_count / self.trials_completed) * 100
        else:
            win_rate = 0.0

        return {
            'active': True,
            'session_id': self.session_id,
            'experiment': self.experiment_name,
            'trials_completed': self.trials_completed,
            'target_trials': self.target_trials,
            'awaiting_choice': self.awaiting_choice,
            'real_preferred': self.real_preferred_count,
            'random_preferred': self.random_preferred_count,
            'equal': self.equal_count,
            'win_rate': win_rate
        }
