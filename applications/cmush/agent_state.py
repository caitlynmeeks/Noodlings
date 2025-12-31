"""
Agent State Persistence Mixin - State save/load functionality

Extracted from agent_bridge.py for maintainability.
Contains methods for persisting and restoring agent state to disk.

Author: cMUSH Project
Date: December 2025
"""

import os
import json
import glob
import shutil
import logging
from datetime import datetime
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class StatePersistenceMixin:
    """
    Mixin providing state persistence methods for CMUSHNoodlingAgent.
    
    Methods:
        save_state_snapshot: In-memory snapshot for lab experiments
        restore_state_snapshot: Restore from in-memory snapshot
        save_state: Persist to disk with rolling history
        load_state: Load from disk
    """

    def save_state_snapshot(self) -> Dict[str, Any]:
        """
        Save complete agent state snapshot for in-memory restoration.

        Used by lab system for dual cognition experiments.
        Captures all stateful components:
        - Consciousness model hidden states (h_fast, c_fast, etc.)
        - Conversation context
        - Affect history
        - Cognitive manifold state
        - World interaction state

        Returns:
            Dictionary with all restorable state
        """
        import copy
        import mlx.core as mx

        # Get consciousness model states
        h_fast, c_fast, h_medium, c_medium, h_slow = self.consciousness.model.get_states()

        # Save states (convert MLX arrays to numpy for JSON compatibility)
        state = {
            'h_fast': np.array(h_fast) if h_fast is not None else None,
            'c_fast': np.array(c_fast) if c_fast is not None else None,
            'h_medium': np.array(h_medium) if h_medium is not None else None,
            'c_medium': np.array(c_medium) if c_medium is not None else None,
            'h_slow': np.array(h_slow) if h_slow is not None else None,
        }

        # Save conversation context (deep copy to prevent mutation)
        # Note: MemoryListWrapper needs special handling
        if hasattr(self.conversation_context, '_memory'):
            # Save hierarchical memory state
            state['conversation_context'] = {
                'working_memory': copy.deepcopy(list(self.conversation_context._memory.working_memory)),
                'episodic_memory': copy.deepcopy(list(self.conversation_context._memory.episodic_memory)),
            }
        else:
            # Fallback: save as list
            state['conversation_context'] = copy.deepcopy(list(self.conversation_context))

        # Save affect history (if it exists)
        if hasattr(self, 'previous_affect') and self.previous_affect is not None:
            state['previous_affect'] = copy.deepcopy(self.previous_affect)
        else:
            state['previous_affect'] = None

        # Save world interaction state
        state['current_room'] = self.current_room
        state['following'] = self.following
        state['last_response_time'] = self.last_response_time
        state['response_count'] = self.response_count

        # Save autonomous cognition state
        if hasattr(self, 'cognition_engine') and self.cognition_engine:
            state['cognition_engine_state'] = self.cognition_engine.save_state()
        else:
            state['cognition_engine_state'] = None

        logger.debug(f"[{self.agent_id}] State saved: h_fast shape={state['h_fast'].shape if state['h_fast'] is not None else None}")

        return state


    def restore_state_snapshot(self, state: Dict[str, Any]):
        """
        Restore agent to saved state snapshot.

        Used by lab system to reset agent between dual cognition trials.

        Args:
            state: State dictionary from save_state_snapshot()
        """
        import mlx.core as mx

        # Restore consciousness model states
        if state['h_fast'] is not None:
            self.consciousness.model.h_fast = mx.array(state['h_fast'])
        if state['c_fast'] is not None:
            self.consciousness.model.c_fast = mx.array(state['c_fast'])
        if state['h_medium'] is not None:
            self.consciousness.model.h_medium = mx.array(state['h_medium'])
        if state['c_medium'] is not None:
            self.consciousness.model.c_medium = mx.array(state['c_medium'])
        if state['h_slow'] is not None:
            self.consciousness.model.h_slow = mx.array(state['h_slow'])

        # Restore conversation context
        if 'conversation_context' in state:
            if isinstance(state['conversation_context'], dict):
                # Restore hierarchical memory
                self.conversation_context._memory.working_memory = list(state['conversation_context']['working_memory'])
                self.conversation_context._memory.episodic_memory = list(state['conversation_context']['episodic_memory'])
            else:
                # Fallback: restore as list
                # Note: This won't work perfectly with MemoryListWrapper, but provides basic restore
                logger.warning(f"[{self.agent_id}] Restoring conversation_context as list (not ideal)")

        # Restore affect history
        if 'previous_affect' in state:
            self.previous_affect = state['previous_affect']

        # Restore world interaction state
        self.current_room = state['current_room']
        self.following = state['following']
        self.last_response_time = state['last_response_time']
        self.response_count = state['response_count']

        # Restore autonomous cognition state
        if state.get('cognition_engine_state') and hasattr(self, 'cognition_engine') and self.cognition_engine:
            self.cognition_engine.restore_state(state['cognition_engine_state'])

        logger.debug(f"[{self.agent_id}] State restored")


    def save_state(self, state_dir: str, max_history: int = 5):
        """
        Save agent state to disk with rolling history.

        Saves to:
        - agent_state.json (current state)
        - checkpoint.npz (current Noodlings checkpoint)
        - history/state_NNN.json (rolling history, keeps last max_history saves)

        Args:
            state_dir: Directory for agent state
            max_history: Maximum number of historical states to keep (default: 5)
        """
        import glob
        import shutil
        from datetime import datetime

        os.makedirs(state_dir, exist_ok=True)
        history_dir = os.path.join(state_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)

        # Get current phenomenal state from consciousness
        current_state = self.consciousness.get_state()
        phenomenal_state = current_state.get('phenomenal_state', [])

        # Convert to list if needed
        if hasattr(phenomenal_state, 'tolist'):
            phenomenal_state = phenomenal_state.tolist()
        else:
            phenomenal_state = list(phenomenal_state) if phenomenal_state is not None else []

        # Sanitize conversation context for JSON serialization
        # Convert any MLX/numpy arrays to lists
        # Use configurable disk save limit
        disk_save_limit = self.config.get('memory_windows', {}).get('disk_save', 100)
        sanitized_context = []
        for entry in self.conversation_context[-disk_save_limit:]:
            sanitized_entry = dict(entry)  # Copy
            # Convert affect arrays to lists
            if 'affect' in sanitized_entry:
                affect = sanitized_entry['affect']
                if hasattr(affect, 'tolist'):
                    sanitized_entry['affect'] = affect.tolist()
                elif isinstance(affect, (list, tuple)):
                    sanitized_entry['affect'] = list(affect)
            sanitized_context.append(sanitized_entry)

        # Save agent-specific state
        agent_state = {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_description': self.agent_description,
            'current_room': self.current_room,
            'conversation_context': sanitized_context,
            'last_response_time': self.last_response_time,
            'response_count': self.response_count,
            'config': self.config,
            'phenomenal_state': phenomenal_state,  # NEW: Save current emotional state
            'timestamp': datetime.now().isoformat()
        }

        state_path = os.path.join(state_dir, 'agent_state.json')
        try:
            with open(state_path, 'w') as f:
                json.dump(agent_state, f, indent=2)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to save agent state: {e}")
            # Try saving without conversation context as fallback
            agent_state_minimal = {
                'agent_id': self.agent_id,
                'agent_name': self.agent_name,
                'agent_description': self.agent_description,
                'current_room': self.current_room,
                'conversation_context': [],
                'last_response_time': self.last_response_time,
                'response_count': self.response_count,
                'config': {},
                'phenomenal_state': phenomenal_state,
                'timestamp': datetime.now().isoformat()
            }
            with open(state_path, 'w') as f:
                json.dump(agent_state_minimal, f, indent=2)

        # ROLLING HISTORY: Copy current state to history/
        # Find existing history files and determine next number
        existing_history = sorted(glob.glob(os.path.join(history_dir, 'state_*.json')))

        if len(existing_history) >= max_history:
            # Remove oldest state to make room
            oldest_state = existing_history[0]
            os.remove(oldest_state)
            logger.info(f"Removed oldest state snapshot: {os.path.basename(oldest_state)}")
            existing_history = existing_history[1:]  # Update list

        # Determine next state number
        if existing_history:
            last_num = int(os.path.basename(existing_history[-1]).split('_')[1].split('.')[0])
            next_num = last_num + 1
        else:
            next_num = 1

        # Copy current state to history
        history_state_path = os.path.join(history_dir, f'state_{next_num:03d}.json')
        shutil.copy2(state_path, history_state_path)
        logger.info(f"Saved state snapshot: state_{next_num:03d}.json")

        # Save Consilience checkpoint
        checkpoint_path = os.path.join(state_dir, 'checkpoint.npz')
        try:
            self.consciousness.save_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except RuntimeError as e:
            # MLX can throw std::bad_cast for newly initialized models
            # This is safe to skip - agent will start with random weights next time
            if "bad_cast" in str(e):
                logger.warning(f"Skipping checkpoint save for {self.agent_id} (MLX serialization issue - agent will use random weights on next load)")
            else:
                raise  # Re-raise if it's a different RuntimeError

        logger.info(f"Agent state saved: {state_dir} (history: {len(existing_history)+1}/{max_history})")


    def load_state(self, state_dir: str, skip_phenomenal_state: bool = False):
        """
        Load agent state from disk.

        Args:
            state_dir: Directory with agent state
            skip_phenomenal_state: If True, don't restore phenomenal state (fresh spawn with -f flag)
        """
        # Load agent-specific state
        state_path = os.path.join(state_dir, 'agent_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                agent_state = json.load(f)

            self.agent_name = agent_state.get('agent_name', self.agent_name)
            self.agent_description = agent_state.get('agent_description', self.agent_description)
            self.current_room = agent_state.get('current_room')
            # Load conversation context using wrapper method
            saved_context = agent_state.get('conversation_context', [])
            self.conversation_context.load_from_list(saved_context)
            self.last_response_time = agent_state.get('last_response_time', 0.0)
            self.response_count = agent_state.get('response_count', 0)
            # Don't override config passed to __init__

            # NEW: Restore phenomenal state if available and not skipping
            if not skip_phenomenal_state:
                phenomenal_state = agent_state.get('phenomenal_state')
                if phenomenal_state:
                    import mlx.core as mx
                    # Convert list back to MLX array and restore to consciousness
                    phenomenal_state_array = mx.array(phenomenal_state, dtype=mx.float32)
                    self.consciousness.set_phenomenal_state(phenomenal_state_array)
                    logger.info(f"Restored phenomenal state from save (timestamp: {agent_state.get('timestamp', 'unknown')})")
                else:
                    logger.info(f"No phenomenal state found in save file (old format)")
            else:
                logger.info(f"Skipped restoring phenomenal state (fresh spawn with -f)")

        # Load Consilience checkpoint
        checkpoint_path = os.path.join(state_dir, 'checkpoint.npz')
        if os.path.exists(checkpoint_path):
            self.consciousness.load_checkpoint(checkpoint_path)

        logger.info(f"Agent state loaded: {state_dir}")

