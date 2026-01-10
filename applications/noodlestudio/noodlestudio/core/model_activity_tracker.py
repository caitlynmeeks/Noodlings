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
#   Model Activity Tracker - Real-time LLM usage monitoring.
#
#   Tracks active requests per model label for ambient activi...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.model_activity_tracker
# PURPOSE:  Model Activity Tracker
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   LabelActivity, ModelActivityTracker, CmushActivityBridge, get_model_activity_tracker(), get_cmush_activity_bridge()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import time
import logging
import requests
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


@dataclass
class LabelActivity:
    """Activity state for a single label."""
    active_requests: int = 0  # Currently in-flight requests
    total_requests: int = 0  # Lifetime request count
    last_activity_time: float = 0.0  # Timestamp of last activity
    total_tokens: int = 0  # Lifetime token count


class ModelActivityTracker(QObject):
    """
    Tracks real-time LLM activity per model label.

    Emits Qt signals for UI updates. Thread-safe for async usage.

    Usage:
        tracker = get_model_activity_tracker()

        # When LLM call starts
        tracker.request_started("LARGE")

        # When LLM call completes
        tracker.request_completed("LARGE", tokens=150)

        # Query current state
        active = tracker.get_active_count("LARGE")  # → 3
    """

    # Signals for UI updates
    activityChanged = pyqtSignal(str, int)  # label, active_count
    requestCompleted = pyqtSignal(str, int)  # label, tokens

    def __init__(self):
        super().__init__()
        self._activity: Dict[str, LabelActivity] = defaultdict(LabelActivity)
        self._active_request_ids: Dict[str, Set[str]] = defaultdict(set)

    def request_started(self, label: str, request_id: Optional[str] = None) -> str:
        """
        Mark a request as started for a label.

        Args:
            label: Model label (e.g., "SMALL", "MEDIUM", "LARGE")
            request_id: Optional unique ID (generated if not provided)

        Returns:
            Request ID for tracking
        """
        if not label:
            return ""

        label = label.upper()

        # Generate request ID if not provided
        if not request_id:
            request_id = f"{label}_{time.time_ns()}"

        # Update activity
        activity = self._activity[label]
        activity.active_requests += 1
        activity.total_requests += 1
        activity.last_activity_time = time.time()

        # Track request ID
        self._active_request_ids[label].add(request_id)

        # Emit signal
        self.activityChanged.emit(label, activity.active_requests)

        return request_id

    def request_completed(self, label: str, request_id: Optional[str] = None, tokens: int = 0):
        """
        Mark a request as completed.

        Args:
            label: Model label
            request_id: Request ID from request_started (optional)
            tokens: Tokens used in this request
        """
        if not label:
            return

        label = label.upper()
        activity = self._activity[label]

        # Decrement active count (minimum 0)
        if activity.active_requests > 0:
            activity.active_requests -= 1

        activity.last_activity_time = time.time()
        activity.total_tokens += tokens

        # Remove request ID if tracked
        if request_id and request_id in self._active_request_ids[label]:
            self._active_request_ids[label].discard(request_id)

        # Emit signals
        self.activityChanged.emit(label, activity.active_requests)
        if tokens > 0:
            self.requestCompleted.emit(label, tokens)

    def get_active_count(self, label: str) -> int:
        """Get number of currently active requests for a label."""
        if not label:
            return 0
        return self._activity[label.upper()].active_requests

    def get_total_requests(self, label: str) -> int:
        """Get lifetime request count for a label."""
        if not label:
            return 0
        return self._activity[label.upper()].total_requests

    def get_total_tokens(self, label: str) -> int:
        """Get lifetime token count for a label."""
        if not label:
            return 0
        return self._activity[label.upper()].total_tokens

    def get_last_activity_time(self, label: str) -> float:
        """Get timestamp of last activity for a label."""
        if not label:
            return 0.0
        return self._activity[label.upper()].last_activity_time

    def get_time_since_activity(self, label: str) -> float:
        """Get seconds since last activity for a label."""
        last = self.get_last_activity_time(label)
        if last == 0:
            return float('inf')
        return time.time() - last

    def get_all_activity(self) -> Dict[str, LabelActivity]:
        """Get activity data for all labels."""
        return dict(self._activity)

    def sync_from_remote(self, activity_data: Dict[str, dict]):
        """
        Sync activity state from remote server (cmush API).

        Used by CmushActivityBridge to update local tracker from server data.
        Emits signals for any labels that changed.

        Args:
            activity_data: Dict mapping label -> activity dict from /api/activity
        """
        for label, data in activity_data.items():
            label = label.upper()
            activity = self._activity[label]

            # Check if active count changed
            new_active = data.get('active_requests', 0)
            if activity.active_requests != new_active:
                activity.active_requests = new_active
                self.activityChanged.emit(label, new_active)

            # Update other fields
            activity.total_requests = data.get('total_requests', activity.total_requests)
            activity.total_tokens = data.get('total_tokens', activity.total_tokens)

            # Use remote timestamp if provided
            remote_time = data.get('last_activity_time', 0)
            if remote_time > 0:
                activity.last_activity_time = remote_time

    def reset(self):
        """Reset all activity tracking."""
        self._activity.clear()
        self._active_request_ids.clear()


# Global singleton
_tracker_instance: Optional[ModelActivityTracker] = None


def get_model_activity_tracker() -> ModelActivityTracker:
    """Get global ModelActivityTracker singleton."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ModelActivityTracker()
    return _tracker_instance


class CmushActivityBridge(QObject):
    """
    Polls cmush server for LLM activity and syncs to local ModelActivityTracker.

    This bridges the gap between the cmush process (where LLM calls happen)
    and NoodleStudio (where the visualization lives).

    Translates model names from cmush to NoodleStudio labels using ModelLabelManager.

    Usage:
        bridge = CmushActivityBridge()
        bridge.start()
        # ... later ...
        bridge.stop()
    """

    def __init__(self, api_base: str = "http://localhost:8081", poll_interval_ms: int = 250):
        """
        Args:
            api_base: Base URL for cmush API
            poll_interval_ms: Polling interval in milliseconds (default 250ms for smooth animation)
        """
        super().__init__()
        self.api_base = api_base.rstrip('/')
        self.poll_interval_ms = poll_interval_ms
        self.tracker = get_model_activity_tracker()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_activity)
        self._connected = False

        # Cache for model->label mapping (refreshed periodically)
        self._model_to_label_cache: Dict[str, str] = {}
        self._cache_refresh_counter = 0

    def start(self):
        """Start polling for activity."""
        logger.info(f"[CmushActivityBridge] Starting activity polling (interval: {self.poll_interval_ms}ms)")
        self._refresh_model_label_cache()
        self._timer.start(self.poll_interval_ms)

    def stop(self):
        """Stop polling."""
        self._timer.stop()
        logger.info("[CmushActivityBridge] Stopped activity polling")

    def _refresh_model_label_cache(self):
        """Refresh the model->label mapping cache from ModelLabelManager."""
        try:
            from .model_label_manager import get_model_label_manager
            manager = get_model_label_manager()

            self._model_to_label_cache.clear()

            for label in manager.get_all_labels():
                provider, model = manager.get_model_for_label(label)
                if model:
                    # Store model name (lowercase) -> label mapping
                    # Strip version/size suffix for matching (e.g., "deepseek-r1:7b" -> "deepseek-r1")
                    model_key = model.lower().split(':')[0]
                    self._model_to_label_cache[model_key] = label
                    # Also store full model name for exact matches
                    self._model_to_label_cache[model.lower()] = label

        except Exception as e:
            logger.debug(f"[CmushActivityBridge] Cache refresh error: {e}")

    def _translate_model_to_label(self, model_name: str) -> str:
        """
        Translate a model name from cmush to a NoodleStudio label.

        Args:
            model_name: Model name from cmush (e.g., "QWEN3", "deepseek-r1")

        Returns:
            Label name (e.g., "Small", "Medium") or original model name if no mapping
        """
        model_key = model_name.lower()

        # Try exact match first
        if model_key in self._model_to_label_cache:
            return self._model_to_label_cache[model_key]

        # Try without version suffix
        base_model = model_key.split(':')[0]
        if base_model in self._model_to_label_cache:
            return self._model_to_label_cache[base_model]

        # No mapping found - return original (will still show as model name)
        return model_name

    def _poll_activity(self):
        """Poll cmush API for activity data."""
        # Refresh cache every ~20 polls (5 seconds at 250ms interval)
        self._cache_refresh_counter += 1
        if self._cache_refresh_counter >= 20:
            self._cache_refresh_counter = 0
            self._refresh_model_label_cache()

        try:
            resp = requests.get(f"{self.api_base}/api/activity", timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                activity = data.get('activity', {})
                if activity:
                    # Translate model names to labels
                    translated = {}
                    for model_name, model_activity in activity.items():
                        label = self._translate_model_to_label(model_name)
                        # If same label appears multiple times, aggregate
                        if label in translated:
                            existing = translated[label]
                            existing['active_requests'] += model_activity.get('active_requests', 0)
                            existing['total_requests'] += model_activity.get('total_requests', 0)
                            existing['total_tokens'] += model_activity.get('total_tokens', 0)
                            # Use most recent activity time
                            existing['last_activity_time'] = max(
                                existing['last_activity_time'],
                                model_activity.get('last_activity_time', 0)
                            )
                        else:
                            translated[label] = dict(model_activity)

                    self.tracker.sync_from_remote(translated)

                if not self._connected:
                    self._connected = True
                    logger.info("[CmushActivityBridge] Connected to cmush activity API")

        except requests.exceptions.ConnectionError:
            if self._connected:
                self._connected = False
                logger.debug("[CmushActivityBridge] Lost connection to cmush")
        except Exception as e:
            logger.debug(f"[CmushActivityBridge] Poll error: {e}")


# Global bridge singleton
_bridge_instance: Optional[CmushActivityBridge] = None


def get_cmush_activity_bridge() -> CmushActivityBridge:
    """Get global CmushActivityBridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = CmushActivityBridge()
    return _bridge_instance


def start_activity_bridge():
    """Start the cmush activity bridge (call from main window init)."""
    bridge = get_cmush_activity_bridge()
    bridge.start()
    return bridge

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
