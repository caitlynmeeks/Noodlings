"""
Affect API - Scripting interface for affect animation tracks.

Enables ScriptedFacets to:
- Load and play affect animation tracks
- Control playback (play, pause, seek, speed)
- Sample track values at any time
- Blend track with live CharmNetwork affect
- Inject affect for momentum handoff
- Listen for markers/events

Example (JavaScript in ScriptedFacet):
    function process(inputs, context) {
        // Load and play an affect track
        var track = context.noodle.affect.loadTrack("grief_reaction.affecttrack");
        track.play();

        // Query current affect
        var state = context.noodle.affect.getState();
        log("Valence: " + state.valence);

        // Blend with live affect
        context.noodle.affect.setBlendMode("weighted", {track: 0.7, live: 0.3});

        // Listen for markers
        track.onMarker("tears_start", function() {
            context.noodle.events.emit("start_tears", {intensity: 0.6});
        });

        return {processed: true};
    }

Author: Commander Spock + Cadet Caity
Date: December 21, 2025
"""

import os
import time
from typing import Dict, Any, Optional, List, Callable

# Import affect track system
try:
    from ..core.affect_track import (
        AffectTrack, AffectTrackPlayer, AffectTrackFacet,
        AffectState, TrackCompletionBehavior, InterpolationType,
        create_example_track
    )
    AFFECT_TRACK_AVAILABLE = True
except ImportError:
    AFFECT_TRACK_AVAILABLE = False


class AffectTrackProxy:
    """
    JavaScript-friendly proxy for AffectTrackPlayer.

    Wraps the track player with methods accessible from ScriptedFacets.
    """

    def __init__(self, player: 'AffectTrackPlayer'):
        self._player = player

    def play(self, options: Optional[Dict[str, Any]] = None):
        """
        Start track playback.

        Args:
            options: Optional dict with:
                - fromTime: Start position in seconds
                - speed: Playback speed multiplier
                - onComplete: 'momentum', 'snap_neutral', 'hold', 'loop'
                - transferScale: Scale for momentum transfer (0-1)

        Example (JS):
            track.play({
                fromTime: 0,
                speed: 1.0,
                onComplete: "momentum",
                transferScale: 0.9
            });
        """
        if options:
            from_time = options.get('fromTime', 0.0)
            if 'speed' in options:
                self._player.speed = options['speed']
            if 'onComplete' in options:
                self._player.on_complete = TrackCompletionBehavior(options['onComplete'])
            if 'transferScale' in options:
                self._player.transfer_scale = options['transferScale']
            self._player.seek(from_time)

        self._player.play()

    def pause(self):
        """Pause playback."""
        self._player.pause()

    def stop(self):
        """Stop and reset to beginning."""
        self._player.stop()

    def resume(self):
        """Resume from paused state."""
        self._player.play()

    def seek(self, time_seconds: float):
        """
        Jump to specific time.

        Args:
            time_seconds: Position in seconds
        """
        self._player.seek(time_seconds)

    @property
    def speed(self) -> float:
        """Get playback speed."""
        return self._player.speed

    @speed.setter
    def speed(self, value: float):
        """Set playback speed."""
        self._player.speed = value

    @property
    def currentTime(self) -> float:
        """Get current playback position in seconds."""
        return self._player.current_time

    @property
    def duration(self) -> float:
        """Get track duration in seconds."""
        return self._player.track.duration

    @property
    def isPlaying(self) -> bool:
        """Check if track is playing."""
        return self._player.is_playing

    @property
    def isLooping(self) -> bool:
        """Check if track is looping."""
        return self._player.is_looping

    @isLooping.setter
    def isLooping(self, value: bool):
        """Set looping mode."""
        self._player.is_looping = value

    def sample(self, t: Optional[float] = None) -> Dict[str, float]:
        """
        Sample track at specific time (or current time).

        Args:
            t: Time in seconds (optional, defaults to current time)

        Returns:
            Dict with valence, arousal, dominance, boredom, sorrow
        """
        state = self._player.sample(t)
        return state.to_dict()

    def update(self) -> Dict[str, float]:
        """
        Update playback and return current state.

        Should be called each frame/tick.

        Returns:
            Dict with current affect values
        """
        state = self._player.update()
        return state.to_dict()

    def onMarker(self, marker_name: str, callback: Callable):
        """
        Register callback for when a marker is reached.

        Args:
            marker_name: Name of the marker
            callback: Function to call (no args)

        Example (JS):
            track.onMarker("tears_start", function() {
                log("Tears starting!");
            });
        """
        self._player.on_marker(marker_name, callback)

    def getMarkers(self) -> List[Dict[str, Any]]:
        """
        Get all markers in the track.

        Returns:
            List of {time, name} dicts
        """
        return [m.to_dict() for m in self._player.track.markers]

    def getLiveBlendWeight(self) -> float:
        """
        Get current live affect blend weight (from blend regions).

        Returns:
            0.0 = 100% track, 1.0 = 100% live
        """
        return self._player.get_live_blend_weight()


class AffectAPI:
    """
    Affect Animation Track API.

    Main API for managing affect tracks in ScriptedFacets.
    Available via context.noodle.affect
    """

    def __init__(self):
        """Initialize Affect API."""
        self._loaded_tracks: Dict[str, AffectTrackProxy] = {}
        self._active_track: Optional[AffectTrackProxy] = None

        # Reference to CharmNetwork for injection
        self._charm_network = None

        # Current blend mode settings
        self._blend_mode = "override"
        self._blend_weights = {"track": 1.0, "live": 0.0}

        # Track search paths
        self._search_paths = []

    def setCharmNetwork(self, charm_network):
        """Set CharmNetwork reference for momentum handoff."""
        self._charm_network = charm_network

    def addSearchPath(self, path: str):
        """Add a directory to search for track files."""
        if path not in self._search_paths:
            self._search_paths.append(path)

    def loadTrack(self, path: str) -> Optional[AffectTrackProxy]:
        """
        Load an affect track from file.

        Args:
            path: Path to .affecttrack file (absolute or relative to search paths)

        Returns:
            AffectTrackProxy for controlling playback, or None if not found

        Example (JS):
            var track = context.noodle.affect.loadTrack("grief_reaction.affecttrack");
            if (track) {
                track.play();
            }
        """
        if not AFFECT_TRACK_AVAILABLE:
            print("[AffectAPI] Affect track system not available")
            return None

        # Check if already loaded
        if path in self._loaded_tracks:
            return self._loaded_tracks[path]

        # Search for file
        full_path = self._find_track_file(path)
        if not full_path:
            print(f"[AffectAPI] Track not found: {path}")
            return None

        # Load track
        try:
            track = AffectTrack.load_yaml(full_path)
            player = AffectTrackPlayer(track)
            proxy = AffectTrackProxy(player)

            # Set up completion callback for momentum handoff
            def on_complete(p):
                self._handle_track_complete(p)
            player.completion_callback = on_complete

            self._loaded_tracks[path] = proxy
            self._active_track = proxy

            print(f"[AffectAPI] Loaded track: {track.name} ({track.duration}s)")
            return proxy

        except Exception as e:
            print(f"[AffectAPI] Failed to load track {path}: {e}")
            return None

    def createTrack(self, name: str = "New Track") -> AffectTrackProxy:
        """
        Create a new empty affect track.

        Args:
            name: Track name

        Returns:
            AffectTrackProxy for the new track

        Example (JS):
            var track = context.noodle.affect.createTrack("Custom Reaction");
            track.addKeyframe("valence", 0.0, 0.5);
            track.addKeyframe("valence", 2.0, -0.8);
        """
        if not AFFECT_TRACK_AVAILABLE:
            return None

        track = AffectTrack(name=name)
        player = AffectTrackPlayer(track)
        proxy = AffectTrackProxy(player)

        # Give it a unique ID
        track_id = f"custom_{int(time.time() * 1000)}"
        self._loaded_tracks[track_id] = proxy
        self._active_track = proxy

        return proxy

    def createExampleTrack(self) -> AffectTrackProxy:
        """
        Create the example "Receiving Bad News" track.

        Useful for testing and learning.

        Returns:
            AffectTrackProxy with demo track
        """
        if not AFFECT_TRACK_AVAILABLE:
            return None

        track = create_example_track()
        player = AffectTrackPlayer(track)
        proxy = AffectTrackProxy(player)

        self._loaded_tracks["example"] = proxy
        return proxy

    def getState(self) -> Dict[str, Any]:
        """
        Get current blended affect state.

        Returns live CharmNetwork state blended with any playing tracks.

        Returns:
            Dict with:
                - valence, arousal, dominance, boredom, sorrow
                - source: 'live', 'track', or 'blended'
                - isTrackPlaying: boolean
        """
        result = {
            'valence': 0.0,
            'arousal': 0.5,
            'dominance': 0.5,
            'boredom': 0.0,
            'sorrow': 0.0,
            'source': 'live',
            'isTrackPlaying': False
        }

        # Get live affect from CharmNetwork if available
        live_affect = None
        if self._charm_network:
            # Try to get last output
            if hasattr(self._charm_network, '_last_output'):
                last = self._charm_network._last_output
                if last:
                    live_affect = {
                        'valence': last.get('valence', 0.0),
                        'arousal': last.get('arousal', 0.5),
                        'dominance': last.get('dominance', 0.5),
                        'boredom': last.get('boredom', 0.0),
                        'sorrow': last.get('sorrow', 0.0)
                    }

        # Get track affect if playing
        track_affect = None
        if self._active_track and self._active_track.isPlaying:
            track_affect = self._active_track.update()
            result['isTrackPlaying'] = True

        # Blend based on mode
        if track_affect and live_affect:
            result['source'] = 'blended'
            result.update(self._blend_affects(track_affect, live_affect))
        elif track_affect:
            result['source'] = 'track'
            result.update(track_affect)
        elif live_affect:
            result['source'] = 'live'
            result.update(live_affect)

        return result

    def setBlendMode(self, mode: str, weights: Optional[Dict[str, float]] = None):
        """
        Set how track and live affect blend.

        Args:
            mode: 'override', 'weighted', 'additive', 'multiplicative', 'maximum'
            weights: Dict with 'track' and 'live' weights (for 'weighted' mode)

        Example (JS):
            context.noodle.affect.setBlendMode("weighted", {track: 0.7, live: 0.3});
        """
        self._blend_mode = mode
        if weights:
            self._blend_weights = weights

    def inject(self, affect: Dict[str, float], decay: str = "natural"):
        """
        Inject affect state directly into the system.

        Same as momentum handoff - the affect will decay naturally.

        Args:
            affect: Dict with valence, arousal, dominance, boredom, sorrow
            decay: 'natural' (CharmNetwork decay) or 'instant' (snap back)

        Example (JS):
            context.noodle.affect.inject({
                valence: -0.5,
                arousal: 0.8,
                dominance: 0.3,
                boredom: 0.0,
                sorrow: 0.2
            }, "natural");
        """
        if not self._charm_network:
            print("[AffectAPI] No CharmNetwork available for injection")
            return

        if hasattr(self._charm_network, 'inject_state'):
            self._charm_network.inject_state(
                valence=affect.get('valence', 0.0),
                arousal=affect.get('arousal', 0.5),
                dominance=affect.get('dominance', 0.5),
                boredom=affect.get('boredom', 0.0),
                sorrow=affect.get('sorrow', 0.0),
                crossfade=0.5 if decay == "natural" else 0.0
            )

    def _find_track_file(self, path: str) -> Optional[str]:
        """Find track file in search paths."""
        # Check if absolute path exists
        if os.path.isabs(path) and os.path.exists(path):
            return path

        # Search in paths
        for search_path in self._search_paths:
            full_path = os.path.join(search_path, path)
            if os.path.exists(full_path):
                return full_path

        # Check relative to current working directory
        if os.path.exists(path):
            return os.path.abspath(path)

        return None

    def _blend_affects(
        self,
        track: Dict[str, float],
        live: Dict[str, float]
    ) -> Dict[str, float]:
        """Blend track and live affect based on current mode."""
        w = self._blend_weights.get('track', 1.0)

        if self._blend_mode == 'override':
            return track

        elif self._blend_mode == 'weighted':
            return {
                'valence': track['valence'] * w + live['valence'] * (1 - w),
                'arousal': track['arousal'] * w + live['arousal'] * (1 - w),
                'dominance': track['dominance'] * w + live['dominance'] * (1 - w),
                'boredom': track['boredom'] * w + live['boredom'] * (1 - w),
                'sorrow': track['sorrow'] * w + live['sorrow'] * (1 - w)
            }

        elif self._blend_mode == 'additive':
            return {
                'valence': max(-1, min(1, live['valence'] + track['valence'] * w)),
                'arousal': max(0, min(1, live['arousal'] + (track['arousal'] - 0.5) * w)),
                'dominance': max(0, min(1, live['dominance'] + (track['dominance'] - 0.5) * w)),
                'boredom': max(0, min(1, live['boredom'] + track['boredom'] * w)),
                'sorrow': max(0, min(1, live['sorrow'] + track['sorrow'] * w))
            }

        elif self._blend_mode == 'maximum':
            return {
                'valence': track['valence'] if abs(track['valence']) > abs(live['valence']) else live['valence'],
                'arousal': max(track['arousal'], live['arousal']),
                'dominance': max(track['dominance'], live['dominance']),
                'boredom': max(track['boredom'], live['boredom']),
                'sorrow': max(track['sorrow'], live['sorrow'])
            }

        return track  # Default

    def _handle_track_complete(self, player: 'AffectTrackPlayer'):
        """Handle track completion for momentum handoff."""
        if player.on_complete == TrackCompletionBehavior.MOMENTUM:
            if self._charm_network and hasattr(self._charm_network, 'inject_state'):
                final = player.sample(player.track.duration)
                self._charm_network.inject_state(
                    valence=final.valence * player.transfer_scale,
                    arousal=final.arousal * player.transfer_scale,
                    dominance=final.dominance * player.transfer_scale,
                    boredom=final.boredom * player.transfer_scale,
                    sorrow=final.sorrow * player.transfer_scale,
                    crossfade=player.crossfade_duration
                )
                print(f"[AffectAPI] Momentum handoff complete")


# Singleton instance
_affect_api: Optional[AffectAPI] = None


def get_affect_api() -> AffectAPI:
    """Get the global AffectAPI singleton."""
    global _affect_api
    if _affect_api is None:
        _affect_api = AffectAPI()
    return _affect_api


# JavaScript-friendly wrapper class
class AffectAPIJS:
    """
    JavaScript-accessible wrapper for AffectAPI.

    Matches JavaScript naming conventions (camelCase).
    """

    def __init__(self, api: AffectAPI):
        self._api = api

    def loadTrack(self, path: str) -> Optional[AffectTrackProxy]:
        return self._api.loadTrack(path)

    def createTrack(self, name: str = "New Track") -> Optional[AffectTrackProxy]:
        return self._api.createTrack(name)

    def createExampleTrack(self) -> Optional[AffectTrackProxy]:
        return self._api.createExampleTrack()

    def getState(self) -> Dict[str, Any]:
        return self._api.getState()

    def setBlendMode(self, mode: str, weights: Optional[Dict[str, float]] = None):
        self._api.setBlendMode(mode, weights)

    def inject(self, affect: Dict[str, float], decay: str = "natural"):
        self._api.inject(affect, decay)
