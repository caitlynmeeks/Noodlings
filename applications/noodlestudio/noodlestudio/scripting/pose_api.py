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
#   Pose API - Scripting interface for body animation
#
#   Provides context.noodle.pose for ScriptedFacets to: - Loa...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.pose_api
# PURPOSE:  Pose API - Scripting interface for body animation
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PoseTrackProxy, PoseAPI, get_pose_api()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional, List, Callable
import os


class PoseTrackProxy:
    """
    JavaScript-friendly wrapper for PoseTrackPlayer.

    Provides Unity-like interface for controlling pose animation.
    """

    def __init__(self, player):
        """
        Args:
            player: PoseTrackPlayer instance
        """
        self._player = player

    def play(self, options: Optional[Dict[str, Any]] = None):
        """
        Start playback.

        Args:
            options: Optional dict with:
                - from_time: Start time (seconds)
                - speed: Playback speed
                - loop: Whether to loop

        Example (JavaScript):
            track.play();
            track.play({from_time: 1.0, speed: 0.5});
        """
        if options:
            if 'from_time' in options:
                self._player.seek(float(options['from_time']))
            if 'speed' in options:
                self._player.speed = float(options['speed'])
            if 'loop' in options:
                self._player.is_looping = bool(options['loop'])

        self._player.play()

    def pause(self):
        """Pause playback."""
        self._player.pause()

    def stop(self):
        """Stop and reset to beginning."""
        self._player.stop()

    def seek(self, time: float):
        """
        Jump to specific time.

        Args:
            time: Time in seconds
        """
        self._player.seek(float(time))

    @property
    def speed(self) -> float:
        """Get playback speed."""
        return self._player.speed

    @speed.setter
    def speed(self, value: float):
        """Set playback speed."""
        self._player.speed = float(value)

    @property
    def isPlaying(self) -> bool:
        """Check if currently playing."""
        return self._player.is_playing

    @property
    def isLooping(self) -> bool:
        """Check if looping."""
        return self._player.is_looping

    @isLooping.setter
    def isLooping(self, value: bool):
        """Set looping."""
        self._player.is_looping = bool(value)

    @property
    def time(self) -> float:
        """Get current playback time."""
        return self._player.current_time

    @property
    def duration(self) -> float:
        """Get track duration."""
        return self._player.track.duration

    @property
    def name(self) -> str:
        """Get track name."""
        return self._player.track.name

    def getMuscles(self) -> Dict[str, float]:
        """
        Get current muscle values.

        Returns:
            Dict mapping muscle names to values [-1, 1]

        Example (JavaScript):
            var muscles = track.getMuscles();
            context.log("Arm: " + muscles["RightArm.DownUp"]);
        """
        pose = self._player.sample()
        return dict(pose.muscles)

    def getBlendShapes(self) -> Dict[str, float]:
        """
        Get current blend shape weights.

        Returns:
            Dict mapping blend shape names to weights [0, 1]
        """
        pose = self._player.sample()
        return dict(pose.blendshapes)

    def sample(self, time: Optional[float] = None) -> Dict[str, Any]:
        """
        Sample pose at specific time (or current time).

        Args:
            time: Optional time to sample at

        Returns:
            Dict with muscles, blendshapes, root
        """
        t = float(time) if time is not None else None
        pose = self._player.sample(t)
        return pose.to_dict()

    def onMarker(self, marker_name: str, callback: Callable):
        """
        Register callback for when a marker is reached.

        Args:
            marker_name: Name of marker
            callback: Function to call

        Example (JavaScript):
            track.onMarker("arm_raised", function() {
                context.log("Arm is up!");
            });
        """
        self._player.on_marker(marker_name, callback)

    def onComplete(self, callback: Callable):
        """
        Register callback for when track finishes.

        Args:
            callback: Function to call with final pose
        """
        self._player.completion_callback = callback

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JavaScript-compatible dict."""
        return {
            'name': self.name,
            'duration': self.duration,
            'time': self.time,
            'isPlaying': self.isPlaying,
            'isLooping': self.isLooping,
            'speed': self.speed
        }


class PoseAPI:
    """
    Pose animation API for ScriptedFacets.

    Provides access to body animation via context.noodle.pose.
    Uses Mecanim-style muscle space for rig-agnostic animation.
    """

    def __init__(self):
        """Initialize Pose API."""
        self._loaded_tracks: Dict[str, 'PoseTrackProxy'] = {}
        self._active_track: Optional[PoseTrackProxy] = None
        self._current_pose: Optional['PoseState'] = None
        self._retargeter: Optional['PoseRetargeter'] = None
        self._avatar_config: Optional[Dict[str, Any]] = None

        # Direct muscle control (procedural animation)
        self._manual_muscles: Dict[str, float] = {}

        # Momentum settings
        self._momentum_muscles: Dict[str, float] = {}
        self._momentum_decay: str = "spring"
        self._momentum_stiffness: float = 0.5
        self._momentum_damping: float = 0.3

    def loadTrack(self, path: str) -> Optional[PoseTrackProxy]:
        """
        Load a pose track from file.

        Args:
            path: Path to .posetrack or .noodletrack file

        Returns:
            PoseTrackProxy for controlling playback, or None if failed

        Example (JavaScript):
            var wave = context.noodle.pose.loadTrack("animations/wave.posetrack");
            wave.play();
        """
        try:
            # Lazy import to avoid circular dependencies
            from noodlestudio.core.pose_track import PoseTrack, PoseTrackPlayer

            # Resolve path
            if not os.path.isabs(path):
                # Look in standard locations
                search_paths = [
                    path,
                    f"animations/{path}",
                    f"library/animations/{path}",
                ]
                for sp in search_paths:
                    if os.path.exists(sp):
                        path = sp
                        break

            if not os.path.exists(path):
                print(f"[PoseAPI] Track not found: {path}")
                return None

            track = PoseTrack.load_yaml(path)
            player = PoseTrackPlayer(track)
            proxy = PoseTrackProxy(player)

            self._loaded_tracks[path] = proxy
            self._active_track = proxy

            print(f"[PoseAPI] Loaded track: {track.name} ({track.duration:.2f}s)")
            return proxy

        except Exception as e:
            print(f"[PoseAPI] Failed to load track: {e}")
            return None

    def getState(self) -> Dict[str, Any]:
        """
        Get current blended pose state.

        Combines active track + manual muscles + momentum.

        Returns:
            Dict with muscles, blendshapes, root

        Example (JavaScript):
            var pose = context.noodle.pose.getState();
            context.log("Arm: " + pose.muscles["RightArm.DownUp"]);
        """
        from noodlestudio.core.pose_track import PoseState

        # Start with neutral
        muscles: Dict[str, float] = {}
        blendshapes: Dict[str, float] = {}

        # Add track pose
        if self._active_track and self._active_track._player:
            track_pose = self._active_track._player.sample()
            muscles.update(track_pose.muscles)
            blendshapes.update(track_pose.blendshapes)

        # Add manual muscles (override)
        muscles.update(self._manual_muscles)

        # Add momentum decay
        # (In real impl, this would decay over time)
        for name, value in self._momentum_muscles.items():
            if name not in muscles:
                muscles[name] = value

        return {
            'muscles': muscles,
            'blendshapes': blendshapes,
            'root': {'position': [0, 0, 0], 'rotation': [0, 0, 0, 1]}
        }

    def getMuscle(self, name: str, default: float = 0.0) -> float:
        """
        Get a single muscle value.

        Args:
            name: Muscle name (e.g., "Head.NodDownUp")
            default: Default if not set

        Returns:
            Muscle value [-1, 1]
        """
        state = self.getState()
        return state['muscles'].get(name, default)

    def setMuscle(self, name: str, value: float):
        """
        Set a muscle value directly (procedural animation).

        Args:
            name: Muscle name
            value: Value [-1, 1]

        Example (JavaScript):
            context.noodle.pose.setMuscle("Head.NodDownUp", 0.5);
        """
        self._manual_muscles[name] = max(-1.0, min(1.0, float(value)))

    def clearManualMuscles(self):
        """Clear all manually set muscle values."""
        self._manual_muscles.clear()

    def setMomentum(self, muscles: Dict[str, float], options: Optional[Dict[str, Any]] = None):
        """
        Set momentum muscles (for track completion handoff).

        When a pose track ends, it can hand off its final pose
        which will then decay naturally via spring physics.

        Args:
            muscles: Dict of muscle values
            options: Optional settings:
                - decay: "spring", "linear", "exponential"
                - stiffness: Spring stiffness (0-1)
                - damping: Spring damping (0-1)

        Example (JavaScript):
            track.onComplete(function(finalPose) {
                context.noodle.pose.setMomentum(finalPose.muscles, {
                    decay: "spring",
                    stiffness: 0.5
                });
            });
        """
        self._momentum_muscles = dict(muscles)
        if options:
            self._momentum_decay = options.get('decay', 'spring')
            self._momentum_stiffness = float(options.get('stiffness', 0.5))
            self._momentum_damping = float(options.get('damping', 0.3))

    def setAvatar(self, avatar_id: str):
        """
        Set the target avatar for retargeting.

        Args:
            avatar_id: Avatar identifier (loads its muscle config)

        Example (JavaScript):
            context.noodle.pose.setAvatar("yuki");
        """
        # TODO: Load avatar-specific muscle definitions
        self._avatar_config = {'id': avatar_id}

        # Initialize retargeter
        from noodlestudio.core.pose_track import PoseRetargeter
        self._retargeter = PoseRetargeter()

    def getBoneRotations(self) -> Dict[str, List[float]]:
        """
        Get bone rotations for current pose (after retargeting).

        Returns:
            Dict mapping bone names to [euler_x, euler_y, euler_z] in degrees

        Example (JavaScript):
            var bones = context.noodle.pose.getBoneRotations();
            for (var bone in bones) {
                context.log(bone + ": " + bones[bone]);
            }
        """
        if not self._retargeter:
            from noodlestudio.core.pose_track import PoseRetargeter
            self._retargeter = PoseRetargeter()

        from noodlestudio.core.pose_track import PoseState
        state = self.getState()
        pose = PoseState(
            muscles=state['muscles'],
            blendshapes=state['blendshapes']
        )

        rotations = self._retargeter.apply_pose(pose)
        return {bone: list(rot) for bone, rot in rotations.items()}

    def getAvailableMuscles(self) -> List[str]:
        """
        Get list of available muscle names.

        Returns:
            List of standard humanoid muscle names

        Example (JavaScript):
            var muscles = context.noodle.pose.getAvailableMuscles();
            context.log("Available: " + muscles.length + " muscles");
        """
        from noodlestudio.core.pose_track import HUMANOID_MUSCLES
        return list(HUMANOID_MUSCLES)

    def getMuscleDefinition(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get definition for a muscle (axis, min, max).

        Args:
            name: Muscle name

        Returns:
            Dict with axis, min, max, default or None

        Example (JavaScript):
            var defn = context.noodle.pose.getMuscleDefinition("Head.NodDownUp");
            context.log("Range: " + defn.min + " to " + defn.max + " degrees");
        """
        from noodlestudio.core.pose_track import MUSCLE_DEFINITIONS
        return MUSCLE_DEFINITIONS.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JavaScript-compatible dict for context injection."""
        return {
            'loadTrack': '__pose_loadTrack__',
            'getState': '__pose_getState__',
            'getMuscle': '__pose_getMuscle__',
            'setMuscle': '__pose_setMuscle__',
            'setMomentum': '__pose_setMomentum__',
            'setAvatar': '__pose_setAvatar__',
            'getBoneRotations': '__pose_getBoneRotations__',
            'getAvailableMuscles': '__pose_getAvailableMuscles__',
            'getMuscleDefinition': '__pose_getMuscleDefinition__'
        }


# Global singleton
_pose_api_instance: Optional[PoseAPI] = None


def get_pose_api() -> PoseAPI:
    """Get global PoseAPI singleton."""
    global _pose_api_instance
    if _pose_api_instance is None:
        _pose_api_instance = PoseAPI()
    return _pose_api_instance

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
