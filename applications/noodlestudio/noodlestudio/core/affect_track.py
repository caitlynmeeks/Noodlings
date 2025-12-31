"""
Affect Track - Keyframeable emotional animation curves

"What Maya did for motion, we do for emotion."

Animators can author affect curves (PAD + boredom + sorrow) with
bezier/linear/step interpolation. When track playback ends, the
final affect state can hand off to CharmNetwork for natural decay
(emotional momentum).

File formats:
- .affecttrack (YAML) - Human-readable, editor-friendly
- .affectbin (binary) - Runtime efficient (future)

Author: Commander Spock + Cadet Caity
Date: December 21, 2025
"""

import math
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
from enum import Enum
import os

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


class InterpolationType(Enum):
    """Curve interpolation method."""
    LINEAR = "linear"
    BEZIER = "bezier"
    STEP = "step"
    HERMITE = "hermite"


class TrackCompletionBehavior(Enum):
    """What happens when the track finishes playing."""
    MOMENTUM = "momentum"      # Hand off to CharmNetwork for natural decay
    SNAP_NEUTRAL = "snap_neutral"  # Immediately return to neutral
    HOLD = "hold"              # Hold final values indefinitely
    LOOP = "loop"              # Loop back to start


@dataclass
class Keyframe:
    """A single keyframe on an affect curve."""
    time: float              # Time in seconds
    value: float             # Affect value (-1 to 1 for valence, 0 to 1 for others)

    # Bezier tangent handles (optional)
    in_tangent: Tuple[float, float] = (0.0, 0.0)   # (time_offset, value_offset)
    out_tangent: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            'time': self.time,
            'value': self.value
        }
        # Only include tangents if non-zero
        if self.in_tangent != (0.0, 0.0):
            result['in_tangent'] = list(self.in_tangent)
        if self.out_tangent != (0.0, 0.0):
            result['out_tangent'] = list(self.out_tangent)
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Keyframe':
        """Deserialize from dictionary."""
        in_tan = data.get('in_tangent', [0.0, 0.0])
        out_tan = data.get('out_tangent', [0.0, 0.0])
        return Keyframe(
            time=float(data['time']),
            value=float(data['value']),
            in_tangent=tuple(in_tan) if isinstance(in_tan, list) else in_tan,
            out_tangent=tuple(out_tan) if isinstance(out_tan, list) else out_tan
        )


@dataclass
class AffectChannel:
    """A single affect dimension with keyframed curve."""
    name: str                                    # valence, arousal, dominance, boredom, sorrow
    interpolation: InterpolationType = InterpolationType.LINEAR
    keyframes: List[Keyframe] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'interpolation': self.interpolation.value,
            'keyframes': [kf.to_dict() for kf in self.keyframes]
        }

    @staticmethod
    def from_dict(name: str, data: Dict[str, Any]) -> 'AffectChannel':
        """Deserialize from dictionary."""
        return AffectChannel(
            name=name,
            interpolation=InterpolationType(data.get('interpolation', 'linear')),
            keyframes=[Keyframe.from_dict(kf) for kf in data.get('keyframes', [])]
        )

    def sample(self, t: float) -> float:
        """
        Sample the channel at time t.

        Args:
            t: Time in seconds

        Returns:
            Interpolated value at time t
        """
        if not self.keyframes:
            return 0.0

        # Sort keyframes by time (should already be sorted, but ensure)
        kfs = sorted(self.keyframes, key=lambda k: k.time)

        # Before first keyframe
        if t <= kfs[0].time:
            return kfs[0].value

        # After last keyframe
        if t >= kfs[-1].time:
            return kfs[-1].value

        # Find surrounding keyframes
        for i in range(len(kfs) - 1):
            if kfs[i].time <= t <= kfs[i + 1].time:
                return self._interpolate(kfs[i], kfs[i + 1], t)

        return kfs[-1].value

    def _interpolate(self, k0: Keyframe, k1: Keyframe, t: float) -> float:
        """Interpolate between two keyframes."""
        if self.interpolation == InterpolationType.STEP:
            return k0.value

        # Normalized time (0 to 1 between keyframes)
        duration = k1.time - k0.time
        if duration <= 0:
            return k0.value
        u = (t - k0.time) / duration

        if self.interpolation == InterpolationType.LINEAR:
            return k0.value + (k1.value - k0.value) * u

        elif self.interpolation == InterpolationType.BEZIER:
            return self._bezier_interpolate(k0, k1, u)

        elif self.interpolation == InterpolationType.HERMITE:
            return self._hermite_interpolate(k0, k1, u)

        return k0.value + (k1.value - k0.value) * u

    def _bezier_interpolate(self, k0: Keyframe, k1: Keyframe, u: float) -> float:
        """
        Cubic bezier interpolation.

        Control points:
        P0 = (k0.time, k0.value)
        P1 = P0 + k0.out_tangent
        P2 = P3 - k1.in_tangent
        P3 = (k1.time, k1.value)
        """
        # Control points in value space
        p0 = k0.value
        p3 = k1.value

        # Tangent contributions
        duration = k1.time - k0.time
        p1 = p0 + k0.out_tangent[1]  # value offset from out tangent
        p2 = p3 - k1.in_tangent[1]   # value offset from in tangent

        # Cubic bezier formula: B(u) = (1-u)^3*P0 + 3*(1-u)^2*u*P1 + 3*(1-u)*u^2*P2 + u^3*P3
        u2 = u * u
        u3 = u2 * u
        inv_u = 1.0 - u
        inv_u2 = inv_u * inv_u
        inv_u3 = inv_u2 * inv_u

        return inv_u3 * p0 + 3 * inv_u2 * u * p1 + 3 * inv_u * u2 * p2 + u3 * p3

    def _hermite_interpolate(self, k0: Keyframe, k1: Keyframe, u: float) -> float:
        """
        Hermite spline interpolation using tangents as slopes.
        """
        p0 = k0.value
        p1 = k1.value

        # Tangents as slopes
        duration = k1.time - k0.time
        m0 = k0.out_tangent[1] / max(k0.out_tangent[0], 0.001) if k0.out_tangent[0] != 0 else 0
        m1 = k1.in_tangent[1] / max(abs(k1.in_tangent[0]), 0.001) if k1.in_tangent[0] != 0 else 0

        # Hermite basis functions
        u2 = u * u
        u3 = u2 * u
        h00 = 2*u3 - 3*u2 + 1
        h10 = u3 - 2*u2 + u
        h01 = -2*u3 + 3*u2
        h11 = u3 - u2

        return h00 * p0 + h10 * duration * m0 + h01 * p1 + h11 * duration * m1

    def add_keyframe(self, time: float, value: float,
                     in_tangent: Tuple[float, float] = (0.0, 0.0),
                     out_tangent: Tuple[float, float] = (0.0, 0.0)):
        """Add a keyframe to the channel."""
        self.keyframes.append(Keyframe(time, value, in_tangent, out_tangent))
        self.keyframes.sort(key=lambda k: k.time)


@dataclass
class Marker:
    """Named sync point in the track."""
    time: float
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {'time': self.time, 'name': self.name}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Marker':
        return Marker(time=float(data['time']), name=data['name'])


@dataclass
class BlendRegion:
    """Region where live affect can blend with track."""
    start: float
    end: float
    live_weight: float = 0.3  # How much live affect influences (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {'start': self.start, 'end': self.end, 'live_weight': self.live_weight}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BlendRegion':
        return BlendRegion(
            start=float(data['start']),
            end=float(data['end']),
            live_weight=float(data.get('live_weight', 0.3))
        )


@dataclass
class TrackEvent:
    """Event to trigger at specific time."""
    time: float
    event: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {'time': self.time, 'event': self.event, 'data': self.data}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TrackEvent':
        return TrackEvent(
            time=float(data['time']),
            event=data['event'],
            data=data.get('data', {})
        )


# Default affect model: PAD+BS (Pleasure-Arousal-Dominance + Boredom-Sorrow)
# But the system supports arbitrary named channels for alien intelligences, etc.
DEFAULT_AFFECT_CHANNELS = ['valence', 'arousal', 'dominance', 'boredom', 'sorrow']

# Channel defaults (neutral state)
CHANNEL_DEFAULTS = {
    'valence': 0.0,      # -1 to +1 (pleasure)
    'arousal': 0.5,      # 0 to 1 (activation)
    'dominance': 0.5,    # 0 to 1 (control)
    'boredom': 0.0,      # 0 to 1
    'sorrow': 0.0,       # 0 to 1
}


@dataclass
class AffectState:
    """
    Arbitrary-dimensional affect state.

    Default is PAD+BS (5D) but can hold any named dimensions.
    Alien intelligences might have: curiosity, aggression, hunger, hive_resonance...
    """
    channels: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with defaults if empty."""
        if not self.channels:
            self.channels = dict(CHANNEL_DEFAULTS)

    # Convenience accessors for PAD+BS (the default model)
    @property
    def valence(self) -> float:
        return self.channels.get('valence', 0.0)

    @valence.setter
    def valence(self, v: float):
        self.channels['valence'] = v

    @property
    def arousal(self) -> float:
        return self.channels.get('arousal', 0.5)

    @arousal.setter
    def arousal(self, v: float):
        self.channels['arousal'] = v

    @property
    def dominance(self) -> float:
        return self.channels.get('dominance', 0.5)

    @dominance.setter
    def dominance(self, v: float):
        self.channels['dominance'] = v

    @property
    def boredom(self) -> float:
        return self.channels.get('boredom', 0.0)

    @boredom.setter
    def boredom(self, v: float):
        self.channels['boredom'] = v

    @property
    def sorrow(self) -> float:
        return self.channels.get('sorrow', 0.0)

    @sorrow.setter
    def sorrow(self, v: float):
        self.channels['sorrow'] = v

    def get(self, channel: str, default: float = 0.0) -> float:
        """Get any channel value."""
        return self.channels.get(channel, default)

    def set(self, channel: str, value: float):
        """Set any channel value."""
        self.channels[channel] = value

    def to_dict(self) -> Dict[str, float]:
        return dict(self.channels)

    def to_list(self, channel_order: Optional[List[str]] = None) -> List[float]:
        """Convert to list in specified channel order."""
        order = channel_order or DEFAULT_AFFECT_CHANNELS
        return [self.channels.get(ch, CHANNEL_DEFAULTS.get(ch, 0.0)) for ch in order]

    @staticmethod
    def from_dict(data: Dict[str, float]) -> 'AffectState':
        return AffectState(channels=dict(data))

    @staticmethod
    def neutral() -> 'AffectState':
        """Create neutral PAD+BS state."""
        return AffectState(channels=dict(CHANNEL_DEFAULTS))

    @staticmethod
    def from_pad_bs(valence: float = 0.0, arousal: float = 0.5,
                    dominance: float = 0.5, boredom: float = 0.0,
                    sorrow: float = 0.0) -> 'AffectState':
        """Create from PAD+BS values (convenience constructor)."""
        return AffectState(channels={
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'boredom': boredom,
            'sorrow': sorrow
        })


@dataclass
class AffectTrack:
    """
    Complete affect animation track.

    Contains keyframed curves for arbitrary affect dimensions.
    Default is PAD+BS (valence, arousal, dominance, boredom, sorrow)
    but can include any custom channels (curiosity, aggression, hive_resonance...).

    The animator has full granularity - any channel can be keyframed.
    """
    name: str = "Untitled Track"
    duration: float = 0.0
    fps: int = 30
    author: str = ""
    created: str = ""
    tags: List[str] = field(default_factory=list)

    # Affect channels - arbitrary named dimensions
    # Default is PAD+BS but can be extended or replaced
    channels: Dict[str, AffectChannel] = field(default_factory=dict)

    # Affect model metadata (what channels this track expects)
    affect_model: str = "PAD+BS"  # Could be "PAD", "PAD+BS", "Alien_Hive_v1", etc.
    channel_definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Sync points and events
    markers: List[Marker] = field(default_factory=list)
    events: List[TrackEvent] = field(default_factory=list)
    blend_regions: List[BlendRegion] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default channels if none specified."""
        if not self.channels:
            # Default to PAD+BS model
            for ch_name in DEFAULT_AFFECT_CHANNELS:
                self.channels[ch_name] = AffectChannel(name=ch_name)

        # Set up channel definitions with defaults if not specified
        if not self.channel_definitions:
            self.channel_definitions = {
                'valence': {'min': -1.0, 'max': 1.0, 'default': 0.0, 'description': 'Pleasure/displeasure'},
                'arousal': {'min': 0.0, 'max': 1.0, 'default': 0.5, 'description': 'Activation/energy'},
                'dominance': {'min': 0.0, 'max': 1.0, 'default': 0.5, 'description': 'Control/confidence'},
                'boredom': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'description': 'Disengagement'},
                'sorrow': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'description': 'Sadness/grief'},
            }

    def add_channel(self, name: str, min_val: float = 0.0, max_val: float = 1.0,
                    default: float = 0.0, description: str = "",
                    interpolation: InterpolationType = InterpolationType.LINEAR):
        """
        Add a custom affect channel.

        For alien intelligences, custom emotion models, etc.

        Args:
            name: Channel name (e.g., "curiosity", "hive_resonance")
            min_val: Minimum value
            max_val: Maximum value
            default: Default/neutral value
            description: Human-readable description
            interpolation: Default interpolation type for this channel
        """
        self.channels[name] = AffectChannel(name=name, interpolation=interpolation)
        self.channel_definitions[name] = {
            'min': min_val,
            'max': max_val,
            'default': default,
            'description': description
        }

    def get_channel_names(self) -> List[str]:
        """Get list of all channel names in this track."""
        return list(self.channels.keys())

    def sample(self, t: float) -> AffectState:
        """
        Sample all channels at time t.

        Args:
            t: Time in seconds

        Returns:
            AffectState with all channels (arbitrary dimensions)
        """
        channel_values = {}
        for ch_name, channel in self.channels.items():
            channel_values[ch_name] = channel.sample(t)
        return AffectState(channels=channel_values)

    def get_live_blend_weight(self, t: float) -> float:
        """
        Get the live affect blend weight at time t.

        Returns:
            0.0 = 100% track, 1.0 = 100% live
        """
        for region in self.blend_regions:
            if region.start <= t <= region.end:
                return region.live_weight
        return 0.0  # Full track weight outside blend regions

    def get_markers_at(self, t: float, tolerance: float = 0.05) -> List[Marker]:
        """Get markers within tolerance of time t."""
        return [m for m in self.markers if abs(m.time - t) <= tolerance]

    def get_events_at(self, t: float, tolerance: float = 0.05) -> List[TrackEvent]:
        """Get events within tolerance of time t."""
        return [e for e in self.events if abs(e.time - t) <= tolerance]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML export."""
        return {
            'format': 'affect-track',
            'version': '1.0',
            'metadata': {
                'name': self.name,
                'duration': self.duration,
                'fps': self.fps,
                'author': self.author,
                'created': self.created,
                'tags': self.tags,
                'affect_model': self.affect_model
            },
            'channel_definitions': self.channel_definitions,
            'channels': {name: ch.to_dict() for name, ch in self.channels.items()},
            'markers': [m.to_dict() for m in self.markers],
            'events': [e.to_dict() for e in self.events],
            'blend_regions': [br.to_dict() for br in self.blend_regions]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AffectTrack':
        """Deserialize from dictionary."""
        metadata = data.get('metadata', {})
        channels_data = data.get('channels', {})

        channels = {}
        for name, ch_data in channels_data.items():
            channels[name] = AffectChannel.from_dict(name, ch_data)

        return AffectTrack(
            name=metadata.get('name', 'Untitled Track'),
            duration=float(metadata.get('duration', 0.0)),
            fps=int(metadata.get('fps', 30)),
            author=metadata.get('author', ''),
            created=metadata.get('created', ''),
            tags=metadata.get('tags', []),
            affect_model=metadata.get('affect_model', 'PAD+BS'),
            channel_definitions=data.get('channel_definitions', {}),
            channels=channels,
            markers=[Marker.from_dict(m) for m in data.get('markers', [])],
            events=[TrackEvent.from_dict(e) for e in data.get('events', [])],
            blend_regions=[BlendRegion.from_dict(br) for br in data.get('blend_regions', [])]
        )

    def save_yaml(self, filepath: str):
        """Save track to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install PyYAML")
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def load_yaml(filepath: str) -> 'AffectTrack':
        """Load track from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install PyYAML")
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return AffectTrack.from_dict(data)

    def add_keyframe(self, channel: str, time: float, value: float,
                     in_tangent: Tuple[float, float] = (0.0, 0.0),
                     out_tangent: Tuple[float, float] = (0.0, 0.0)):
        """Add a keyframe to a channel."""
        if channel not in self.channels:
            self.channels[channel] = AffectChannel(name=channel)
        self.channels[channel].add_keyframe(time, value, in_tangent, out_tangent)

        # Update duration
        for ch in self.channels.values():
            for kf in ch.keyframes:
                if kf.time > self.duration:
                    self.duration = kf.time


class AffectTrackPlayer:
    """
    Plays an affect track with timing control.

    Supports play/pause/seek/speed control and marker callbacks.
    """

    def __init__(self, track: AffectTrack):
        self.track = track
        self.current_time: float = 0.0
        self.speed: float = 1.0
        self.is_playing: bool = False
        self.is_looping: bool = False

        # Completion behavior
        self.on_complete: TrackCompletionBehavior = TrackCompletionBehavior.HOLD
        self.transfer_scale: float = 1.0
        self.crossfade_duration: float = 0.5

        # Callbacks
        self.marker_callbacks: Dict[str, List[Callable]] = {}
        self.event_callback: Optional[Callable] = None
        self.completion_callback: Optional[Callable] = None

        # State tracking
        self._last_update_time: float = 0.0
        self._triggered_markers: set = set()
        self._triggered_events: set = set()

    def play(self):
        """Start playback."""
        self.is_playing = True
        self._last_update_time = time.time()

    def pause(self):
        """Pause playback."""
        self.is_playing = False

    def stop(self):
        """Stop and reset to beginning."""
        self.is_playing = False
        self.current_time = 0.0
        self._triggered_markers.clear()
        self._triggered_events.clear()

    def seek(self, t: float):
        """Jump to specific time."""
        self.current_time = max(0.0, min(t, self.track.duration))
        # Clear triggered sets for markers/events we've passed
        self._triggered_markers = {m for m in self._triggered_markers if m > self.current_time}
        self._triggered_events = {e for e in self._triggered_events if e > self.current_time}

    def update(self) -> AffectState:
        """
        Update playback and return current affect state.

        Should be called each frame/tick.

        Returns:
            Current AffectState
        """
        if self.is_playing:
            now = time.time()
            delta = (now - self._last_update_time) * self.speed
            self._last_update_time = now

            old_time = self.current_time
            self.current_time += delta

            # Check for markers/events in the time we just passed
            self._check_triggers(old_time, self.current_time)

            # Handle end of track
            if self.current_time >= self.track.duration:
                if self.is_looping:
                    self.current_time = self.current_time % self.track.duration
                    self._triggered_markers.clear()
                    self._triggered_events.clear()
                else:
                    self.current_time = self.track.duration
                    self.is_playing = False
                    if self.completion_callback:
                        self.completion_callback(self)

        return self.track.sample(self.current_time)

    def sample(self, t: Optional[float] = None) -> AffectState:
        """Sample track at time t (or current time if not specified)."""
        return self.track.sample(t if t is not None else self.current_time)

    def get_live_blend_weight(self) -> float:
        """Get current live affect blend weight."""
        return self.track.get_live_blend_weight(self.current_time)

    def on_marker(self, marker_name: str, callback: Callable):
        """Register callback for when a marker is reached."""
        if marker_name not in self.marker_callbacks:
            self.marker_callbacks[marker_name] = []
        self.marker_callbacks[marker_name].append(callback)

    def _check_triggers(self, old_t: float, new_t: float):
        """Check and trigger markers/events between old and new time."""
        # Check markers
        for marker in self.track.markers:
            if old_t < marker.time <= new_t and marker.time not in self._triggered_markers:
                self._triggered_markers.add(marker.time)
                if marker.name in self.marker_callbacks:
                    for cb in self.marker_callbacks[marker.name]:
                        cb()

        # Check events
        for event in self.track.events:
            if old_t < event.time <= new_t and event.time not in self._triggered_events:
                self._triggered_events.add(event.time)
                if self.event_callback:
                    self.event_callback(event.event, event.data)


class AffectTrackFacet:
    """
    Facet that plays affect tracks in the cognitive assembly.

    Integrates with FacetExecutor to provide authored affect values
    that can override or blend with live CharmNetwork output.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize affect track facet.

        Config options:
            track: Path to .affecttrack file
            trigger: Event name that starts playback (optional)
            blend_mode: 'override', 'blend', 'additive', 'multiplicative', 'maximum'
            blend_weight: Weight for blend mode (0-1)
            loop: Whether to loop playback
            speed: Playback speed multiplier
            on_complete: 'momentum', 'snap_neutral', 'hold', 'loop'
            transfer_scale: Scale for momentum transfer (0-1)
            crossfade_duration: Duration of crossfade from track to live
        """
        self.config = config
        self.track: Optional[AffectTrack] = None
        self.player: Optional[AffectTrackPlayer] = None

        # Load track if specified
        track_path = config.get('track')
        if track_path and os.path.exists(track_path):
            self.track = AffectTrack.load_yaml(track_path)
            self.player = AffectTrackPlayer(self.track)
            self.player.speed = config.get('speed', 1.0)
            self.player.is_looping = config.get('loop', False)
            self.player.on_complete = TrackCompletionBehavior(config.get('on_complete', 'hold'))
            self.player.transfer_scale = config.get('transfer_scale', 1.0)
            self.player.crossfade_duration = config.get('crossfade_duration', 0.5)

        self.blend_mode = config.get('blend_mode', 'override')
        self.blend_weight = config.get('blend_weight', 1.0)
        self.trigger = config.get('trigger')

        # State
        self.is_active = False

        # CharmNetwork reference for momentum handoff
        self.charm_network = None

        # Execution stats
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0

    def set_charm_network(self, charm_network):
        """Set reference to CharmNetwork for momentum handoff."""
        self.charm_network = charm_network

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and return affect values.

        Args:
            inputs: May contain:
                - trigger: Event name to start playback
                - live_affect: Current CharmNetwork affect (for blending)

        Returns:
            Dict with blended affect values
        """
        start_time = time.time()

        # Check for trigger
        if self.trigger and inputs.get('trigger') == self.trigger:
            self.start_playback()

        # Get track affect (or neutral if not playing)
        if self.player and self.player.is_playing:
            track_affect = self.player.update()
        elif self.player:
            track_affect = self.player.sample()
        else:
            track_affect = AffectState()

        # Blend with live affect if provided
        live_affect = inputs.get('live_affect')
        if live_affect and isinstance(live_affect, dict):
            live_state = AffectState.from_dict(live_affect)
            blended = self._blend_affect(track_affect, live_state)
        else:
            blended = track_affect

        # Record execution
        elapsed = time.time() - start_time
        self.execution_count += 1
        self.total_execution_time += elapsed
        self.last_execution_time = elapsed

        return {
            'affect': blended.to_dict(),
            'is_playing': self.player.is_playing if self.player else False,
            'current_time': self.player.current_time if self.player else 0.0,
            'duration': self.track.duration if self.track else 0.0
        }

    def start_playback(self, from_time: float = 0.0):
        """Start track playback."""
        if self.player:
            self.player.seek(from_time)
            self.player.play()
            self.is_active = True

    def stop_playback(self):
        """Stop track playback."""
        if self.player:
            self.player.stop()
            self.is_active = False

    def on_track_complete(self):
        """
        Handle track completion with momentum handoff.

        Called when track finishes - transfers final affect state
        to CharmNetwork for natural decay.
        """
        if not self.player or not self.charm_network:
            return

        behavior = self.player.on_complete

        if behavior == TrackCompletionBehavior.MOMENTUM:
            # Get final affect values
            final_affect = self.player.sample(self.track.duration)

            # Inject into CharmNetwork
            # The charm network will let these decay naturally
            if hasattr(self.charm_network, 'inject_state'):
                self.charm_network.inject_state(
                    valence=final_affect.valence * self.player.transfer_scale,
                    arousal=final_affect.arousal * self.player.transfer_scale,
                    dominance=final_affect.dominance * self.player.transfer_scale,
                    boredom=final_affect.boredom * self.player.transfer_scale,
                    sorrow=final_affect.sorrow * self.player.transfer_scale,
                    crossfade=self.player.crossfade_duration
                )

        elif behavior == TrackCompletionBehavior.SNAP_NEUTRAL:
            # Nothing to do - affect returns to live CharmNetwork
            pass

        elif behavior == TrackCompletionBehavior.LOOP:
            self.player.seek(0.0)
            self.player.play()

    def _blend_affect(self, track: AffectState, live: AffectState) -> AffectState:
        """
        Blend track affect with live CharmNetwork affect.

        Blend modes:
        - override: 100% track
        - blend: Weighted average
        - additive: Track offsets live values
        - multiplicative: Track scales live values
        - maximum: Take more extreme value
        """
        w = self.blend_weight

        # Also consider track's blend regions
        if self.player:
            region_live_weight = self.player.get_live_blend_weight()
            if region_live_weight > 0:
                # Blend region wants more live influence
                w = w * (1 - region_live_weight)

        if self.blend_mode == 'override':
            return track

        elif self.blend_mode == 'blend' or self.blend_mode == 'weighted':
            return AffectState(
                valence=track.valence * w + live.valence * (1 - w),
                arousal=track.arousal * w + live.arousal * (1 - w),
                dominance=track.dominance * w + live.dominance * (1 - w),
                boredom=track.boredom * w + live.boredom * (1 - w),
                sorrow=track.sorrow * w + live.sorrow * (1 - w)
            )

        elif self.blend_mode == 'additive':
            # Track offsets from neutral (0.5) applied to live
            return AffectState(
                valence=max(-1, min(1, live.valence + (track.valence - 0) * w)),
                arousal=max(0, min(1, live.arousal + (track.arousal - 0.5) * w)),
                dominance=max(0, min(1, live.dominance + (track.dominance - 0.5) * w)),
                boredom=max(0, min(1, live.boredom + (track.boredom - 0) * w)),
                sorrow=max(0, min(1, live.sorrow + (track.sorrow - 0) * w))
            )

        elif self.blend_mode == 'multiplicative':
            return AffectState(
                valence=live.valence * (1 + (track.valence - 0) * w),
                arousal=live.arousal * track.arousal,
                dominance=live.dominance * track.dominance,
                boredom=live.boredom * track.boredom,
                sorrow=live.sorrow * track.sorrow
            )

        elif self.blend_mode == 'maximum':
            # Take more extreme value
            return AffectState(
                valence=track.valence if abs(track.valence) > abs(live.valence) else live.valence,
                arousal=max(track.arousal, live.arousal),
                dominance=max(track.dominance, live.dominance),
                boredom=max(track.boredom, live.boredom),
                sorrow=max(track.sorrow, live.sorrow)
            )

        return track  # Default to track

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.execution_count,
            'total_tokens': 0,  # No LLM tokens
            'avg_tokens': 0,
            'total_time': self.total_execution_time,
            'avg_time': (
                self.total_execution_time / self.execution_count
                if self.execution_count > 0 else 0
            ),
            'last_tokens': 0,
            'last_time': self.last_execution_time
        }

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage (always 0 - no LLM calls)."""
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.execution_count,
            'avg_tokens': 0
        }


def create_example_track() -> AffectTrack:
    """
    Create an example affect track for testing.

    Demonstrates "receiving bad news" emotional arc.
    """
    track = AffectTrack(
        name="Receiving Bad News",
        duration=8.5,
        author="NoodleStudio",
        tags=["dramatic", "grief", "reaction"]
    )

    # Valence: starts positive, drops on news, settles into grief
    track.channels['valence'].interpolation = InterpolationType.BEZIER
    track.add_keyframe('valence', 0.0, 0.6)
    track.add_keyframe('valence', 1.2, 0.1, out_tangent=(0.2, -0.5))  # Shock
    track.add_keyframe('valence', 3.5, -0.4)  # Grief settles
    track.add_keyframe('valence', 8.5, -0.2)  # Numb acceptance

    # Arousal: spikes on news, then exhaustion
    track.channels['arousal'].interpolation = InterpolationType.BEZIER
    track.add_keyframe('arousal', 0.0, 0.4)
    track.add_keyframe('arousal', 1.0, 0.9)  # Spike
    track.add_keyframe('arousal', 2.5, 0.7)
    track.add_keyframe('arousal', 8.5, 0.3)  # Exhausted

    # Dominance: loses control then slowly regains
    track.channels['dominance'].interpolation = InterpolationType.BEZIER
    track.add_keyframe('dominance', 0.0, 0.7)
    track.add_keyframe('dominance', 1.5, 0.2)  # Lost control
    track.add_keyframe('dominance', 6.0, 0.5)  # Regaining
    track.add_keyframe('dominance', 8.5, 0.6)

    # Boredom: stays 0 (fully engaged throughout)
    track.add_keyframe('boredom', 0.0, 0.0)
    track.add_keyframe('boredom', 8.5, 0.0)

    # Sorrow: emerges and peaks
    track.channels['sorrow'].interpolation = InterpolationType.BEZIER
    track.add_keyframe('sorrow', 0.0, 0.0)
    track.add_keyframe('sorrow', 2.0, 0.3)
    track.add_keyframe('sorrow', 5.0, 0.7)  # Peak grief
    track.add_keyframe('sorrow', 8.5, 0.5)  # Lingering

    # Markers
    track.markers = [
        Marker(time=1.0, name="news_delivered"),
        Marker(time=3.5, name="tears_start"),
        Marker(time=6.0, name="composure_begins")
    ]

    # Blend region - allow some live affect influence during processing
    track.blend_regions = [
        BlendRegion(start=4.0, end=6.0, live_weight=0.3)
    ]

    # Events
    track.events = [
        TrackEvent(time=1.0, event="play_sound", data={"clip": "gasp.ogg"}),
        TrackEvent(time=3.5, event="start_tears", data={"intensity": 0.6})
    ]

    return track


if __name__ == "__main__":
    # Test: Create and save example track
    print("Creating example affect track...")
    track = create_example_track()

    # Save to YAML
    test_path = "/tmp/test_affect_track.affecttrack"
    track.save_yaml(test_path)
    print(f"Saved to {test_path}")

    # Load it back
    loaded = AffectTrack.load_yaml(test_path)
    print(f"Loaded: {loaded.name}, duration: {loaded.duration}s")

    # Test sampling
    print("\n=== Sampling test ===")
    for t in [0.0, 1.0, 2.0, 4.0, 6.0, 8.5]:
        state = loaded.sample(t)
        print(f"t={t:.1f}s: v={state.valence:.2f} a={state.arousal:.2f} "
              f"d={state.dominance:.2f} b={state.boredom:.2f} s={state.sorrow:.2f}")

    # Test player
    print("\n=== Player test ===")
    player = AffectTrackPlayer(loaded)
    player.play()

    for _ in range(10):
        state = player.update()
        print(f"t={player.current_time:.2f}s: v={state.valence:.2f} a={state.arousal:.2f}")
        time.sleep(0.1)

    print("\nAffect Track system working!")
