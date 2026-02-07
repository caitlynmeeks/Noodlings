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
#   Pose Track - Keyframeable body animation using muscle space
#
#   Rig-agnostic body animation inspired by Unity's Mecanim m...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.pose_track
# PURPOSE:  Pose Track
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   InterpolationType, TrackCompletionBehavior, Keyframe, MuscleChannel, BlendShapeChannel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# =============================================================================
# Interpolation (shared with affect_track.py)
# =============================================================================

class InterpolationType(Enum):
    """Curve interpolation method."""
    LINEAR = "linear"
    BEZIER = "bezier"
    STEP = "step"
    HERMITE = "hermite"


class TrackCompletionBehavior(Enum):
    """What happens when the track finishes playing."""
    SPRING = "spring"          # Spring back to neutral pose
    HOLD = "hold"              # Hold final pose indefinitely
    LOOP = "loop"              # Loop back to start


# =============================================================================
# Standard Humanoid Muscles (~47 like Mecanim)
# =============================================================================

HUMANOID_MUSCLES = [
    # Body/Spine (9)
    'Spine.FrontBack', 'Spine.LeftRight', 'Spine.TwistLeftRight',
    'Chest.FrontBack', 'Chest.LeftRight', 'Chest.TwistLeftRight',
    'UpperChest.FrontBack', 'UpperChest.LeftRight', 'UpperChest.TwistLeftRight',

    # Neck/Head (6)
    'Neck.NodDownUp', 'Neck.TiltLeftRight', 'Neck.TurnLeftRight',
    'Head.NodDownUp', 'Head.TiltLeftRight', 'Head.TurnLeftRight',

    # Eyes/Jaw (5)
    'LeftEye.DownUp', 'LeftEye.InOut',
    'RightEye.DownUp', 'RightEye.InOut',
    'Jaw.Close',

    # Left Arm (9)
    'LeftShoulder.DownUp', 'LeftShoulder.FrontBack',
    'LeftArm.DownUp', 'LeftArm.FrontBack', 'LeftArm.TwistInOut',
    'LeftForeArm.Stretch', 'LeftForeArm.TwistInOut',
    'LeftHand.DownUp', 'LeftHand.InOut',

    # Right Arm (9)
    'RightShoulder.DownUp', 'RightShoulder.FrontBack',
    'RightArm.DownUp', 'RightArm.FrontBack', 'RightArm.TwistInOut',
    'RightForeArm.Stretch', 'RightForeArm.TwistInOut',
    'RightHand.DownUp', 'RightHand.InOut',

    # Left Leg (8)
    'LeftUpperLeg.FrontBack', 'LeftUpperLeg.InOut', 'LeftUpperLeg.TwistInOut',
    'LeftLowerLeg.Stretch', 'LeftLowerLeg.TwistInOut',
    'LeftFoot.UpDown', 'LeftFoot.TwistInOut',
    'LeftToes.UpDown',

    # Right Leg (8)
    'RightUpperLeg.FrontBack', 'RightUpperLeg.InOut', 'RightUpperLeg.TwistInOut',
    'RightLowerLeg.Stretch', 'RightLowerLeg.TwistInOut',
    'RightFoot.UpDown', 'RightFoot.TwistInOut',
    'RightToes.UpDown',
]

# Default muscle definitions (min/max in degrees, default value)
MUSCLE_DEFINITIONS = {
    # Spine
    'Spine.FrontBack': {'axis': 'X', 'min': -40, 'max': 40, 'default': 0.0},
    'Spine.LeftRight': {'axis': 'Z', 'min': -40, 'max': 40, 'default': 0.0},
    'Spine.TwistLeftRight': {'axis': 'Y', 'min': -40, 'max': 40, 'default': 0.0},
    'Chest.FrontBack': {'axis': 'X', 'min': -40, 'max': 40, 'default': 0.0},
    'Chest.LeftRight': {'axis': 'Z', 'min': -40, 'max': 40, 'default': 0.0},
    'Chest.TwistLeftRight': {'axis': 'Y', 'min': -40, 'max': 40, 'default': 0.0},
    'UpperChest.FrontBack': {'axis': 'X', 'min': -20, 'max': 20, 'default': 0.0},
    'UpperChest.LeftRight': {'axis': 'Z', 'min': -20, 'max': 20, 'default': 0.0},
    'UpperChest.TwistLeftRight': {'axis': 'Y', 'min': -20, 'max': 20, 'default': 0.0},

    # Head/Neck
    'Neck.NodDownUp': {'axis': 'X', 'min': -40, 'max': 40, 'default': 0.0},
    'Neck.TiltLeftRight': {'axis': 'Z', 'min': -40, 'max': 40, 'default': 0.0},
    'Neck.TurnLeftRight': {'axis': 'Y', 'min': -40, 'max': 40, 'default': 0.0},
    'Head.NodDownUp': {'axis': 'X', 'min': -40, 'max': 60, 'default': 0.0},
    'Head.TiltLeftRight': {'axis': 'Z', 'min': -40, 'max': 40, 'default': 0.0},
    'Head.TurnLeftRight': {'axis': 'Y', 'min': -70, 'max': 70, 'default': 0.0},

    # Eyes
    'LeftEye.DownUp': {'axis': 'X', 'min': -15, 'max': 12, 'default': 0.0},
    'LeftEye.InOut': {'axis': 'Y', 'min': -20, 'max': 20, 'default': 0.0},
    'RightEye.DownUp': {'axis': 'X', 'min': -15, 'max': 12, 'default': 0.0},
    'RightEye.InOut': {'axis': 'Y', 'min': -20, 'max': 20, 'default': 0.0},
    'Jaw.Close': {'axis': 'X', 'min': 0, 'max': 30, 'default': 0.0},

    # Arms
    'LeftShoulder.DownUp': {'axis': 'Z', 'min': -15, 'max': 30, 'default': 0.0},
    'LeftShoulder.FrontBack': {'axis': 'Y', 'min': -15, 'max': 15, 'default': 0.0},
    'LeftArm.DownUp': {'axis': 'Z', 'min': -60, 'max': 100, 'default': 0.0},
    'LeftArm.FrontBack': {'axis': 'X', 'min': -100, 'max': 60, 'default': 0.0},
    'LeftArm.TwistInOut': {'axis': 'Y', 'min': -90, 'max': 50, 'default': 0.0},
    'LeftForeArm.Stretch': {'axis': 'X', 'min': 0, 'max': 145, 'default': 0.0},
    'LeftForeArm.TwistInOut': {'axis': 'Y', 'min': -90, 'max': 90, 'default': 0.0},
    'LeftHand.DownUp': {'axis': 'X', 'min': -80, 'max': 80, 'default': 0.0},
    'LeftHand.InOut': {'axis': 'Z', 'min': -40, 'max': 25, 'default': 0.0},

    'RightShoulder.DownUp': {'axis': 'Z', 'min': -15, 'max': 30, 'default': 0.0},
    'RightShoulder.FrontBack': {'axis': 'Y', 'min': -15, 'max': 15, 'default': 0.0},
    'RightArm.DownUp': {'axis': 'Z', 'min': -60, 'max': 100, 'default': 0.0},
    'RightArm.FrontBack': {'axis': 'X', 'min': -100, 'max': 60, 'default': 0.0},
    'RightArm.TwistInOut': {'axis': 'Y', 'min': -90, 'max': 50, 'default': 0.0},
    'RightForeArm.Stretch': {'axis': 'X', 'min': 0, 'max': 145, 'default': 0.0},
    'RightForeArm.TwistInOut': {'axis': 'Y', 'min': -90, 'max': 90, 'default': 0.0},
    'RightHand.DownUp': {'axis': 'X', 'min': -80, 'max': 80, 'default': 0.0},
    'RightHand.InOut': {'axis': 'Z', 'min': -40, 'max': 25, 'default': 0.0},

    # Legs
    'LeftUpperLeg.FrontBack': {'axis': 'X', 'min': -90, 'max': 50, 'default': 0.0},
    'LeftUpperLeg.InOut': {'axis': 'Z', 'min': -60, 'max': 60, 'default': 0.0},
    'LeftUpperLeg.TwistInOut': {'axis': 'Y', 'min': -60, 'max': 60, 'default': 0.0},
    'LeftLowerLeg.Stretch': {'axis': 'X', 'min': 0, 'max': 145, 'default': 0.0},
    'LeftLowerLeg.TwistInOut': {'axis': 'Y', 'min': -45, 'max': 45, 'default': 0.0},
    'LeftFoot.UpDown': {'axis': 'X', 'min': -50, 'max': 50, 'default': 0.0},
    'LeftFoot.TwistInOut': {'axis': 'Y', 'min': -30, 'max': 30, 'default': 0.0},
    'LeftToes.UpDown': {'axis': 'X', 'min': -50, 'max': 50, 'default': 0.0},

    'RightUpperLeg.FrontBack': {'axis': 'X', 'min': -90, 'max': 50, 'default': 0.0},
    'RightUpperLeg.InOut': {'axis': 'Z', 'min': -60, 'max': 60, 'default': 0.0},
    'RightUpperLeg.TwistInOut': {'axis': 'Y', 'min': -60, 'max': 60, 'default': 0.0},
    'RightLowerLeg.Stretch': {'axis': 'X', 'min': 0, 'max': 145, 'default': 0.0},
    'RightLowerLeg.TwistInOut': {'axis': 'Y', 'min': -45, 'max': 45, 'default': 0.0},
    'RightFoot.UpDown': {'axis': 'X', 'min': -50, 'max': 50, 'default': 0.0},
    'RightFoot.TwistInOut': {'axis': 'Y', 'min': -30, 'max': 30, 'default': 0.0},
    'RightToes.UpDown': {'axis': 'X', 'min': -50, 'max': 50, 'default': 0.0},
}

# VRM blend shape presets
BLENDSHAPE_PRESETS = [
    'happy', 'angry', 'sad', 'relaxed', 'surprised',
    'aa', 'ih', 'ou', 'ee', 'oh',  # Visemes
    'blink', 'blinkLeft', 'blinkRight',
    'lookUp', 'lookDown', 'lookLeft', 'lookRight',
    'neutral',
]


# =============================================================================
# Keyframe and Channel (same pattern as affect_track.py)
# =============================================================================

@dataclass
class Keyframe:
    """A single keyframe on a pose curve."""
    time: float
    value: float  # Normalized muscle value [-1, 1]

    # Bezier tangent handles
    in_tangent: Tuple[float, float] = (0.0, 0.0)
    out_tangent: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        result = {'time': self.time, 'value': self.value}
        if self.in_tangent != (0.0, 0.0):
            result['in_tangent'] = list(self.in_tangent)
        if self.out_tangent != (0.0, 0.0):
            result['out_tangent'] = list(self.out_tangent)
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Keyframe':
        in_tan = data.get('in_tangent', [0.0, 0.0])
        out_tan = data.get('out_tangent', [0.0, 0.0])
        return Keyframe(
            time=float(data['time']),
            value=float(data['value']),
            in_tangent=tuple(in_tan) if isinstance(in_tan, list) else in_tan,
            out_tangent=tuple(out_tan) if isinstance(out_tan, list) else out_tan
        )


@dataclass
class MuscleChannel:
    """A single muscle dimension with keyframed curve."""
    name: str
    interpolation: InterpolationType = InterpolationType.LINEAR
    keyframes: List[Keyframe] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'interpolation': self.interpolation.value,
            'keyframes': [kf.to_dict() for kf in self.keyframes]
        }

    @staticmethod
    def from_dict(name: str, data: Dict[str, Any]) -> 'MuscleChannel':
        return MuscleChannel(
            name=name,
            interpolation=InterpolationType(data.get('interpolation', 'linear')),
            keyframes=[Keyframe.from_dict(kf) for kf in data.get('keyframes', [])]
        )

    def sample(self, t: float) -> float:
        """Sample the channel at time t."""
        if not self.keyframes:
            return 0.0

        kfs = sorted(self.keyframes, key=lambda k: k.time)

        if t <= kfs[0].time:
            return kfs[0].value
        if t >= kfs[-1].time:
            return kfs[-1].value

        for i in range(len(kfs) - 1):
            if kfs[i].time <= t <= kfs[i + 1].time:
                return self._interpolate(kfs[i], kfs[i + 1], t)

        return kfs[-1].value

    def _interpolate(self, k0: Keyframe, k1: Keyframe, t: float) -> float:
        """Interpolate between two keyframes."""
        if self.interpolation == InterpolationType.STEP:
            return k0.value

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
        """Cubic bezier interpolation."""
        p0 = k0.value
        p3 = k1.value
        p1 = p0 + k0.out_tangent[1]
        p2 = p3 - k1.in_tangent[1]

        u2 = u * u
        u3 = u2 * u
        inv_u = 1.0 - u
        inv_u2 = inv_u * inv_u
        inv_u3 = inv_u2 * inv_u

        return inv_u3 * p0 + 3 * inv_u2 * u * p1 + 3 * inv_u * u2 * p2 + u3 * p3

    def _hermite_interpolate(self, k0: Keyframe, k1: Keyframe, u: float) -> float:
        """Hermite spline interpolation."""
        p0 = k0.value
        p1 = k1.value
        duration = k1.time - k0.time
        m0 = k0.out_tangent[1] / max(k0.out_tangent[0], 0.001) if k0.out_tangent[0] != 0 else 0
        m1 = k1.in_tangent[1] / max(abs(k1.in_tangent[0]), 0.001) if k1.in_tangent[0] != 0 else 0

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
class BlendShapeChannel:
    """A blend shape with keyframed weight."""
    name: str
    interpolation: InterpolationType = InterpolationType.LINEAR
    keyframes: List[Keyframe] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'interpolation': self.interpolation.value,
            'keyframes': [kf.to_dict() for kf in self.keyframes]
        }

    @staticmethod
    def from_dict(name: str, data: Dict[str, Any]) -> 'BlendShapeChannel':
        return BlendShapeChannel(
            name=name,
            interpolation=InterpolationType(data.get('interpolation', 'linear')),
            keyframes=[Keyframe.from_dict(kf) for kf in data.get('keyframes', [])]
        )

    def sample(self, t: float) -> float:
        """Sample at time t (same logic as MuscleChannel)."""
        if not self.keyframes:
            return 0.0

        kfs = sorted(self.keyframes, key=lambda k: k.time)

        if t <= kfs[0].time:
            return kfs[0].value
        if t >= kfs[-1].time:
            return kfs[-1].value

        for i in range(len(kfs) - 1):
            if kfs[i].time <= t <= kfs[i + 1].time:
                # Linear interpolation for blend shapes
                duration = kfs[i + 1].time - kfs[i].time
                if duration <= 0:
                    return kfs[i].value
                u = (t - kfs[i].time) / duration
                return kfs[i].value + (kfs[i + 1].value - kfs[i].value) * u

        return kfs[-1].value

    def add_keyframe(self, time: float, value: float):
        """Add a keyframe (0-1 blend weight)."""
        self.keyframes.append(Keyframe(time, max(0, min(1, value))))
        self.keyframes.sort(key=lambda k: k.time)


# =============================================================================
# Pose State
# =============================================================================

@dataclass
class RootMotion:
    """Root/center-of-mass transform."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion

    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': list(self.position),
            'rotation': list(self.rotation)
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'RootMotion':
        return RootMotion(
            position=tuple(data.get('position', [0, 0, 0])),
            rotation=tuple(data.get('rotation', [0, 0, 0, 1]))
        )


@dataclass
class PoseState:
    """
    Complete body pose state.

    Like Unity's HumanPose - abstracted from any specific skeleton.
    Contains normalized muscle values + root motion + blend shapes.
    """
    muscles: Dict[str, float] = field(default_factory=dict)
    blendshapes: Dict[str, float] = field(default_factory=dict)
    root: RootMotion = field(default_factory=RootMotion)

    def get_muscle(self, name: str, default: float = 0.0) -> float:
        """Get a muscle value."""
        return self.muscles.get(name, default)

    def set_muscle(self, name: str, value: float):
        """Set a muscle value (clamped to [-1, 1])."""
        self.muscles[name] = max(-1.0, min(1.0, value))

    def get_blendshape(self, name: str, default: float = 0.0) -> float:
        """Get a blend shape weight."""
        return self.blendshapes.get(name, default)

    def set_blendshape(self, name: str, value: float):
        """Set a blend shape weight (clamped to [0, 1])."""
        self.blendshapes[name] = max(0.0, min(1.0, value))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'muscles': dict(self.muscles),
            'blendshapes': dict(self.blendshapes),
            'root': self.root.to_dict()
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PoseState':
        return PoseState(
            muscles=dict(data.get('muscles', {})),
            blendshapes=dict(data.get('blendshapes', {})),
            root=RootMotion.from_dict(data.get('root', {}))
        )

    @staticmethod
    def neutral() -> 'PoseState':
        """Create neutral T-pose state."""
        return PoseState(
            muscles={m: 0.0 for m in HUMANOID_MUSCLES},
            blendshapes={},
            root=RootMotion()
        )

    def to_muscle_array(self, muscle_order: Optional[List[str]] = None) -> List[float]:
        """Convert muscles to array in specified order."""
        order = muscle_order or HUMANOID_MUSCLES
        return [self.muscles.get(m, 0.0) for m in order]

    @staticmethod
    def from_muscle_array(values: List[float], muscle_order: Optional[List[str]] = None) -> 'PoseState':
        """Create from muscle array."""
        order = muscle_order or HUMANOID_MUSCLES
        muscles = {m: v for m, v in zip(order, values)}
        return PoseState(muscles=muscles)


# =============================================================================
# Pose Track
# =============================================================================

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
class PoseTrack:
    """
    Complete pose animation track.

    Rig-agnostic body animation using muscle space.
    Can be applied to any humanoid avatar via retargeting.
    """
    name: str = "Untitled Pose"
    duration: float = 0.0
    fps: int = 30
    author: str = ""
    created: str = ""
    tags: List[str] = field(default_factory=list)

    # Muscle channels
    muscles: Dict[str, MuscleChannel] = field(default_factory=dict)

    # Blend shape channels
    blendshapes: Dict[str, BlendShapeChannel] = field(default_factory=dict)

    # Root motion (center of mass)
    root_position: List[Keyframe] = field(default_factory=list)
    root_rotation: List[Keyframe] = field(default_factory=list)  # Quaternion components

    # Sync points
    markers: List[Marker] = field(default_factory=list)

    # Avatar hint (what body type this was authored for)
    archetype: str = "humanoid"  # humanoid, quadruped, custom

    def sample(self, t: float) -> PoseState:
        """Sample all channels at time t."""
        # Sample muscles
        muscle_values = {}
        for name, channel in self.muscles.items():
            muscle_values[name] = channel.sample(t)

        # Sample blend shapes
        blendshape_values = {}
        for name, channel in self.blendshapes.items():
            blendshape_values[name] = channel.sample(t)

        # Sample root (simplified - just position for now)
        root = RootMotion()
        if self.root_position:
            # Sample each component
            # For now, store as list of keyframes per component
            pass

        return PoseState(
            muscles=muscle_values,
            blendshapes=blendshape_values,
            root=root
        )

    def add_muscle_keyframe(self, muscle: str, time: float, value: float,
                            in_tangent: Tuple[float, float] = (0.0, 0.0),
                            out_tangent: Tuple[float, float] = (0.0, 0.0)):
        """Add a keyframe to a muscle channel."""
        if muscle not in self.muscles:
            self.muscles[muscle] = MuscleChannel(name=muscle)
        self.muscles[muscle].add_keyframe(time, value, in_tangent, out_tangent)
        self._update_duration()

    def add_blendshape_keyframe(self, blendshape: str, time: float, value: float):
        """Add a keyframe to a blend shape channel."""
        if blendshape not in self.blendshapes:
            self.blendshapes[blendshape] = BlendShapeChannel(name=blendshape)
        self.blendshapes[blendshape].add_keyframe(time, value)
        self._update_duration()

    def _update_duration(self):
        """Update duration based on keyframes."""
        max_time = 0.0
        for channel in self.muscles.values():
            for kf in channel.keyframes:
                max_time = max(max_time, kf.time)
        for channel in self.blendshapes.values():
            for kf in channel.keyframes:
                max_time = max(max_time, kf.time)
        self.duration = max_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML export."""
        return {
            'format': 'pose-track',
            'version': '1.0',
            'metadata': {
                'name': self.name,
                'duration': self.duration,
                'fps': self.fps,
                'author': self.author,
                'created': self.created,
                'tags': self.tags,
                'archetype': self.archetype
            },
            'muscles': {name: ch.to_dict() for name, ch in self.muscles.items()},
            'blendshapes': {name: ch.to_dict() for name, ch in self.blendshapes.items()},
            'markers': [m.to_dict() for m in self.markers]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PoseTrack':
        """Deserialize from dictionary."""
        metadata = data.get('metadata', {})

        muscles = {}
        for name, ch_data in data.get('muscles', {}).items():
            muscles[name] = MuscleChannel.from_dict(name, ch_data)

        blendshapes = {}
        for name, ch_data in data.get('blendshapes', {}).items():
            blendshapes[name] = BlendShapeChannel.from_dict(name, ch_data)

        return PoseTrack(
            name=metadata.get('name', 'Untitled'),
            duration=float(metadata.get('duration', 0.0)),
            fps=int(metadata.get('fps', 30)),
            author=metadata.get('author', ''),
            created=metadata.get('created', ''),
            tags=metadata.get('tags', []),
            archetype=metadata.get('archetype', 'humanoid'),
            muscles=muscles,
            blendshapes=blendshapes,
            markers=[Marker.from_dict(m) for m in data.get('markers', [])]
        )

    def save_yaml(self, filepath: str):
        """Save track to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed")
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def load_yaml(filepath: str) -> 'PoseTrack':
        """Load track from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed")
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return PoseTrack.from_dict(data)


# =============================================================================
# Pose Track Player
# =============================================================================

class PoseTrackPlayer:
    """Plays a pose track with timing control."""

    def __init__(self, track: PoseTrack):
        self.track = track
        self.current_time: float = 0.0
        self.speed: float = 1.0
        self.is_playing: bool = False
        self.is_looping: bool = False

        self.on_complete: TrackCompletionBehavior = TrackCompletionBehavior.HOLD

        # Callbacks
        self.marker_callbacks: Dict[str, List[Callable]] = {}
        self.completion_callback: Optional[Callable] = None

        # State tracking
        self._last_update_time: float = 0.0
        self._triggered_markers: set = set()

    def play(self):
        self.is_playing = True
        self._last_update_time = time.time()

    def pause(self):
        self.is_playing = False

    def stop(self):
        self.is_playing = False
        self.current_time = 0.0
        self._triggered_markers.clear()

    def seek(self, t: float):
        self.current_time = max(0.0, min(t, self.track.duration))
        self._triggered_markers = {m for m in self._triggered_markers if m > self.current_time}

    def update(self) -> PoseState:
        """Update playback and return current pose state."""
        if self.is_playing:
            now = time.time()
            delta = (now - self._last_update_time) * self.speed
            self._last_update_time = now

            old_time = self.current_time
            self.current_time += delta

            self._check_markers(old_time, self.current_time)

            if self.current_time >= self.track.duration:
                if self.is_looping:
                    self.current_time = self.current_time % self.track.duration
                    self._triggered_markers.clear()
                else:
                    self.current_time = self.track.duration
                    self.is_playing = False
                    if self.completion_callback:
                        self.completion_callback(self)

        return self.track.sample(self.current_time)

    def sample(self, t: Optional[float] = None) -> PoseState:
        return self.track.sample(t if t is not None else self.current_time)

    def on_marker(self, marker_name: str, callback: Callable):
        if marker_name not in self.marker_callbacks:
            self.marker_callbacks[marker_name] = []
        self.marker_callbacks[marker_name].append(callback)

    def _check_markers(self, old_t: float, new_t: float):
        for marker in self.track.markers:
            if old_t < marker.time <= new_t and marker.time not in self._triggered_markers:
                self._triggered_markers.add(marker.time)
                if marker.name in self.marker_callbacks:
                    for cb in self.marker_callbacks[marker.name]:
                        cb()


# =============================================================================
# Pose Retargeter
# =============================================================================

class PoseRetargeter:
    """
    Converts muscle values to bone rotations for a specific avatar.

    Like Unity's RetargetTo phase - takes rig-agnostic muscle clip
    and applies it to a specific skeleton with proper bone limits.
    """

    # Default mapping from muscle bone-part names to VRM humanoid bone names.
    # Our muscle naming uses Unity-style PascalCase (LeftArm, RightForeArm),
    # while VRM humanoid bones use camelCase (leftUpperArm, rightLowerArm).
    DEFAULT_VRM_BONE_MAP: Dict[str, str] = {
        'Hips': 'hips',
        'Spine': 'spine',
        'Chest': 'chest',
        'UpperChest': 'upperChest',
        'Neck': 'neck',
        'Head': 'head',
        'LeftEye': 'leftEye',
        'RightEye': 'rightEye',
        'Jaw': 'jaw',
        'LeftShoulder': 'leftShoulder',
        'LeftArm': 'leftUpperArm',
        'LeftForeArm': 'leftLowerArm',
        'LeftHand': 'leftHand',
        'RightShoulder': 'rightShoulder',
        'RightArm': 'rightUpperArm',
        'RightForeArm': 'rightLowerArm',
        'RightHand': 'rightHand',
        'LeftUpperLeg': 'leftUpperLeg',
        'LeftLowerLeg': 'leftLowerLeg',
        'LeftFoot': 'leftFoot',
        'LeftToes': 'leftToes',
        'RightUpperLeg': 'rightUpperLeg',
        'RightLowerLeg': 'rightLowerLeg',
        'RightFoot': 'rightFoot',
        'RightToes': 'rightToes',
    }

    # Left-side bone parts that need Z and Y axis sign negation.
    # In T-pose, left limbs extend in -X while right limbs extend in +X.
    # A Z rotation that brings the right arm down (+X -> down) would bring
    # the left arm UP (-X -> up). Negating Z and Y for left-side bones
    # gives symmetric muscle behavior: DownUp=-1 means "down" for both sides.
    LEFT_MIRROR_BONES: set = {
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
        'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToes',
        'LeftEye',
    }

    def __init__(self, avatar_config: Optional[Dict[str, Any]] = None):
        """
        Initialize with avatar muscle configuration.

        Args:
            avatar_config: Dict mapping bone names to muscle definitions.
                           If None, uses default humanoid definitions.
        """
        self.config = avatar_config or MUSCLE_DEFINITIONS

        # Bone name mapping (avatar-specific bone names)
        # Pre-populated with VRM humanoid standard names
        self.bone_map: Dict[str, str] = dict(self.DEFAULT_VRM_BONE_MAP)

    def set_bone_map(self, bone_map: Dict[str, str]):
        """
        Set mapping from standard muscle bones to avatar bone names.

        Args:
            bone_map: Dict mapping e.g. "Head" to "head_bone" in avatar
        """
        self.bone_map = bone_map

    def muscle_to_rotation(self, muscle_name: str, value: float) -> Optional[Tuple[str, float, float, float]]:
        """
        Convert a muscle value to bone rotation.

        Args:
            muscle_name: Standard muscle name (e.g., "Head.NodDownUp")
            value: Normalized value [-1, 1]

        Returns:
            Tuple of (bone_name, euler_x, euler_y, euler_z) or None if unknown muscle
        """
        if muscle_name not in self.config:
            return None

        defn = self.config[muscle_name]
        axis = defn['axis']
        min_deg = defn['min']
        max_deg = defn['max']

        # Map [-1, 1] to [min, max] degrees
        if value >= 0:
            degrees = value * max_deg
        else:
            degrees = -value * min_deg

        # Get bone name
        bone_part = muscle_name.split('.')[0]  # "Head" from "Head.NodDownUp"
        bone_name = self.bone_map.get(bone_part, bone_part.lower())

        # Mirror correction for left-side bones: negate Y and Z axes
        # so that symmetric muscle values produce symmetric visual results.
        if bone_part in self.LEFT_MIRROR_BONES and axis in ('Y', 'Z'):
            degrees = -degrees

        # Create euler rotation based on axis
        euler = [0.0, 0.0, 0.0]
        axis_idx = {'X': 0, 'Y': 1, 'Z': 2}.get(axis, 0)
        euler[axis_idx] = degrees

        return (bone_name, euler[0], euler[1], euler[2])

    def apply_pose(self, pose: PoseState) -> Dict[str, Tuple[float, float, float]]:
        """
        Convert complete pose state to bone rotations.

        Args:
            pose: PoseState with muscle values

        Returns:
            Dict mapping bone names to euler rotations (x, y, z in degrees)
        """
        bone_rotations: Dict[str, List[float]] = {}

        for muscle_name, value in pose.muscles.items():
            result = self.muscle_to_rotation(muscle_name, value)
            if result:
                bone_name, rx, ry, rz = result

                # Accumulate rotations for same bone (multiple muscles per bone)
                if bone_name not in bone_rotations:
                    bone_rotations[bone_name] = [0.0, 0.0, 0.0]

                bone_rotations[bone_name][0] += rx
                bone_rotations[bone_name][1] += ry
                bone_rotations[bone_name][2] += rz

        return {name: tuple(rot) for name, rot in bone_rotations.items()}


# =============================================================================
# Pose Track Facet
# =============================================================================

class PoseTrackFacet:
    """
    Facet that plays pose tracks in the cognitive assembly.

    Provides authored body animation that can be applied to avatars.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.track: Optional[PoseTrack] = None
        self.player: Optional[PoseTrackPlayer] = None
        self.retargeter: PoseRetargeter = PoseRetargeter()

        # Load track if specified
        track_path = config.get('track')
        if track_path and os.path.exists(track_path):
            self.track = PoseTrack.load_yaml(track_path)
            self.player = PoseTrackPlayer(self.track)
            self.player.speed = config.get('speed', 1.0)
            self.player.is_looping = config.get('loop', False)

        # State
        self.is_active = False

        # Execution stats
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process and return current pose."""
        start_time = time.time()

        # Get pose
        if self.player and self.player.is_playing:
            pose = self.player.update()
        elif self.player:
            pose = self.player.sample()
        else:
            pose = PoseState.neutral()

        # Apply retargeting
        bone_rotations = self.retargeter.apply_pose(pose)

        elapsed = time.time() - start_time
        self.execution_count += 1
        self.total_execution_time += elapsed
        self.last_execution_time = elapsed

        return {
            'pose': pose.to_dict(),
            'bone_rotations': bone_rotations,
            'is_playing': self.player.is_playing if self.player else False,
            'current_time': self.player.current_time if self.player else 0.0,
            'duration': self.track.duration if self.track else 0.0
        }

    def start_playback(self, from_time: float = 0.0):
        if self.player:
            self.player.seek(from_time)
            self.player.play()
            self.is_active = True

    def stop_playback(self):
        if self.player:
            self.player.stop()
            self.is_active = False

    def get_execution_stats(self) -> Dict[str, Any]:
        return {
            'execution_count': self.execution_count,
            'total_tokens': 0,
            'avg_tokens': 0,
            'total_time': self.total_execution_time,
            'avg_time': self.total_execution_time / max(1, self.execution_count),
            'last_tokens': 0,
            'last_time': self.last_execution_time
        }

    def get_token_usage(self) -> Dict[str, Any]:
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.execution_count,
            'avg_tokens': 0
        }


# =============================================================================
# Example
# =============================================================================

def create_example_wave() -> PoseTrack:
    """Create example wave animation."""
    track = PoseTrack(
        name="Friendly Wave",
        author="NoodleStudio",
        tags=["greeting", "gesture", "friendly"]
    )

    # Raise right arm
    track.add_muscle_keyframe('RightArm.DownUp', 0.0, 0.0)
    track.add_muscle_keyframe('RightArm.DownUp', 0.3, 0.8)
    track.add_muscle_keyframe('RightArm.DownUp', 2.0, 0.8)
    track.add_muscle_keyframe('RightArm.DownUp', 2.5, 0.0)

    # Bend elbow
    track.add_muscle_keyframe('RightForeArm.Stretch', 0.0, 0.0)
    track.add_muscle_keyframe('RightForeArm.Stretch', 0.3, 0.3)
    track.add_muscle_keyframe('RightForeArm.Stretch', 2.0, 0.3)
    track.add_muscle_keyframe('RightForeArm.Stretch', 2.5, 0.0)

    # Wave motion (twist forearm)
    for i in range(4):
        base_time = 0.5 + i * 0.4
        track.add_muscle_keyframe('RightForeArm.TwistInOut', base_time, -0.3)
        track.add_muscle_keyframe('RightForeArm.TwistInOut', base_time + 0.2, 0.3)

    # Smile blend shape
    track.add_blendshape_keyframe('happy', 0.0, 0.0)
    track.add_blendshape_keyframe('happy', 0.3, 0.7)
    track.add_blendshape_keyframe('happy', 2.0, 0.7)
    track.add_blendshape_keyframe('happy', 2.5, 0.0)

    # Markers
    track.markers = [
        Marker(time=0.3, name="arm_raised"),
        Marker(time=2.0, name="wave_done")
    ]

    return track


if __name__ == "__main__":
    print("Creating example pose track...")
    track = create_example_wave()

    test_path = "/tmp/test_pose_track.posetrack"
    track.save_yaml(test_path)
    print(f"Saved to {test_path}")

    loaded = PoseTrack.load_yaml(test_path)
    print(f"Loaded: {loaded.name}, duration: {loaded.duration:.2f}s")

    print("\n=== Sampling test ===")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        pose = loaded.sample(t)
        arm = pose.get_muscle('RightArm.DownUp')
        twist = pose.get_muscle('RightForeArm.TwistInOut')
        happy = pose.get_blendshape('happy')
        print(f"t={t:.1f}s: arm={arm:.2f}, twist={twist:.2f}, happy={happy:.2f}")

    print("\n=== Retargeting test ===")
    retargeter = PoseRetargeter()
    retargeter.set_bone_map({'RightArm': 'arm_r', 'RightForeArm': 'forearm_r'})

    pose = loaded.sample(1.0)
    rotations = retargeter.apply_pose(pose)
    for bone, rot in rotations.items():
        print(f"  {bone}: ({rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f}) degrees")

    print("\nPose Track system working!")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
