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
#   FBX Animation Importer - Import Mixamo and other FBX animations
#
#   Converts FBX bone animations to Noodle muscle space for r...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.fbx_importer
# PURPOSE:  Fbx Importer
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AnimationKeyframe, AnimationCurve, AnimationClip, FBXAnimation, FBXParser
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import struct
import zlib
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
from dataclasses import dataclass, field
from pathlib import Path
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# =============================================================================
# Bone Name Mappings (Mixamo -> Standard Muscles)
# =============================================================================

# Mixamo uses specific naming conventions
MIXAMO_BONE_MAP = {
    # Spine
    'Hips': 'Hips',
    'Spine': 'Spine',
    'Spine1': 'Chest',
    'Spine2': 'UpperChest',

    # Head
    'Neck': 'Neck',
    'Head': 'Head',
    'HeadTop_End': None,

    # Left Arm
    'LeftShoulder': 'LeftShoulder',
    'LeftArm': 'LeftArm',
    'LeftForeArm': 'LeftForeArm',
    'LeftHand': 'LeftHand',

    # Right Arm
    'RightShoulder': 'RightShoulder',
    'RightArm': 'RightArm',
    'RightForeArm': 'RightForeArm',
    'RightHand': 'RightHand',

    # Left Leg
    'LeftUpLeg': 'LeftUpperLeg',
    'LeftLeg': 'LeftLowerLeg',
    'LeftFoot': 'LeftFoot',
    'LeftToeBase': 'LeftToes',

    # Right Leg
    'RightUpLeg': 'RightUpperLeg',
    'RightLeg': 'RightLowerLeg',
    'RightFoot': 'RightFoot',
    'RightToeBase': 'RightToes',

    # Fingers (optional)
    'LeftHandThumb1': 'LeftThumb.1',
    'LeftHandThumb2': 'LeftThumb.2',
    'LeftHandThumb3': 'LeftThumb.3',
    'LeftHandIndex1': 'LeftIndex.1',
    'LeftHandIndex2': 'LeftIndex.2',
    'LeftHandIndex3': 'LeftIndex.3',
    # ... etc
}

# Unity Humanoid bone names
UNITY_BONE_MAP = {
    'Hips': 'Hips',
    'Spine': 'Spine',
    'Chest': 'Chest',
    'UpperChest': 'UpperChest',
    'Neck': 'Neck',
    'Head': 'Head',
    'LeftShoulder': 'LeftShoulder',
    'LeftUpperArm': 'LeftArm',
    'LeftLowerArm': 'LeftForeArm',
    'LeftHand': 'LeftHand',
    'RightShoulder': 'RightShoulder',
    'RightUpperArm': 'RightArm',
    'RightLowerArm': 'RightForeArm',
    'RightHand': 'RightHand',
    'LeftUpperLeg': 'LeftUpperLeg',
    'LeftLowerLeg': 'LeftLowerLeg',
    'LeftFoot': 'LeftFoot',
    'LeftToes': 'LeftToes',
    'RightUpperLeg': 'RightUpperLeg',
    'RightLowerLeg': 'RightLowerLeg',
    'RightFoot': 'RightFoot',
    'RightToes': 'RightToes',
}

# Muscle axis mappings (which rotation axis maps to which muscle)
MUSCLE_AXIS_MAP = {
    # Format: standard_bone -> {axis: muscle_suffix}
    'Spine': {'X': '.FrontBack', 'Y': '.TwistLeftRight', 'Z': '.LeftRight'},
    'Chest': {'X': '.FrontBack', 'Y': '.TwistLeftRight', 'Z': '.LeftRight'},
    'UpperChest': {'X': '.FrontBack', 'Y': '.TwistLeftRight', 'Z': '.LeftRight'},
    'Neck': {'X': '.NodDownUp', 'Y': '.TurnLeftRight', 'Z': '.TiltLeftRight'},
    'Head': {'X': '.NodDownUp', 'Y': '.TurnLeftRight', 'Z': '.TiltLeftRight'},

    'LeftShoulder': {'X': '.FrontBack', 'Z': '.DownUp'},
    'LeftArm': {'X': '.FrontBack', 'Y': '.TwistInOut', 'Z': '.DownUp'},
    'LeftForeArm': {'X': '.Stretch', 'Y': '.TwistInOut'},
    'LeftHand': {'X': '.DownUp', 'Z': '.InOut'},

    'RightShoulder': {'X': '.FrontBack', 'Z': '.DownUp'},
    'RightArm': {'X': '.FrontBack', 'Y': '.TwistInOut', 'Z': '.DownUp'},
    'RightForeArm': {'X': '.Stretch', 'Y': '.TwistInOut'},
    'RightHand': {'X': '.DownUp', 'Z': '.InOut'},

    'LeftUpperLeg': {'X': '.FrontBack', 'Y': '.TwistInOut', 'Z': '.InOut'},
    'LeftLowerLeg': {'X': '.Stretch', 'Y': '.TwistInOut'},
    'LeftFoot': {'X': '.UpDown', 'Y': '.TwistInOut'},
    'LeftToes': {'X': '.UpDown'},

    'RightUpperLeg': {'X': '.FrontBack', 'Y': '.TwistInOut', 'Z': '.InOut'},
    'RightLowerLeg': {'X': '.Stretch', 'Y': '.TwistInOut'},
    'RightFoot': {'X': '.UpDown', 'Y': '.TwistInOut'},
    'RightToes': {'X': '.UpDown'},
}

# Default rotation ranges (degrees) for normalization
DEFAULT_ROTATION_RANGES = {
    'Spine.FrontBack': (-40, 40),
    'Spine.TwistLeftRight': (-40, 40),
    'Spine.LeftRight': (-40, 40),
    'Chest.FrontBack': (-40, 40),
    'Chest.TwistLeftRight': (-40, 40),
    'Chest.LeftRight': (-40, 40),
    'Neck.NodDownUp': (-40, 40),
    'Neck.TurnLeftRight': (-40, 40),
    'Neck.TiltLeftRight': (-40, 40),
    'Head.NodDownUp': (-40, 60),
    'Head.TurnLeftRight': (-70, 70),
    'Head.TiltLeftRight': (-40, 40),

    'LeftShoulder.DownUp': (-15, 30),
    'LeftShoulder.FrontBack': (-15, 15),
    'LeftArm.DownUp': (-60, 100),
    'LeftArm.FrontBack': (-100, 60),
    'LeftArm.TwistInOut': (-90, 50),
    'LeftForeArm.Stretch': (0, 145),
    'LeftForeArm.TwistInOut': (-90, 90),
    'LeftHand.DownUp': (-80, 80),
    'LeftHand.InOut': (-40, 25),

    'RightShoulder.DownUp': (-15, 30),
    'RightShoulder.FrontBack': (-15, 15),
    'RightArm.DownUp': (-60, 100),
    'RightArm.FrontBack': (-100, 60),
    'RightArm.TwistInOut': (-90, 50),
    'RightForeArm.Stretch': (0, 145),
    'RightForeArm.TwistInOut': (-90, 90),
    'RightHand.DownUp': (-80, 80),
    'RightHand.InOut': (-40, 25),

    'LeftUpperLeg.FrontBack': (-90, 50),
    'LeftUpperLeg.InOut': (-60, 60),
    'LeftUpperLeg.TwistInOut': (-60, 60),
    'LeftLowerLeg.Stretch': (0, 145),
    'LeftLowerLeg.TwistInOut': (-45, 45),
    'LeftFoot.UpDown': (-50, 50),
    'LeftFoot.TwistInOut': (-30, 30),
    'LeftToes.UpDown': (-50, 50),

    'RightUpperLeg.FrontBack': (-90, 50),
    'RightUpperLeg.InOut': (-60, 60),
    'RightUpperLeg.TwistInOut': (-60, 60),
    'RightLowerLeg.Stretch': (0, 145),
    'RightLowerLeg.TwistInOut': (-45, 45),
    'RightFoot.UpDown': (-50, 50),
    'RightFoot.TwistInOut': (-30, 30),
    'RightToes.UpDown': (-50, 50),
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AnimationKeyframe:
    """Single keyframe with time and value."""
    time: float
    value: float


@dataclass
class AnimationCurve:
    """Animation curve for one property."""
    bone_name: str
    property_name: str  # 'rotation.x', 'rotation.y', 'rotation.z', 'position.x', etc.
    keyframes: List[AnimationKeyframe] = field(default_factory=list)


@dataclass
class AnimationClip:
    """Complete animation clip from FBX."""
    name: str
    duration: float
    fps: float
    curves: Dict[str, AnimationCurve] = field(default_factory=dict)  # bone.property -> curve

    # Root motion
    root_position_curves: Dict[str, AnimationCurve] = field(default_factory=dict)
    root_rotation_curves: Dict[str, AnimationCurve] = field(default_factory=dict)


@dataclass
class FBXAnimation:
    """Parsed FBX file with animations."""
    clips: List[AnimationClip] = field(default_factory=list)
    bone_hierarchy: Dict[str, str] = field(default_factory=dict)  # bone -> parent
    bone_names: List[str] = field(default_factory=list)


# =============================================================================
# FBX Binary Parser (minimal implementation)
# =============================================================================

class FBXParser:
    """
    Minimal FBX binary parser for animation extraction.

    FBX is a complex format. This parser focuses on extracting
    animation curves from Mixamo-style FBX files.
    """

    FBX_MAGIC = b'Kaydara FBX Binary  \x00'

    def __init__(self):
        self.version = 0
        self.nodes = []

    def parse(self, file_path: str) -> FBXAnimation:
        """Parse FBX file and extract animations."""
        with open(file_path, 'rb') as f:
            # Check magic
            magic = f.read(21)
            if not magic.startswith(b'Kaydara FBX Binary'):
                raise ValueError("Not a valid binary FBX file")

            # Skip to version
            f.seek(23)
            self.version = struct.unpack('<I', f.read(4))[0]

            # Parse nodes
            self.nodes = self._parse_nodes(f)

        return self._extract_animation()

    def _parse_nodes(self, f: BinaryIO, depth: int = 0) -> List[Dict]:
        """Parse FBX node tree."""
        nodes = []

        while True:
            # Node header
            if self.version >= 7500:
                end_offset = struct.unpack('<Q', f.read(8))[0]
                num_props = struct.unpack('<Q', f.read(8))[0]
                props_len = struct.unpack('<Q', f.read(8))[0]
            else:
                end_offset = struct.unpack('<I', f.read(4))[0]
                num_props = struct.unpack('<I', f.read(4))[0]
                props_len = struct.unpack('<I', f.read(4))[0]

            name_len = struct.unpack('<B', f.read(1))[0]

            if end_offset == 0:
                break  # Null node = end of list

            name = f.read(name_len).decode('utf-8', errors='ignore')

            # Parse properties
            props = []
            for _ in range(num_props):
                prop = self._parse_property(f)
                props.append(prop)

            # Parse children
            children = []
            if f.tell() < end_offset:
                children = self._parse_nodes(f, depth + 1)

            nodes.append({
                'name': name,
                'props': props,
                'children': children
            })

            f.seek(end_offset)

        return nodes

    def _parse_property(self, f: BinaryIO) -> Any:
        """Parse a single FBX property."""
        type_code = f.read(1).decode('ascii')

        if type_code == 'Y':  # Int16
            return struct.unpack('<h', f.read(2))[0]
        elif type_code == 'C':  # Bool
            return struct.unpack('<?', f.read(1))[0]
        elif type_code == 'I':  # Int32
            return struct.unpack('<i', f.read(4))[0]
        elif type_code == 'F':  # Float
            return struct.unpack('<f', f.read(4))[0]
        elif type_code == 'D':  # Double
            return struct.unpack('<d', f.read(8))[0]
        elif type_code == 'L':  # Int64
            return struct.unpack('<q', f.read(8))[0]
        elif type_code == 'R':  # Raw bytes
            length = struct.unpack('<I', f.read(4))[0]
            return f.read(length)
        elif type_code == 'S':  # String
            length = struct.unpack('<I', f.read(4))[0]
            return f.read(length).decode('utf-8', errors='ignore')
        elif type_code in 'fdilbc':  # Arrays
            return self._parse_array(f, type_code)
        else:
            return None

    def _parse_array(self, f: BinaryIO, type_code: str) -> List:
        """Parse array property."""
        array_len = struct.unpack('<I', f.read(4))[0]
        encoding = struct.unpack('<I', f.read(4))[0]
        comp_len = struct.unpack('<I', f.read(4))[0]

        if encoding == 1:  # Compressed
            data = zlib.decompress(f.read(comp_len))
        else:
            data = f.read(comp_len)

        # Parse array elements
        if type_code == 'd':
            return list(struct.unpack(f'<{array_len}d', data))
        elif type_code == 'f':
            return list(struct.unpack(f'<{array_len}f', data))
        elif type_code == 'l':
            return list(struct.unpack(f'<{array_len}q', data))
        elif type_code == 'i':
            return list(struct.unpack(f'<{array_len}i', data))
        else:
            return list(data)

    def _extract_animation(self) -> FBXAnimation:
        """Extract animation data from parsed nodes."""
        result = FBXAnimation()

        # Find animation stack
        for node in self.nodes:
            if node['name'] == 'Objects':
                self._process_objects(node['children'], result)
            elif node['name'] == 'Connections':
                self._process_connections(node['children'], result)

        return result

    def _process_objects(self, nodes: List[Dict], result: FBXAnimation):
        """Process Objects section."""
        for node in nodes:
            name = node['name']

            if name == 'Model':
                # Bone/node
                if len(node['props']) >= 2:
                    model_name = node['props'][1]
                    if isinstance(model_name, str):
                        # Extract bone name (remove "Model::" prefix)
                        if model_name.startswith('Model::'):
                            model_name = model_name[7:]
                        result.bone_names.append(model_name)

            elif name == 'AnimationCurve':
                # Animation curve data
                curve = AnimationCurve(bone_name='', property_name='')

                for child in node['children']:
                    if child['name'] == 'KeyTime':
                        if child['props']:
                            times = child['props'][0]
                            if isinstance(times, list):
                                # FBX stores time in 1/46186158000 seconds
                                curve.keyframes = [
                                    AnimationKeyframe(t / 46186158000.0, 0.0)
                                    for t in times
                                ]

                    elif child['name'] == 'KeyValueFloat':
                        if child['props']:
                            values = child['props'][0]
                            if isinstance(values, list) and curve.keyframes:
                                for i, v in enumerate(values):
                                    if i < len(curve.keyframes):
                                        curve.keyframes[i].value = v

                if curve.keyframes:
                    # Store with node ID as key for later connection
                    if node['props']:
                        node_id = node['props'][0]
                        result.clips.append(AnimationClip(
                            name=str(node_id),
                            duration=max(kf.time for kf in curve.keyframes) if curve.keyframes else 0,
                            fps=30.0,
                            curves={str(node_id): curve}
                        ))

    def _process_connections(self, nodes: List[Dict], result: FBXAnimation):
        """Process Connections section to link curves to bones."""
        # This is simplified - full implementation would build connection graph
        pass


# =============================================================================
# RetargetFrom - Convert Bone Rotations to Muscle Values
# =============================================================================

class AnimationRetargeter:
    """
    Convert bone animations to muscle space.

    This is the "RetargetFrom" phase like Unity's Mecanim -
    taking source bone rotations and converting to normalized
    muscle values that can be applied to any humanoid.
    """

    def __init__(self, source_type: str = 'mixamo'):
        """
        Args:
            source_type: 'mixamo', 'unity', or 'auto'
        """
        self.source_type = source_type

        # Select bone mapping
        if source_type == 'mixamo':
            self.bone_map = MIXAMO_BONE_MAP
        elif source_type == 'unity':
            self.bone_map = UNITY_BONE_MAP
        else:
            # Auto-detect (use both)
            self.bone_map = {**MIXAMO_BONE_MAP, **UNITY_BONE_MAP}

    def retarget_clip(self, clip: AnimationClip) -> 'PoseTrack':
        """
        Convert animation clip to pose track with muscle values.

        Args:
            clip: Source animation clip with bone rotations

        Returns:
            PoseTrack with normalized muscle curves
        """
        from noodlestudio.core.pose_track import PoseTrack, MuscleChannel, Keyframe

        track = PoseTrack(
            name=clip.name,
            duration=clip.duration,
            fps=int(clip.fps)
        )

        # Process each bone curve
        for curve_key, curve in clip.curves.items():
            bone_name = curve.bone_name or self._extract_bone_from_key(curve_key)

            # Map to standard bone name
            std_bone = self.bone_map.get(bone_name, bone_name)
            if not std_bone:
                continue

            # Get property (rotation axis)
            prop = curve.property_name or self._extract_prop_from_key(curve_key)
            axis = self._prop_to_axis(prop)
            if not axis:
                continue

            # Get muscle name
            axis_map = MUSCLE_AXIS_MAP.get(std_bone, {})
            muscle_suffix = axis_map.get(axis)
            if not muscle_suffix:
                continue

            muscle_name = std_bone + muscle_suffix

            # Get rotation range for normalization
            rot_range = DEFAULT_ROTATION_RANGES.get(muscle_name, (-90, 90))

            # Convert keyframes to normalized muscle values
            muscle_channel = MuscleChannel(name=muscle_name)
            for kf in curve.keyframes:
                # Normalize rotation to [-1, 1]
                normalized = self._normalize_rotation(kf.value, rot_range)
                muscle_channel.keyframes.append(Keyframe(kf.time, normalized))

            track.muscles[muscle_name] = muscle_channel

        track._update_duration()
        return track

    def _extract_bone_from_key(self, key: str) -> str:
        """Extract bone name from curve key."""
        # Keys might be like "Hips|rotation.x" or just bone names
        if '|' in key:
            return key.split('|')[0]
        return key

    def _extract_prop_from_key(self, key: str) -> str:
        """Extract property from curve key."""
        if '|' in key:
            return key.split('|')[1]
        return ''

    def _prop_to_axis(self, prop: str) -> Optional[str]:
        """Convert property name to axis letter."""
        prop_lower = prop.lower()
        if 'x' in prop_lower:
            return 'X'
        elif 'y' in prop_lower:
            return 'Y'
        elif 'z' in prop_lower:
            return 'Z'
        return None

    def _normalize_rotation(self, degrees: float, rot_range: Tuple[float, float]) -> float:
        """
        Normalize rotation to [-1, 1] range.

        Args:
            degrees: Rotation in degrees
            rot_range: (min_degrees, max_degrees) tuple

        Returns:
            Normalized value [-1, 1]
        """
        min_deg, max_deg = rot_range

        # Handle asymmetric ranges
        if degrees >= 0:
            if max_deg != 0:
                return min(1.0, degrees / max_deg)
            return 0.0
        else:
            if min_deg != 0:
                return max(-1.0, degrees / abs(min_deg))
            return 0.0


# =============================================================================
# High-Level Import Function
# =============================================================================

def import_fbx_animation(file_path: str, source_type: str = 'auto') -> Optional['PoseTrack']:
    """
    Import FBX animation and convert to PoseTrack.

    Args:
        file_path: Path to FBX file
        source_type: 'mixamo', 'unity', or 'auto'

    Returns:
        PoseTrack with muscle curves, or None if failed
    """
    try:
        # Parse FBX
        parser = FBXParser()
        fbx_data = parser.parse(file_path)

        if not fbx_data.clips:
            print(f"[FBX Import] No animation clips found in {file_path}")
            return None

        # Use first clip (Mixamo usually has one)
        clip = fbx_data.clips[0]

        # Retarget to muscle space
        retargeter = AnimationRetargeter(source_type)
        pose_track = retargeter.retarget_clip(clip)

        # Set name from file
        pose_track.name = Path(file_path).stem

        print(f"[FBX Import] Imported {pose_track.name}: "
              f"{len(pose_track.muscles)} muscle channels, "
              f"{pose_track.duration:.2f}s")

        return pose_track

    except Exception as e:
        print(f"[FBX Import] Failed to import {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def import_fbx_with_affect_layer(
    fbx_path: str,
    affect_template: Optional[str] = None,
    source_type: str = 'auto',
    target_fps: float = 30.0,
    include_root_motion: bool = True
) -> Tuple[Optional['PoseTrack'], Optional['AffectTrack']]:
    """
    Import FBX and create paired affect track for annotation.

    Args:
        fbx_path: Path to FBX file
        affect_template: Optional template affect track to copy structure from
        source_type: 'mixamo', 'unity', or 'auto'
        target_fps: Target framerate for resampling (default 30)
        include_root_motion: Whether to include root motion curves (default True)

    Returns:
        Tuple of (PoseTrack, AffectTrack) - affect track is empty for annotation
    """
    from noodlestudio.core.affect_track import AffectTrack

    # Import pose
    pose_track = import_fbx_animation(fbx_path, source_type)
    if not pose_track:
        return None, None

    # Update FPS if needed
    if target_fps != pose_track.fps:
        pose_track.fps = int(target_fps)

    # Create matching affect track for PAD+BS annotation
    affect_track = AffectTrack(
        name=pose_track.name + "_affect",
        duration=pose_track.duration,
        fps=pose_track.fps
    )

    # Add start/end keyframes for each affect channel
    for channel_name in ['valence', 'arousal', 'dominance', 'boredom', 'sorrow']:
        affect_track.add_keyframe(channel_name, 0.0, 0.0)
        affect_track.add_keyframe(channel_name, pose_track.duration, 0.0)

    return pose_track, affect_track


# =============================================================================
# Assimp-based Import (if available)
# =============================================================================

def import_fbx_via_assimp(file_path: str) -> Optional['PoseTrack']:
    """
    Import FBX using the assimp library (more robust).

    Requires: pip install pyassimp

    Args:
        file_path: Path to FBX file

    Returns:
        PoseTrack or None
    """
    try:
        import pyassimp
    except ImportError:
        print("[FBX Import] pyassimp not available, using built-in parser")
        return import_fbx_animation(file_path)

    try:
        from noodlestudio.core.pose_track import PoseTrack, MuscleChannel, Keyframe

        scene = pyassimp.load(file_path)

        if not scene.animations:
            print(f"[FBX Import] No animations found in {file_path}")
            return None

        # Get first animation
        anim = scene.animations[0]

        track = PoseTrack(
            name=Path(file_path).stem,
            duration=anim.duration / anim.tickspersecond if anim.tickspersecond else anim.duration,
            fps=int(anim.tickspersecond) if anim.tickspersecond else 30
        )

        retargeter = AnimationRetargeter('auto')

        # Process each channel (bone)
        for channel in anim.channels:
            bone_name = channel.nodename.data.decode('utf-8')
            std_bone = retargeter.bone_map.get(bone_name, bone_name)

            if not std_bone:
                continue

            # Process rotation keys
            if channel.rotationkeys:
                for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
                    axis_map = MUSCLE_AXIS_MAP.get(std_bone, {})
                    muscle_suffix = axis_map.get(axis)
                    if not muscle_suffix:
                        continue

                    muscle_name = std_bone + muscle_suffix
                    rot_range = DEFAULT_ROTATION_RANGES.get(muscle_name, (-90, 90))

                    muscle_channel = MuscleChannel(name=muscle_name)

                    for key in channel.rotationkeys:
                        time = key.time / anim.tickspersecond if anim.tickspersecond else key.time

                        # Convert quaternion to euler (simplified)
                        quat = key.value
                        euler = _quat_to_euler(quat)
                        degrees = math.degrees(euler[axis_idx])

                        normalized = retargeter._normalize_rotation(degrees, rot_range)
                        muscle_channel.keyframes.append(Keyframe(time, normalized))

                    if muscle_channel.keyframes:
                        track.muscles[muscle_name] = muscle_channel

        pyassimp.release(scene)

        track._update_duration()
        return track

    except Exception as e:
        print(f"[FBX Import] Assimp import failed: {e}")
        return None


def _quat_to_euler(quat) -> Tuple[float, float, float]:
    """Convert quaternion to euler angles (XYZ order)."""
    x, y, z, w = quat.x, quat.y, quat.z, quat.w

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fbx_importer.py <path_to_fbx>")
        sys.exit(1)

    fbx_path = sys.argv[1]

    # Try assimp first, fall back to built-in
    track = import_fbx_via_assimp(fbx_path)
    if not track:
        track = import_fbx_animation(fbx_path)

    if track:
        print(f"\nImported: {track.name}")
        print(f"Duration: {track.duration:.2f}s")
        print(f"Muscles: {len(track.muscles)}")
        for name in sorted(track.muscles.keys())[:10]:
            channel = track.muscles[name]
            print(f"  {name}: {len(channel.keyframes)} keyframes")

        # Save as .posetrack
        output_path = fbx_path.replace('.fbx', '.posetrack').replace('.FBX', '.posetrack')
        track.save_yaml(output_path)
        print(f"\nSaved to: {output_path}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
