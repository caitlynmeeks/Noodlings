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
#   Gaussian Adapter - Bridge between Noodlings Scene Protocol and Gaussian Splatting
#
#   This module connects the semantic world (NSP ScenePackets...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.semantic_world.gaussian_adapter
# PURPOSE:  Gaussian Adapter
# LAYER:    Studio / Semantic World
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SemanticGaussian, SplatHitInfo, SkeletonBinding, GaussianAsset, GaussianInstance
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import json
import struct
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from .scene_packet import (
    ScenePacket,
    Noodling,
    Player,
    Prim,
    Zone,
    Vector3,
    Transform,
    VisualForm,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Per-Splat Semantic Metadata (LangSplat-style)
# =============================================================================

@dataclass
class SemanticGaussian:
    """
    A single Gaussian splat with full semantic metadata.

    Beyond standard Gaussian params (position, scale, rotation, color),
    each splat knows what it represents, which bones influence it,
    and can be queried with natural language (LangSplat-style).

    This enables:
    - Click-to-inspect: "You clicked on Red's left knee at (1.2, 0.4, 0.3)"
    - Semantic queries: "Where is the character's head?"
    - Skeletal animation: LBS deformation using bone weights
    """
    # Index in the Gaussian array
    index: int

    # Standard Gaussian params
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion
    opacity: float = 1.0

    # SKELETAL BINDING (for animation)
    bone_indices: Tuple[int, int, int, int] = (-1, -1, -1, -1)  # Up to 4 bones
    bone_weights: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    primary_bone: str = ""  # Human-readable: "leftLowerLeg"
    primary_bone_index: int = -1

    # ENTITY ASSOCIATION
    entity_type: str = ""  # "noodling", "prim", "environment"
    entity_id: str = ""  # "red_fire_anklebiter"

    # BODY CONTEXT
    semantic_label: str = ""  # "left_knee", "torso", "hair_strand_42"
    body_region: str = ""  # "lower_body", "head", "left_arm", "accessory"
    distance_from_bone: float = 0.0  # Distance from primary bone center

    # CLIP EMBEDDING (for LangSplat queries)
    clip_embedding: Optional[np.ndarray] = None  # 512-D or 768-D vector


@dataclass
class SplatHitInfo:
    """
    Result of clicking on / raycasting to a Gaussian splat.

    Provides full semantic context about what was clicked.
    """
    # Hit info
    hit: bool = False
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gaussian_index: int = -1
    distance: float = float('inf')

    # Entity info
    entity_type: str = ""  # "noodling", "prim"
    entity_id: str = ""
    display_name: str = ""

    # Body context
    body_part: str = ""  # "left_knee"
    body_region: str = ""  # "lower_body"
    bone: str = ""  # "leftLowerLeg"
    bone_index: int = -1

    # Semantic
    semantic_label: str = ""
    semantic_query: str = ""  # "left knee of fire imp character named Red"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/Inspector."""
        return {
            'hit': self.hit,
            'position': list(self.position),
            'gaussian_index': self.gaussian_index,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'display_name': self.display_name,
            'body_part': self.body_part,
            'body_region': self.body_region,
            'bone': self.bone,
            'bone_index': self.bone_index,
            'semantic_query': self.semantic_query,
        }


@dataclass
class SkeletonBinding:
    """
    Skeleton binding data for a Gaussian asset.

    Maps each Gaussian to its influencing bones, enabling
    LBS (Linear Blend Skinning) deformation for animation.

    Stored alongside the .radiance file as .skeleton
    """
    # Source VRM/skeleton info
    source_vrm: str = ""
    bone_count: int = 0
    bone_names: List[str] = field(default_factory=list)
    humanoid_map: Dict[str, int] = field(default_factory=dict)  # humanoid_name -> bone_index

    # Per-Gaussian binding (parallel arrays)
    gaussian_count: int = 0
    bone_indices: Optional[np.ndarray] = None  # (N, 4) int - 4 bones per Gaussian
    bone_weights: Optional[np.ndarray] = None  # (N, 4) float - weights per bone
    primary_bones: List[str] = field(default_factory=list)  # (N,) primary bone name
    semantic_labels: List[str] = field(default_factory=list)  # (N,) body part label
    body_regions: List[str] = field(default_factory=list)  # (N,) region

    def get_gaussian_binding(self, index: int) -> SemanticGaussian:
        """Get semantic data for a single Gaussian."""
        if index < 0 or index >= self.gaussian_count:
            return SemanticGaussian(index=index)

        bone_idx = tuple(self.bone_indices[index]) if self.bone_indices is not None else (-1, -1, -1, -1)
        bone_wgt = tuple(self.bone_weights[index]) if self.bone_weights is not None else (0.0, 0.0, 0.0, 0.0)

        return SemanticGaussian(
            index=index,
            bone_indices=bone_idx,
            bone_weights=bone_wgt,
            primary_bone=self.primary_bones[index] if index < len(self.primary_bones) else "",
            primary_bone_index=bone_idx[0] if bone_idx[0] >= 0 else -1,
            semantic_label=self.semantic_labels[index] if index < len(self.semantic_labels) else "",
            body_region=self.body_regions[index] if index < len(self.body_regions) else "",
        )

    def save_json(self, path: str):
        """Save skeleton binding to JSON."""
        data = {
            'source_vrm': self.source_vrm,
            'bone_count': self.bone_count,
            'bone_names': self.bone_names,
            'humanoid_map': self.humanoid_map,
            'gaussian_count': self.gaussian_count,
            'bone_indices': self.bone_indices.tolist() if self.bone_indices is not None else [],
            'bone_weights': self.bone_weights.tolist() if self.bone_weights is not None else [],
            'primary_bones': self.primary_bones,
            'semantic_labels': self.semantic_labels,
            'body_regions': self.body_regions,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved skeleton binding: {path}")

    @classmethod
    def load_json(cls, path: str) -> 'SkeletonBinding':
        """Load skeleton binding from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        binding = cls(
            source_vrm=data.get('source_vrm', ''),
            bone_count=data.get('bone_count', 0),
            bone_names=data.get('bone_names', []),
            humanoid_map=data.get('humanoid_map', {}),
            gaussian_count=data.get('gaussian_count', 0),
            primary_bones=data.get('primary_bones', []),
            semantic_labels=data.get('semantic_labels', []),
            body_regions=data.get('body_regions', []),
        )

        if data.get('bone_indices'):
            binding.bone_indices = np.array(data['bone_indices'], dtype=np.int32)
        if data.get('bone_weights'):
            binding.bone_weights = np.array(data['bone_weights'], dtype=np.float32)

        return binding


# Body region classification based on bone names
BONE_TO_REGION = {
    'hips': 'torso', 'spine': 'torso', 'chest': 'torso', 'upperChest': 'torso',
    'neck': 'head', 'head': 'head', 'leftEye': 'head', 'rightEye': 'head', 'jaw': 'head',
    'leftShoulder': 'left_arm', 'leftUpperArm': 'left_arm', 'leftLowerArm': 'left_arm', 'leftHand': 'left_arm',
    'rightShoulder': 'right_arm', 'rightUpperArm': 'right_arm', 'rightLowerArm': 'right_arm', 'rightHand': 'right_arm',
    'leftUpperLeg': 'left_leg', 'leftLowerLeg': 'left_leg', 'leftFoot': 'left_leg', 'leftToes': 'left_leg',
    'rightUpperLeg': 'right_leg', 'rightLowerLeg': 'right_leg', 'rightFoot': 'right_leg', 'rightToes': 'right_leg',
}

# Finger bones
for side in ['left', 'right']:
    for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Little']:
        for segment in ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']:
            bone = f'{side}{finger}{segment}'
            BONE_TO_REGION[bone] = f'{side}_hand'


def get_body_region(bone_name: str) -> str:
    """Get body region from bone name."""
    return BONE_TO_REGION.get(bone_name, 'other')


# =============================================================================
# Gaussian Asset Types
# =============================================================================

@dataclass
class GaussianAsset:
    """
    A 3D Gaussian splat asset with semantic metadata.

    Each asset represents a single entity (character, prop, environment piece).
    The semantic_tags enable future LangSplat-style queries.
    """
    id: str
    name: str
    asset_type: str  # character, prop, environment, fx

    # File paths
    ply_path: str = ""
    thumbnail_path: str = ""
    source_images: List[str] = field(default_factory=list)

    # Generation info
    generator: str = "unknown"  # sharp, opensplat, dreamgaussian
    generated_at: str = ""
    generation_params: Dict[str, Any] = field(default_factory=dict)

    # Spatial info from PLY
    gaussian_count: int = 0
    bounds_min: Vector3 = field(default_factory=Vector3)
    bounds_max: Vector3 = field(default_factory=Vector3)
    center: Vector3 = field(default_factory=Vector3)

    # Semantic metadata
    semantic_tags: List[str] = field(default_factory=list)
    description: str = ""

    # For characters: form/state associations
    noodling_id: Optional[str] = None
    visual_form: Optional[str] = None  # which VisualForm this represents

    # For animation (4D-GS support)
    is_animated: bool = False
    frame_count: int = 1
    fps: float = 30.0

    # Skeleton binding for animation (loaded separately)
    skeleton_binding_path: str = ""
    _skeleton_binding: Optional['SkeletonBinding'] = field(default=None, repr=False)

    @property
    def skeleton_binding(self) -> Optional['SkeletonBinding']:
        """Get skeleton binding, loading from file if needed."""
        if self._skeleton_binding is None and self.skeleton_binding_path:
            if os.path.exists(self.skeleton_binding_path):
                try:
                    self._skeleton_binding = SkeletonBinding.load_json(self.skeleton_binding_path)
                except Exception as e:
                    logger.warning(f"Failed to load skeleton binding: {e}")
        return self._skeleton_binding

    @skeleton_binding.setter
    def skeleton_binding(self, value: 'SkeletonBinding'):
        """Set skeleton binding."""
        self._skeleton_binding = value

    def get_hit_info(self, gaussian_index: int, position: Tuple[float, float, float]) -> 'SplatHitInfo':
        """Get semantic hit info for a Gaussian by index."""
        hit = SplatHitInfo(
            hit=True,
            position=position,
            gaussian_index=gaussian_index,
            entity_type="noodling" if self.noodling_id else "prim",
            entity_id=self.noodling_id or self.id,
            display_name=self.name,
        )

        # Add skeleton binding info if available
        if self.skeleton_binding:
            binding = self.skeleton_binding.get_gaussian_binding(gaussian_index)
            hit.body_part = binding.semantic_label
            hit.body_region = binding.body_region
            hit.bone = binding.primary_bone
            hit.bone_index = binding.primary_bone_index

            # Build semantic query string
            hit.semantic_query = f"{hit.body_part} of {self.name}"
            if hit.body_region:
                hit.semantic_query = f"{hit.body_part} ({hit.body_region}) of {self.name}"

        return hit

    @property
    def file_size_mb(self) -> float:
        """Get file size in MB."""
        if self.ply_path and os.path.exists(self.ply_path):
            return os.path.getsize(self.ply_path) / (1024 * 1024)
        return 0.0


@dataclass
class GaussianInstance:
    """
    An instance of a GaussianAsset placed in a scene.

    Like Unity's prefab instances - same asset, different transform.
    """
    instance_id: str
    asset_id: str

    # Scene placement
    transform: Transform = field(default_factory=Transform)
    zone_id: str = ""

    # Visibility/rendering
    visible: bool = True
    opacity: float = 1.0
    tint_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Entity link (which NSP entity this instance represents)
    entity_type: str = ""  # noodling, player, prim
    entity_id: str = ""


@dataclass
class GaussianScene:
    """
    A composed scene of Gaussian instances ready for rendering.

    This is what gets sent to the renderer - a collection of instances
    with their transforms, plus camera and lighting info.
    """
    scene_id: str
    stage_id: str
    stage_name: str

    # All instances in the scene
    instances: Dict[str, GaussianInstance] = field(default_factory=dict)

    # Assets referenced by instances
    assets: Dict[str, GaussianAsset] = field(default_factory=dict)

    # Environment
    environment_asset_id: Optional[str] = None
    skybox_path: Optional[str] = None
    ambient_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)

    # Camera (from NSP CameraDirective)
    camera_position: Vector3 = field(default_factory=Vector3)
    camera_target: Vector3 = field(default_factory=Vector3)
    camera_fov: float = 60.0

    # Lighting
    key_light_direction: Vector3 = field(default_factory=lambda: Vector3(-0.5, -1.0, -0.5))
    key_light_color: Tuple[float, float, float] = (1.0, 0.95, 0.9)
    key_light_intensity: float = 1.0


# =============================================================================
# Asset Manager
# =============================================================================

class GaussianAssetManager:
    """
    Manages the library of Gaussian assets.

    Provides:
    - Asset discovery from project folders
    - Asset generation tracking
    - Semantic search (future)
    """

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.assets: Dict[str, GaussianAsset] = {}
        self.index_path = self.project_path / "Library" / "gaussian_index.json"

        # Asset storage paths
        self.character_path = self.project_path / "Library" / "Gaussians" / "Characters"
        self.environment_path = self.project_path / "Library" / "Gaussians" / "Environments"
        self.prop_path = self.project_path / "Library" / "Gaussians" / "Props"

        self._ensure_directories()

    def _ensure_directories(self):
        """Create asset directories if they don't exist."""
        for path in [self.character_path, self.environment_path, self.prop_path]:
            path.mkdir(parents=True, exist_ok=True)

    def load_index(self):
        """Load asset index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    data = json.load(f)
                for asset_data in data.get("assets", []):
                    asset = self._dict_to_asset(asset_data)
                    self.assets[asset.id] = asset
                logger.info(f"Loaded {len(self.assets)} Gaussian assets from index")
            except Exception as e:
                logger.error(f"Failed to load Gaussian index: {e}")

    def save_index(self):
        """Save asset index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "assets": [self._asset_to_dict(a) for a in self.assets.values()]
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.assets)} Gaussian assets to index")

    def register_asset(self, asset: GaussianAsset) -> str:
        """Register a new Gaussian asset."""
        self.assets[asset.id] = asset
        self.save_index()
        return asset.id

    def get_asset(self, asset_id: str) -> Optional[GaussianAsset]:
        """Get asset by ID."""
        return self.assets.get(asset_id)

    def get_assets_for_noodling(self, noodling_id: str) -> List[GaussianAsset]:
        """Get all Gaussian assets for a noodling (may have multiple forms)."""
        return [a for a in self.assets.values()
                if a.noodling_id == noodling_id]

    def get_asset_for_form(self, noodling_id: str, form: str) -> Optional[GaussianAsset]:
        """Get Gaussian asset for a specific noodling form."""
        for asset in self.assets.values():
            if asset.noodling_id == noodling_id and asset.visual_form == form:
                return asset
        return None

    def search_by_tags(self, tags: List[str]) -> List[GaussianAsset]:
        """Search assets by semantic tags."""
        results = []
        for asset in self.assets.values():
            if any(tag in asset.semantic_tags for tag in tags):
                results.append(asset)
        return results

    def discover_ply_files(self):
        """Scan directories for PLY files and register new assets."""
        for category, path in [
            ("character", self.character_path),
            ("environment", self.environment_path),
            ("prop", self.prop_path)
        ]:
            if not path.exists():
                continue
            for ply_file in path.rglob("*.ply"):
                # Check if already registered
                existing = [a for a in self.assets.values()
                           if a.ply_path == str(ply_file)]
                if not existing:
                    # Create new asset entry
                    asset = GaussianAsset(
                        id=f"gs_{ply_file.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        name=ply_file.stem,
                        asset_type=category,
                        ply_path=str(ply_file),
                        generator="unknown"
                    )
                    # Read PLY metadata
                    self._read_ply_metadata(asset)
                    self.assets[asset.id] = asset
                    logger.info(f"Discovered Gaussian asset: {asset.name}")

        self.save_index()

    def _read_ply_metadata(self, asset: GaussianAsset):
        """Read basic metadata from PLY file header."""
        try:
            with open(asset.ply_path, 'rb') as f:
                # Read header
                header_lines = []
                while True:
                    line = f.readline().decode('ascii').strip()
                    header_lines.append(line)
                    if line == 'end_header':
                        break

                # Parse vertex count
                for line in header_lines:
                    if line.startswith('element vertex'):
                        asset.gaussian_count = int(line.split()[-1])
                        break

                logger.debug(f"PLY {asset.name}: {asset.gaussian_count} Gaussians")
        except Exception as e:
            logger.warning(f"Failed to read PLY metadata for {asset.name}: {e}")

    def _asset_to_dict(self, asset: GaussianAsset) -> Dict[str, Any]:
        """Convert asset to dictionary for serialization."""
        return {
            "id": asset.id,
            "name": asset.name,
            "asset_type": asset.asset_type,
            "ply_path": asset.ply_path,
            "thumbnail_path": asset.thumbnail_path,
            "source_images": asset.source_images,
            "generator": asset.generator,
            "generated_at": asset.generated_at,
            "generation_params": asset.generation_params,
            "gaussian_count": asset.gaussian_count,
            "bounds_min": [asset.bounds_min.x, asset.bounds_min.y, asset.bounds_min.z],
            "bounds_max": [asset.bounds_max.x, asset.bounds_max.y, asset.bounds_max.z],
            "center": [asset.center.x, asset.center.y, asset.center.z],
            "semantic_tags": asset.semantic_tags,
            "description": asset.description,
            "noodling_id": asset.noodling_id,
            "visual_form": asset.visual_form,
            "is_animated": asset.is_animated,
            "frame_count": asset.frame_count,
            "fps": asset.fps,
        }

    def _dict_to_asset(self, d: Dict[str, Any]) -> GaussianAsset:
        """Convert dictionary to GaussianAsset."""
        bounds_min = d.get("bounds_min", [0, 0, 0])
        bounds_max = d.get("bounds_max", [0, 0, 0])
        center = d.get("center", [0, 0, 0])

        return GaussianAsset(
            id=d.get("id", ""),
            name=d.get("name", ""),
            asset_type=d.get("asset_type", "unknown"),
            ply_path=d.get("ply_path", ""),
            thumbnail_path=d.get("thumbnail_path", ""),
            source_images=d.get("source_images", []),
            generator=d.get("generator", "unknown"),
            generated_at=d.get("generated_at", ""),
            generation_params=d.get("generation_params", {}),
            gaussian_count=d.get("gaussian_count", 0),
            bounds_min=Vector3(bounds_min[0], bounds_min[1], bounds_min[2]),
            bounds_max=Vector3(bounds_max[0], bounds_max[1], bounds_max[2]),
            center=Vector3(center[0], center[1], center[2]),
            semantic_tags=d.get("semantic_tags", []),
            description=d.get("description", ""),
            noodling_id=d.get("noodling_id"),
            visual_form=d.get("visual_form"),
            is_animated=d.get("is_animated", False),
            frame_count=d.get("frame_count", 1),
            fps=d.get("fps", 30.0),
        )


# =============================================================================
# Scene Compositor
# =============================================================================

class GaussianSceneCompositor:
    """
    Composes Gaussian scenes from NSP ScenePackets.

    Takes a semantic scene description (who's where, what's happening)
    and produces a renderable Gaussian scene (positioned PLY instances).
    """

    def __init__(self, asset_manager: GaussianAssetManager):
        self.asset_manager = asset_manager

    def compose_scene(self, packet: ScenePacket) -> GaussianScene:
        """
        Convert an NSP ScenePacket to a GaussianScene.

        This is the core bridge: semantic truth -> renderable Gaussians.
        """
        scene = GaussianScene(
            scene_id=f"scene_{packet.header.packet_id}",
            stage_id=packet.header.stage_id,
            stage_name=packet.header.stage_name,
        )

        # Add noodling instances
        for nid, noodling in packet.noodlings.items():
            instance = self._create_noodling_instance(noodling)
            if instance:
                scene.instances[instance.instance_id] = instance
                if instance.asset_id and instance.asset_id not in scene.assets:
                    asset = self.asset_manager.get_asset(instance.asset_id)
                    if asset:
                        scene.assets[instance.asset_id] = asset

        # Add prim instances
        for pid, prim in packet.prims.items():
            instance = self._create_prim_instance(prim)
            if instance:
                scene.instances[instance.instance_id] = instance
                if instance.asset_id and instance.asset_id not in scene.assets:
                    asset = self.asset_manager.get_asset(instance.asset_id)
                    if asset:
                        scene.assets[instance.asset_id] = asset

        # Set up camera from directive
        self._apply_camera_directive(scene, packet)

        # Set up lighting from ambient
        self._apply_ambient(scene, packet)

        return scene

    def _create_noodling_instance(self, noodling: Noodling) -> Optional[GaussianInstance]:
        """Create a Gaussian instance for a noodling."""
        # Try to find Gaussian asset for this noodling's current form
        asset = self.asset_manager.get_asset_for_form(
            noodling.id,
            noodling.visual_state
        )

        # Fall back to any asset for this noodling
        if not asset:
            assets = self.asset_manager.get_assets_for_noodling(noodling.id)
            if assets:
                asset = assets[0]

        if not asset:
            logger.warning(f"No Gaussian asset for noodling {noodling.display_name}")
            return None

        return GaussianInstance(
            instance_id=f"inst_{noodling.id}",
            asset_id=asset.id,
            transform=noodling.transform,
            zone_id=noodling.zone,
            entity_type="noodling",
            entity_id=noodling.id,
        )

    def _create_prim_instance(self, prim: Prim) -> Optional[GaussianInstance]:
        """Create a Gaussian instance for a prim."""
        # Search for asset by prim type in tags
        assets = self.asset_manager.search_by_tags([prim.prim_type])
        if not assets:
            logger.warning(f"No Gaussian asset for prim type {prim.prim_type}")
            return None

        asset = assets[0]

        return GaussianInstance(
            instance_id=f"inst_{prim.id}",
            asset_id=asset.id,
            transform=prim.transform,
            zone_id=prim.zone,
            entity_type="prim",
            entity_id=prim.id,
        )

    def _apply_camera_directive(self, scene: GaussianScene, packet: ScenePacket):
        """Apply camera settings from NSP to Gaussian scene."""
        cam = packet.camera_directive

        # Default camera position (will be refined based on mode)
        if cam.position:
            scene.camera_position = cam.position

        if cam.look_at:
            scene.camera_target = cam.look_at
        elif cam.subject and cam.subject in packet.noodlings:
            # Look at subject
            noodling = packet.noodlings[cam.subject]
            scene.camera_target = noodling.transform.position

            # Position camera based on framing
            distance = self._framing_to_distance(cam.framing.value)
            # Simple: behind and above the subject's forward direction
            fwd = noodling.transform.forward
            scene.camera_position = Vector3(
                noodling.transform.position.x - fwd.x * distance,
                noodling.transform.position.y + noodling.height * 0.8,
                noodling.transform.position.z - fwd.z * distance,
            )

        # FOV from style
        scene.camera_fov = cam.style.focal_length  # Approximate

    def _framing_to_distance(self, framing: str) -> float:
        """Convert framing type to camera distance."""
        distances = {
            "extreme_closeup": 0.5,
            "closeup": 1.0,
            "medium_closeup": 1.5,
            "medium": 2.5,
            "medium_wide": 4.0,
            "wide": 6.0,
            "extreme_wide": 12.0,
        }
        return distances.get(framing, 2.5)

    def _apply_ambient(self, scene: GaussianScene, packet: ScenePacket):
        """Apply ambient/lighting from NSP to Gaussian scene."""
        ambient = packet.spatial_truth.ambient

        # Time of day affects lighting
        time_colors = {
            "dawn": (1.0, 0.7, 0.5),
            "morning": (1.0, 0.95, 0.9),
            "midday": (1.0, 1.0, 1.0),
            "afternoon": (1.0, 0.95, 0.85),
            "dusk": (1.0, 0.6, 0.4),
            "night": (0.4, 0.45, 0.7),
        }
        scene.key_light_color = time_colors.get(
            ambient.time_of_day,
            (1.0, 0.95, 0.9)
        )

        # Weather affects intensity
        weather_intensity = {
            "clear": 1.0,
            "cloudy": 0.7,
            "fog": 0.5,
            "rain": 0.6,
            "storm": 0.3,
            "snow": 0.8,
        }
        scene.key_light_intensity = weather_intensity.get(
            ambient.weather,
            1.0
        )


# =============================================================================
# Generator Interface
# =============================================================================

class GaussianGenerator:
    """
    Interface for generating Gaussian assets.

    Wraps SHARP (character generation) and OpenSplat (environment training).
    """

    def __init__(self, asset_manager: GaussianAssetManager):
        self.asset_manager = asset_manager

        # Tool paths (auto-detect)
        self.sharp_path = self._find_sharp()
        self.opensplat_path = self._find_opensplat()

    def _find_sharp(self) -> Optional[str]:
        """Find SHARP executable/venv."""
        # Common locations
        candidates = [
            Path.home() / "git/noodlings_clean/external/ml-sharp/venv/bin/sharp",
            Path("/Users/thistlequell/git/noodlings_clean/external/ml-sharp/venv/bin/sharp"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _find_opensplat(self) -> Optional[str]:
        """Find OpenSplat executable."""
        candidates = [
            Path.home() / "git/noodlings_clean/external/OpenSplat/build/opensplat",
            Path("/Users/thistlequell/git/noodlings_clean/external/OpenSplat/build/opensplat"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def generate_character(
        self,
        noodling_id: str,
        visual_form: str,
        reference_image: str,
        tags: Optional[List[str]] = None,
        description: str = ""
    ) -> Optional[GaussianAsset]:
        """
        Generate Gaussian asset for a character using SHARP.

        Args:
            noodling_id: ID of the noodling this represents
            visual_form: Which visual form (e.g., "ghostly_fox", "human_form")
            reference_image: Path to reference image
            tags: Semantic tags for search
            description: Human-readable description

        Returns:
            Generated GaussianAsset, or None if generation failed
        """
        if not self.sharp_path:
            logger.error("SHARP not found - cannot generate character Gaussian")
            return None

        if not os.path.exists(reference_image):
            logger.error(f"Reference image not found: {reference_image}")
            return None

        # Output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.asset_manager.character_path / f"{noodling_id}_{visual_form}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run SHARP
        import subprocess
        cmd = [
            self.sharp_path,
            "predict",
            "-i", reference_image,
            "-o", str(output_dir)
        ]

        logger.info(f"Running SHARP: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"SHARP failed: {result.stderr}")
            return None

        # Find output PLY
        ply_files = list(output_dir.glob("*.ply"))
        if not ply_files:
            logger.error("SHARP produced no PLY output")
            return None

        ply_path = ply_files[0]

        # Create asset
        asset = GaussianAsset(
            id=f"gs_char_{noodling_id}_{visual_form}_{timestamp}",
            name=f"{noodling_id}_{visual_form}",
            asset_type="character",
            ply_path=str(ply_path),
            source_images=[reference_image],
            generator="sharp",
            generated_at=datetime.now().isoformat(),
            generation_params={"reference": reference_image},
            semantic_tags=tags or [],
            description=description,
            noodling_id=noodling_id,
            visual_form=visual_form,
        )

        # Read PLY metadata
        self.asset_manager._read_ply_metadata(asset)

        # Register
        self.asset_manager.register_asset(asset)

        logger.info(f"Generated character Gaussian: {asset.name} "
                   f"({asset.gaussian_count} Gaussians, {asset.file_size_mb:.1f}MB)")

        return asset

    def train_environment(
        self,
        name: str,
        images_path: str,
        iterations: int = 2000,
        tags: Optional[List[str]] = None,
        description: str = ""
    ) -> Optional[GaussianAsset]:
        """
        Train Gaussian asset for an environment using OpenSplat.

        Args:
            name: Environment name
            images_path: Path to COLMAP/SfM project with images
            iterations: Training iterations
            tags: Semantic tags
            description: Human-readable description

        Returns:
            Generated GaussianAsset, or None if training failed
        """
        if not self.opensplat_path:
            logger.error("OpenSplat not found - cannot train environment Gaussian")
            return None

        if not os.path.exists(images_path):
            logger.error(f"Images path not found: {images_path}")
            return None

        # Output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.asset_manager.environment_path / f"{name}_{timestamp}.ply"

        # Run OpenSplat
        import subprocess
        cmd = [
            self.opensplat_path,
            images_path,
            "-n", str(iterations),
            "-o", str(output_file)
        ]

        logger.info(f"Running OpenSplat: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            logger.error(f"OpenSplat failed: {result.stderr}")
            return None

        if not output_file.exists():
            logger.error("OpenSplat produced no output")
            return None

        # Create asset
        asset = GaussianAsset(
            id=f"gs_env_{name}_{timestamp}",
            name=name,
            asset_type="environment",
            ply_path=str(output_file),
            source_images=[images_path],
            generator="opensplat",
            generated_at=datetime.now().isoformat(),
            generation_params={"iterations": iterations, "source": images_path},
            semantic_tags=tags or [],
            description=description,
        )

        # Read PLY metadata
        self.asset_manager._read_ply_metadata(asset)

        # Register
        self.asset_manager.register_asset(asset)

        logger.info(f"Trained environment Gaussian: {asset.name} "
                   f"({asset.gaussian_count} Gaussians, {asset.file_size_mb:.1f}MB)")

        return asset


# =============================================================================
# Module-level Interface
# =============================================================================

_asset_manager: Optional[GaussianAssetManager] = None
_compositor: Optional[GaussianSceneCompositor] = None
_generator: Optional[GaussianGenerator] = None


def init_gaussian_adapter(project_path: str):
    """Initialize the Gaussian adapter for a project."""
    global _asset_manager, _compositor, _generator

    _asset_manager = GaussianAssetManager(project_path)
    _asset_manager.load_index()
    _asset_manager.discover_ply_files()

    _compositor = GaussianSceneCompositor(_asset_manager)
    _generator = GaussianGenerator(_asset_manager)

    logger.info(f"Gaussian adapter initialized with {len(_asset_manager.assets)} assets")


def get_asset_manager() -> Optional[GaussianAssetManager]:
    """Get the Gaussian asset manager."""
    return _asset_manager


def get_compositor() -> Optional[GaussianSceneCompositor]:
    """Get the scene compositor."""
    return _compositor


def get_generator() -> Optional[GaussianGenerator]:
    """Get the Gaussian generator."""
    return _generator


def compose_scene_from_packet(packet: ScenePacket) -> Optional[GaussianScene]:
    """Compose a Gaussian scene from an NSP ScenePacket."""
    if _compositor:
        return _compositor.compose_scene(packet)
    return None


def create_skeleton_binding_from_vrm(
    vrm_path: str,
    gaussian_positions: np.ndarray,
    noodling_id: str = "",
) -> SkeletonBinding:
    """
    Create skeleton binding by transferring VRM skinning weights to Gaussians.

    For each Gaussian, finds the nearest mesh vertices and transfers their
    bone weights. This enables LBS deformation of the Gaussian cloud.

    Args:
        vrm_path: Path to source VRM file
        gaussian_positions: (N, 3) array of Gaussian positions
        noodling_id: ID of the noodling for labeling

    Returns:
        SkeletonBinding with per-Gaussian bone weights
    """
    from .vrm_parser import parse_vrm, VRMAvatar

    # Parse VRM
    avatar = parse_vrm(vrm_path)

    # Create binding
    binding = SkeletonBinding(
        source_vrm=vrm_path,
        bone_count=len(avatar.skeleton.bones),
        bone_names=[b.name for b in avatar.skeleton.bones],
        humanoid_map=avatar.skeleton.humanoid_map,
        gaussian_count=len(gaussian_positions),
    )

    # Collect all mesh vertices and their skinning data
    all_vertices = []
    all_joint_indices = []
    all_joint_weights = []
    all_bone_names = []

    for mesh in avatar.meshes:
        if mesh.joint_indices is None or mesh.joint_weights is None:
            continue

        all_vertices.append(mesh.vertices)
        all_joint_indices.append(mesh.joint_indices)
        all_joint_weights.append(mesh.joint_weights)

    if not all_vertices:
        logger.warning("No skinned meshes found in VRM")
        return binding

    # Concatenate all vertex data
    vertices = np.vstack(all_vertices)
    joint_indices = np.vstack(all_joint_indices)
    joint_weights = np.vstack(all_joint_weights)

    # For each Gaussian, find nearest vertex and transfer weights
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)

    binding.bone_indices = np.zeros((len(gaussian_positions), 4), dtype=np.int32)
    binding.bone_weights = np.zeros((len(gaussian_positions), 4), dtype=np.float32)
    binding.primary_bones = []
    binding.semantic_labels = []
    binding.body_regions = []

    for i, pos in enumerate(gaussian_positions):
        # Find nearest vertex
        dist, idx = tree.query(pos, k=1)

        # Transfer skinning weights
        binding.bone_indices[i] = joint_indices[idx]
        binding.bone_weights[i] = joint_weights[idx]

        # Get primary bone (highest weight)
        primary_idx = int(joint_indices[idx][np.argmax(joint_weights[idx])])
        if 0 <= primary_idx < len(binding.bone_names):
            primary_bone = binding.bone_names[primary_idx]

            # Map to humanoid name if available
            for humanoid_name, bone_idx in avatar.skeleton.humanoid_map.items():
                if bone_idx == primary_idx:
                    primary_bone = humanoid_name
                    break

            binding.primary_bones.append(primary_bone)
            binding.body_regions.append(get_body_region(primary_bone))
            binding.semantic_labels.append(primary_bone.replace('left', 'left_').replace('right', 'right_'))
        else:
            binding.primary_bones.append("")
            binding.body_regions.append("other")
            binding.semantic_labels.append("")

    logger.info(f"Created skeleton binding: {binding.gaussian_count} Gaussians, "
               f"{binding.bone_count} bones from {vrm_path}")

    return binding


__all__ = [
    # Per-Splat Semantic Types
    "SemanticGaussian",
    "SplatHitInfo",
    "SkeletonBinding",
    "get_body_region",
    "BONE_TO_REGION",

    # Asset Types
    "GaussianAsset",
    "GaussianInstance",
    "GaussianScene",

    # Manager
    "GaussianAssetManager",
    "GaussianSceneCompositor",
    "GaussianGenerator",

    # Module interface
    "init_gaussian_adapter",
    "get_asset_manager",
    "get_compositor",
    "get_generator",
    "compose_scene_from_packet",

    # VRM Integration
    "create_skeleton_binding_from_vrm",
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
