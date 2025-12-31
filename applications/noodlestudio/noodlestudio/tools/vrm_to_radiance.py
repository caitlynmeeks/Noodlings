#!/usr/bin/env python3
"""
VRM to Radiance Converter - Convert VRM avatars to semantic Gaussian splats.

This is Phase 1 of the Gaussian world engine pipeline:
    VRM Avatar -> .radiance (Semantic Gaussians)

Each mesh vertex becomes a Gaussian splat that knows:
- Its position, scale, and orientation
- Which bones influence it (for animation)
- What body part it represents (for queries)
- What entity it belongs to

Usage:
    python -m noodlestudio.tools.vrm_to_radiance path/to/avatar.vrm [options]

Options:
    --output, -o    Output .radiance file path
    --name          Entity display name
    --entity-id     Entity ID for scene protocol
    --downsample    Vertex downsampling ratio (default: 1.0 = all vertices)
    --preview       Generate preview PLY alongside .radiance

Author: Caitlyn + Claude
Date: December 2025
"""

import argparse
import logging
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from noodlestudio.core.semantic_world.vrm_parser import parse_vrm, VRMAvatar
from noodlestudio.core.semantic_world.radiance_format import (
    RadianceAsset,
    RadianceSkeleton,
    RadianceBone,
    SpringChain,
    SpringCollider,
    BodyRegion,
)
from noodlestudio.core.semantic_world.gaussian_adapter import (
    get_body_region,
    BONE_TO_REGION,
)

logger = logging.getLogger(__name__)


def filter_trained_gaussians(
    asset: 'RadianceAsset',
    min_opacity: float = 0.8,
    max_scale: float = 0.05,
    max_sh_brightness: float = 2.0,
) -> 'RadianceAsset':
    """
    Filter Gaussians from OpenSplat training to remove background artifacts.

    OpenSplat training on white-background images creates semi-transparent
    Gaussians with extreme SH values to represent the background. When
    rendering on black background, these cause over-exposure.

    This filter keeps only:
    - High-opacity Gaussians (actual surface, not background)
    - Small-scale Gaussians (not huge background blobs)
    - Reasonable-brightness Gaussians (not saturated white)

    Args:
        asset: RadianceAsset to filter
        min_opacity: Minimum opacity threshold (default 0.8)
        max_scale: Maximum scale in any dimension (default 0.05)
        max_sh_brightness: Maximum SH DC value (default 2.0)

    Returns:
        Filtered RadianceAsset (new instance)
    """
    # Build filter mask
    opacity_mask = asset.opacities >= min_opacity
    scale_mask = asset.scales.max(axis=1) <= max_scale
    brightness_mask = asset.sh_dc.max(axis=1) <= max_sh_brightness

    combined_mask = opacity_mask & scale_mask & brightness_mask
    n_kept = combined_mask.sum()
    n_total = len(combined_mask)

    logger.info(f"  Filtering: keeping {n_kept:,} of {n_total:,} Gaussians ({100*n_kept/n_total:.1f}%)")
    logger.info(f"    Opacity >= {min_opacity}: {opacity_mask.sum():,} pass")
    logger.info(f"    Scale <= {max_scale}: {scale_mask.sum():,} pass")
    logger.info(f"    SH brightness <= {max_sh_brightness}: {brightness_mask.sum():,} pass")

    # Create filtered asset
    filtered = RadianceAsset()
    filtered.positions = asset.positions[combined_mask].astype(np.float32)
    filtered.scales = asset.scales[combined_mask].astype(np.float32)
    filtered.rotations = asset.rotations[combined_mask].astype(np.float32)
    filtered.opacities = asset.opacities[combined_mask].astype(np.float32)
    filtered.sh_dc = asset.sh_dc[combined_mask].astype(np.float32)

    # Copy optional arrays if present
    if asset.sh_rest is not None:
        filtered.sh_rest = asset.sh_rest[combined_mask]
    if asset.skin_bone_indices is not None:
        filtered.skin_bone_indices = asset.skin_bone_indices[combined_mask]
    if asset.skin_bone_weights is not None:
        filtered.skin_bone_weights = asset.skin_bone_weights[combined_mask]
    if asset.body_regions is not None:
        filtered.body_regions = asset.body_regions[combined_mask]
    if asset.clip_embeddings is not None:
        filtered.clip_embeddings = asset.clip_embeddings[combined_mask]

    # Copy non-per-gaussian data
    filtered.skeleton = asset.skeleton
    filtered.spring_chains = asset.spring_chains
    filtered.spring_colliders = asset.spring_colliders
    filtered.semantic_labels = [asset.semantic_labels[i] for i in range(n_total) if combined_mask[i]] if asset.semantic_labels else []
    filtered.metadata = asset.metadata

    return filtered


def compute_adaptive_scales(
    positions: np.ndarray,
    k_neighbors: int = 8,
    scale_factor: float = 0.5,
) -> np.ndarray:
    """
    Compute adaptive scales based on local point density.

    Gaussians in dense regions get smaller scales, sparse regions get larger.

    Args:
        positions: (N, 3) point positions
        k_neighbors: Number of neighbors to consider for density estimation
        scale_factor: Multiplier for computed scale

    Returns:
        (N, 3) scales where each Gaussian's scale is based on local spacing
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        # Fallback to uniform scale
        avg_scale = 0.005
        return np.full((len(positions), 3), avg_scale, dtype=np.float32)

    n = len(positions)
    if n < k_neighbors + 1:
        avg_scale = 0.005
        return np.full((n, 3), avg_scale, dtype=np.float32)

    tree = cKDTree(positions)

    # Query k nearest neighbors for each point
    distances, _ = tree.query(positions, k=k_neighbors + 1)

    # Use mean distance to neighbors as base scale
    # Skip first distance (self, distance=0)
    mean_distances = distances[:, 1:].mean(axis=1)

    # Scale should be roughly 1/2 the spacing to get good overlap
    scales = mean_distances * scale_factor

    # Clamp to reasonable range
    scales = np.clip(scales, 0.001, 0.05)

    # Return as (N, 3) with slightly flattened Z
    out_scales = np.zeros((n, 3), dtype=np.float32)
    out_scales[:, 0] = scales
    out_scales[:, 1] = scales
    out_scales[:, 2] = scales * 0.7  # Slightly flatter

    return out_scales


def densify_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
    normals: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
    joint_indices: Optional[np.ndarray] = None,
    joint_weights: Optional[np.ndarray] = None,
    add_face_centers: bool = True,
    add_edge_midpoints: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[Tuple[int, int]]]:
    """
    Densify mesh by adding face centers and edge midpoints.

    This increases Gaussian density from sparse vertices to dense coverage.

    Args:
        vertices: (N, 3) original vertex positions
        indices: (M,) triangle indices
        normals: (N, 3) vertex normals (optional)
        uvs: (N, 2) UV coordinates (optional)
        joint_indices: (N, 4) skinning bone indices (optional)
        joint_weights: (N, 4) skinning weights (optional)
        add_face_centers: Add one Gaussian per triangle face center
        add_edge_midpoints: Add one Gaussian per unique edge midpoint

    Returns:
        new_positions, new_normals, new_uvs, new_joint_indices, new_joint_weights,
        mesh_vertex_indices (list of (mesh_idx, vert_idx) for texture sampling)
    """
    n_verts = len(vertices)
    n_tris = len(indices) // 3

    # Start with original vertices
    new_positions = [vertices]
    new_normals = [normals] if normals is not None else []
    new_uvs = [uvs] if uvs is not None else []
    new_joint_indices = [joint_indices] if joint_indices is not None else []
    new_joint_weights = [joint_weights] if joint_weights is not None else []

    # Track source mesh/vertex for texture sampling
    # Original vertices: interpolate from themselves (use vertex 0 of each original vertex)
    mesh_vertex_indices = [(0, i) for i in range(n_verts)]

    # Reshape indices to triangles
    triangles = indices.reshape(-1, 3)

    if add_face_centers:
        # Add face center for each triangle
        face_centers = np.zeros((n_tris, 3), dtype=np.float32)
        face_normals = np.zeros((n_tris, 3), dtype=np.float32) if normals is not None else None
        face_uvs = np.zeros((n_tris, 2), dtype=np.float32) if uvs is not None else None
        face_joints = np.zeros((n_tris, 4), dtype=joint_indices.dtype) if joint_indices is not None else None
        face_weights = np.zeros((n_tris, 4), dtype=np.float32) if joint_weights is not None else None

        for i, tri in enumerate(triangles):
            v0, v1, v2 = tri
            # Position: centroid
            face_centers[i] = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0

            # Normal: average of vertex normals
            if normals is not None:
                face_normals[i] = (normals[v0] + normals[v1] + normals[v2]) / 3.0
                norm = np.linalg.norm(face_normals[i])
                if norm > 0.001:
                    face_normals[i] /= norm

            # UVs: centroid
            if uvs is not None:
                face_uvs[i] = (uvs[v0] + uvs[v1] + uvs[v2]) / 3.0

            # Skinning: weighted average (use weights from all 3 vertices)
            if joint_indices is not None and joint_weights is not None:
                # Combine all bone influences from the 3 vertices
                # Simple approach: use dominant vertex's skinning
                weights_sum = joint_weights[v0].sum() + joint_weights[v1].sum() + joint_weights[v2].sum()
                if weights_sum > 0:
                    # Weighted combination
                    combined_weights = (joint_weights[v0] + joint_weights[v1] + joint_weights[v2]) / 3.0
                    face_weights[i] = combined_weights
                    # Use bone indices from vertex with highest total weight
                    max_vert = max([v0, v1, v2], key=lambda v: joint_weights[v].sum())
                    face_joints[i] = joint_indices[max_vert]

            # Track source vertex for texture (use first vertex of triangle)
            mesh_vertex_indices.append((0, v0))

        new_positions.append(face_centers)
        if face_normals is not None:
            new_normals.append(face_normals)
        if face_uvs is not None:
            new_uvs.append(face_uvs)
        if face_joints is not None:
            new_joint_indices.append(face_joints)
        if face_weights is not None:
            new_joint_weights.append(face_weights)

    if add_edge_midpoints:
        # Track unique edges to avoid duplicates
        edge_set = set()
        edge_data = []

        for tri in triangles:
            v0, v1, v2 = tri
            # 3 edges per triangle
            edges = [(min(v0, v1), max(v0, v1)),
                    (min(v1, v2), max(v1, v2)),
                    (min(v2, v0), max(v2, v0))]

            for e0, e1 in edges:
                if (e0, e1) not in edge_set:
                    edge_set.add((e0, e1))
                    edge_data.append((e0, e1))

        n_edges = len(edge_data)
        edge_midpoints = np.zeros((n_edges, 3), dtype=np.float32)
        edge_normals = np.zeros((n_edges, 3), dtype=np.float32) if normals is not None else None
        edge_uvs = np.zeros((n_edges, 2), dtype=np.float32) if uvs is not None else None
        edge_joints = np.zeros((n_edges, 4), dtype=joint_indices.dtype) if joint_indices is not None else None
        edge_weights = np.zeros((n_edges, 4), dtype=np.float32) if joint_weights is not None else None

        for i, (e0, e1) in enumerate(edge_data):
            # Position: midpoint
            edge_midpoints[i] = (vertices[e0] + vertices[e1]) / 2.0

            # Normal: average
            if normals is not None:
                edge_normals[i] = (normals[e0] + normals[e1]) / 2.0
                norm = np.linalg.norm(edge_normals[i])
                if norm > 0.001:
                    edge_normals[i] /= norm

            # UVs: midpoint
            if uvs is not None:
                edge_uvs[i] = (uvs[e0] + uvs[e1]) / 2.0

            # Skinning: average
            if joint_indices is not None and joint_weights is not None:
                edge_weights[i] = (joint_weights[e0] + joint_weights[e1]) / 2.0
                # Use bone indices from vertex with higher weight
                if joint_weights[e0].sum() >= joint_weights[e1].sum():
                    edge_joints[i] = joint_indices[e0]
                else:
                    edge_joints[i] = joint_indices[e1]

            # Track source vertex for texture
            mesh_vertex_indices.append((0, e0))

        new_positions.append(edge_midpoints)
        if edge_normals is not None:
            new_normals.append(edge_normals)
        if edge_uvs is not None:
            new_uvs.append(edge_uvs)
        if edge_joints is not None:
            new_joint_indices.append(edge_joints)
        if edge_weights is not None:
            new_joint_weights.append(edge_weights)

    # Concatenate all
    out_positions = np.vstack(new_positions)
    out_normals = np.vstack(new_normals) if new_normals else None
    out_uvs = np.vstack(new_uvs) if new_uvs else None
    out_joint_indices = np.vstack(new_joint_indices) if new_joint_indices else None
    out_joint_weights = np.vstack(new_joint_weights) if new_joint_weights else None

    return out_positions, out_normals, out_uvs, out_joint_indices, out_joint_weights, mesh_vertex_indices


def vertices_to_gaussians(
    positions: np.ndarray,
    normals: Optional[np.ndarray] = None,
    base_scale: float = 0.005,
    use_adaptive_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert mesh vertices to Gaussian parameters.

    Each vertex becomes a Gaussian splat with:
    - Position from vertex position
    - Scale based on local density (adaptive) or fixed base_scale
    - Rotation aligned to surface normal
    - Opacity of 1.0
    - SH DC from vertex color (white if no color)

    Args:
        positions: (N, 3) vertex positions
        normals: (N, 3) vertex normals (optional)
        base_scale: Base scale for Gaussians (used if adaptive=False)
        use_adaptive_scale: Compute scale from local point density

    Returns:
        positions, scales, rotations, opacities, sh_dc
    """
    n = len(positions)

    # Positions stay the same
    out_positions = positions.astype(np.float32)

    # Scales - adaptive or fixed
    # scale_factor=0.55 provides ~75% overlap coverage at default scale_mult=1.0
    if use_adaptive_scale and n > 10:
        scales = compute_adaptive_scales(positions, k_neighbors=8, scale_factor=0.55)
    else:
        scales = np.full((n, 3), base_scale, dtype=np.float32)

    # Rotations - identity quaternion (x, y, z, w)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 3] = 1.0  # w = 1 for identity

    if normals is not None:
        # Align Gaussians to surface normals
        for i in range(n):
            normal = normals[i]
            if np.linalg.norm(normal) > 0.001:
                normal = normal / np.linalg.norm(normal)
                rotations[i] = normal_to_quaternion(normal)
                # Make slightly flatter along normal direction
                scales[i, 2] = scales[i, 2] * 0.5

    # Opacities - all fully opaque
    opacities = np.ones(n, dtype=np.float32)

    # SH DC - neutral gray (will be colored by textures in full pipeline)
    sh_dc = np.full((n, 3), 0.5, dtype=np.float32)

    return out_positions, scales, rotations, opacities, sh_dc


def normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
    """
    Create quaternion that rotates Z-axis to align with normal.

    Args:
        normal: Unit normal vector (3,)

    Returns:
        Quaternion (x, y, z, w)
    """
    # Z-axis reference
    z = np.array([0.0, 0.0, 1.0])

    # If already aligned, return identity
    dot = np.dot(z, normal)
    if dot > 0.999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.999:
        return np.array([1.0, 0.0, 0.0, 0.0])  # 180 rotation around X

    # Cross product gives rotation axis
    axis = np.cross(z, normal)
    axis = axis / np.linalg.norm(axis)

    # Angle from dot product
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    # Quaternion from axis-angle
    half_angle = angle / 2
    s = np.sin(half_angle)
    return np.array([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        np.cos(half_angle)
    ], dtype=np.float32)


def transfer_skinning(
    avatar: VRMAvatar,
    gaussian_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, list, list, list]:
    """
    Transfer VRM skinning weights to Gaussians by nearest-vertex lookup.

    Args:
        avatar: Parsed VRM avatar
        gaussian_positions: (N, 3) Gaussian positions

    Returns:
        bone_indices (N, 4), bone_weights (N, 4),
        primary_bones, semantic_labels, body_regions
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        logger.warning("scipy not available, using fallback nearest-neighbor")
        return fallback_skinning(avatar, gaussian_positions)

    # Collect all mesh vertices with skinning data
    all_vertices = []
    all_joint_indices = []
    all_joint_weights = []

    for mesh in avatar.meshes:
        if mesh.joint_indices is None or mesh.joint_weights is None:
            # Use first bone for unskinned meshes
            n = len(mesh.vertices)
            all_vertices.append(mesh.vertices)
            all_joint_indices.append(np.zeros((n, 4), dtype=np.int32))
            all_joint_weights.append(np.zeros((n, 4), dtype=np.float32))
            all_joint_weights[-1][:, 0] = 1.0  # 100% weight to bone 0
        else:
            all_vertices.append(mesh.vertices)
            all_joint_indices.append(mesh.joint_indices.astype(np.int32))
            all_joint_weights.append(mesh.joint_weights.astype(np.float32))

    if not all_vertices:
        logger.warning("No mesh vertices found in VRM")
        n = len(gaussian_positions)
        return (
            np.zeros((n, 4), dtype=np.uint16),
            np.zeros((n, 4), dtype=np.float32),
            ["" for _ in range(n)],
            ["" for _ in range(n)],
            ["other" for _ in range(n)],
        )

    # Concatenate all data
    vertices = np.vstack(all_vertices)
    joint_indices = np.vstack(all_joint_indices)
    joint_weights = np.vstack(all_joint_weights)

    # Build KD-tree for fast nearest-neighbor lookup
    tree = cKDTree(vertices)

    n = len(gaussian_positions)
    out_bone_indices = np.zeros((n, 4), dtype=np.uint16)
    out_bone_weights = np.zeros((n, 4), dtype=np.float32)
    primary_bones = []
    semantic_labels = []
    body_regions = []

    # Map bone indices to names
    bone_names = [b.name for b in avatar.skeleton.bones]
    humanoid_reverse = {v: k for k, v in avatar.skeleton.humanoid_map.items()}

    for i, pos in enumerate(gaussian_positions):
        # Find nearest vertex
        dist, idx = tree.query(pos, k=1)

        # Transfer weights
        out_bone_indices[i] = joint_indices[idx]
        out_bone_weights[i] = joint_weights[idx]

        # Normalize weights
        weight_sum = out_bone_weights[i].sum()
        if weight_sum > 0:
            out_bone_weights[i] /= weight_sum
        else:
            out_bone_weights[i, 0] = 1.0

        # Get primary bone (highest weight)
        primary_idx = int(out_bone_indices[i][np.argmax(out_bone_weights[i])])

        if 0 <= primary_idx < len(bone_names):
            bone_name = bone_names[primary_idx]

            # Try humanoid name first
            humanoid_name = humanoid_reverse.get(primary_idx, bone_name)

            primary_bones.append(humanoid_name)
            body_regions.append(get_body_region(humanoid_name))

            # Create readable semantic label
            label = humanoid_name.replace('left', 'left ').replace('right', 'right ')
            label = label.replace('Upper', ' upper ').replace('Lower', ' lower ')
            label = ' '.join(label.split()).strip()
            semantic_labels.append(label)
        else:
            primary_bones.append("")
            body_regions.append("other")
            semantic_labels.append("")

    return out_bone_indices, out_bone_weights, primary_bones, semantic_labels, body_regions


def fallback_skinning(
    avatar: VRMAvatar,
    gaussian_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, list, list, list]:
    """
    Fallback skinning transfer without scipy (slower, brute force).
    """
    # Collect all mesh vertices with skinning data
    all_vertices = []
    all_joint_indices = []
    all_joint_weights = []

    for mesh in avatar.meshes:
        if mesh.joint_indices is not None and mesh.joint_weights is not None:
            all_vertices.append(mesh.vertices)
            all_joint_indices.append(mesh.joint_indices.astype(np.int32))
            all_joint_weights.append(mesh.joint_weights.astype(np.float32))

    if not all_vertices:
        n = len(gaussian_positions)
        return (
            np.zeros((n, 4), dtype=np.uint16),
            np.zeros((n, 4), dtype=np.float32),
            ["" for _ in range(n)],
            ["" for _ in range(n)],
            ["other" for _ in range(n)],
        )

    vertices = np.vstack(all_vertices)
    joint_indices = np.vstack(all_joint_indices)
    joint_weights = np.vstack(all_joint_weights)

    n = len(gaussian_positions)
    out_bone_indices = np.zeros((n, 4), dtype=np.uint16)
    out_bone_weights = np.zeros((n, 4), dtype=np.float32)
    primary_bones = []
    semantic_labels = []
    body_regions = []

    bone_names = [b.name for b in avatar.skeleton.bones]
    humanoid_reverse = {v: k for k, v in avatar.skeleton.humanoid_map.items()}

    for i, pos in enumerate(gaussian_positions):
        # Brute force nearest neighbor
        dists = np.linalg.norm(vertices - pos, axis=1)
        idx = np.argmin(dists)

        out_bone_indices[i] = joint_indices[idx]
        out_bone_weights[i] = joint_weights[idx]

        weight_sum = out_bone_weights[i].sum()
        if weight_sum > 0:
            out_bone_weights[i] /= weight_sum
        else:
            out_bone_weights[i, 0] = 1.0

        primary_idx = int(out_bone_indices[i][np.argmax(out_bone_weights[i])])

        if 0 <= primary_idx < len(bone_names):
            bone_name = bone_names[primary_idx]
            humanoid_name = humanoid_reverse.get(primary_idx, bone_name)
            primary_bones.append(humanoid_name)
            body_regions.append(get_body_region(humanoid_name))
            label = humanoid_name.replace('left', 'left ').replace('right', 'right ')
            semantic_labels.append(label.strip())
        else:
            primary_bones.append("")
            body_regions.append("other")
            semantic_labels.append("")

    return out_bone_indices, out_bone_weights, primary_bones, semantic_labels, body_regions


def body_region_to_enum(region_name: str) -> int:
    """Convert body region name to BodyRegion enum value."""
    mapping = {
        "head": BodyRegion.HEAD,
        "torso": BodyRegion.TORSO,
        "left_arm": BodyRegion.LEFT_ARM,
        "right_arm": BodyRegion.RIGHT_ARM,
        "left_leg": BodyRegion.LEFT_LEG,
        "right_leg": BodyRegion.RIGHT_LEG,
        "left_hand": BodyRegion.LEFT_HAND,
        "right_hand": BodyRegion.RIGHT_HAND,
        "tail": BodyRegion.TAIL,
        "accessory": BodyRegion.ACCESSORY,
        "other": BodyRegion.OTHER,
    }
    return mapping.get(region_name, BodyRegion.OTHER)


def sample_texture_colors(
    avatar: 'VRMAvatar',
    mesh_vertex_indices: List[Tuple[int, int]],  # List of (mesh_idx, vertex_idx)
) -> np.ndarray:
    """
    Sample texture colors for vertices using UV coordinates.

    Args:
        avatar: Parsed VRM avatar with textures and materials
        mesh_vertex_indices: List of (mesh_index, vertex_index) tuples

    Returns:
        (N, 3) array of RGB colors in [0, 1] range
    """
    if not HAS_PIL:
        logger.warning("PIL not available, using default gray colors")
        return np.full((len(mesh_vertex_indices), 3), 0.5, dtype=np.float32)

    # Load textures into PIL images
    texture_images = {}
    for i, tex_data in enumerate(avatar.textures):
        if isinstance(tex_data, bytes):
            try:
                img = Image.open(io.BytesIO(tex_data)).convert('RGB')
                texture_images[i] = np.array(img, dtype=np.float32) / 255.0
            except Exception as e:
                logger.warning(f"Failed to load texture {i}: {e}")

    colors = np.full((len(mesh_vertex_indices), 3), 0.5, dtype=np.float32)

    for i, (mesh_idx, vert_idx) in enumerate(mesh_vertex_indices):
        mesh = avatar.meshes[mesh_idx]

        # Get material and texture
        mat_idx = mesh.material_index
        if mat_idx is None or mat_idx >= len(avatar.materials):
            continue

        material = avatar.materials[mat_idx]
        tex_idx = material.diffuse_texture
        if tex_idx is None or tex_idx not in texture_images:
            # Use material's diffuse color as fallback
            if hasattr(material, 'diffuse_color') and material.diffuse_color:
                colors[i] = np.array(material.diffuse_color[:3], dtype=np.float32)
            continue

        # Get UV coordinates
        if mesh.uvs is None or vert_idx >= len(mesh.uvs):
            continue

        uv = mesh.uvs[vert_idx]
        tex_img = texture_images[tex_idx]
        h, w = tex_img.shape[:2]

        # Sample texture (with wrapping)
        u = uv[0] % 1.0
        v = uv[1] % 1.0

        # UV to pixel (V is often flipped in textures)
        px = int(u * (w - 1))
        py = int((1.0 - v) * (h - 1))  # Flip V

        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))

        colors[i] = tex_img[py, px]

    return colors


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to spherical harmonics DC component.

    The SH DC coefficient is related to RGB by:
        rgb = sh_dc * SH_C0 + 0.5

    So:
        sh_dc = (rgb - 0.5) / SH_C0

    Args:
        rgb: (N, 3) RGB colors in [0, 1]

    Returns:
        (N, 3) SH DC coefficients
    """
    SH_C0 = 0.28209479177387814
    return (rgb - 0.5) / SH_C0


def vrm_to_radiance(
    vrm_path: str,
    output_path: Optional[str] = None,
    entity_id: str = "",
    display_name: str = "",
    downsample: float = 1.0,
    base_scale: float = 0.005,
    densify: bool = True,
    add_face_centers: bool = True,
    add_edge_midpoints: bool = True,
    use_adaptive_scale: bool = True,
) -> RadianceAsset:
    """
    Convert a VRM avatar to a .radiance file.

    Each mesh vertex becomes a semantic Gaussian that knows:
    - Its spatial properties (position, scale, rotation)
    - Its skeletal binding (which bones move it)
    - Its semantic label (what body part it represents)

    Args:
        vrm_path: Path to input VRM file
        output_path: Path for output .radiance file (optional)
        entity_id: Entity ID for scene protocol
        display_name: Human-readable name
        downsample: Fraction of vertices to keep (1.0 = all)
        base_scale: Base scale for Gaussians (used if adaptive=False)
        densify: Add face centers and edge midpoints for denser coverage
        add_face_centers: Add Gaussians at triangle centroids
        add_edge_midpoints: Add Gaussians at edge midpoints
        use_adaptive_scale: Compute scale from local point density

    Returns:
        Created RadianceAsset
    """
    vrm_path = Path(vrm_path)
    if not vrm_path.exists():
        raise FileNotFoundError(f"VRM file not found: {vrm_path}")

    logger.info(f"Converting VRM to Radiance: {vrm_path}")

    # Parse VRM
    avatar = parse_vrm(str(vrm_path))
    logger.info(f"  Parsed: {avatar.vertex_count} vertices, {avatar.bone_count} bones, "
               f"{len(avatar.blend_shapes)} blend shapes, {avatar.spring_chain_count} spring chains")

    # Collect and densify mesh data
    all_positions = []
    all_normals = []
    all_uvs = []
    all_joint_indices = []
    all_joint_weights = []
    mesh_vertex_indices = []  # Track (mesh_idx, vert_idx) for texture sampling

    for mesh_idx, mesh in enumerate(avatar.meshes):
        # Get mesh data
        vertices = mesh.vertices
        mesh_normals = mesh.normals if mesh.normals is not None else np.zeros_like(vertices)
        mesh_uvs = mesh.uvs
        mesh_joint_indices = mesh.joint_indices
        mesh_joint_weights = mesh.joint_weights

        if densify and mesh.indices is not None and len(mesh.indices) >= 3:
            # Densify this mesh
            dense_pos, dense_norm, dense_uvs, dense_joints, dense_weights, dense_indices = densify_mesh(
                vertices=vertices,
                indices=mesh.indices,
                normals=mesh_normals,
                uvs=mesh_uvs,
                joint_indices=mesh_joint_indices,
                joint_weights=mesh_joint_weights,
                add_face_centers=add_face_centers,
                add_edge_midpoints=add_edge_midpoints,
            )

            all_positions.append(dense_pos)
            all_normals.append(dense_norm if dense_norm is not None else np.zeros_like(dense_pos))

            if dense_uvs is not None:
                all_uvs.append(dense_uvs)

            if dense_joints is not None:
                all_joint_indices.append(dense_joints.astype(np.int32))
            if dense_weights is not None:
                all_joint_weights.append(dense_weights.astype(np.float32))

            # Update mesh_vertex_indices with correct mesh_idx
            for orig_mesh_idx, vert_idx in dense_indices:
                mesh_vertex_indices.append((mesh_idx, vert_idx))

            logger.info(f"  Mesh {mesh_idx} ({mesh.name}): {len(vertices)} -> {len(dense_pos)} points")
        else:
            # No densification - use original vertices
            n_verts = len(vertices)
            all_positions.append(vertices)
            all_normals.append(mesh_normals)

            if mesh_uvs is not None:
                all_uvs.append(mesh_uvs)

            if mesh_joint_indices is not None:
                all_joint_indices.append(mesh_joint_indices.astype(np.int32))
            if mesh_joint_weights is not None:
                all_joint_weights.append(mesh_joint_weights.astype(np.float32))

            for vert_idx in range(n_verts):
                mesh_vertex_indices.append((mesh_idx, vert_idx))

    positions = np.vstack(all_positions)
    normals = np.vstack(all_normals)

    logger.info(f"  Total after densification: {len(positions)} points "
               f"({len(positions) / avatar.vertex_count:.1f}x original)")

    # Downsample if requested
    if downsample < 1.0:
        n_keep = max(1, int(len(positions) * downsample))
        indices = np.random.choice(len(positions), n_keep, replace=False)
        indices.sort()
        positions = positions[indices]
        normals = normals[indices]
        mesh_vertex_indices = [mesh_vertex_indices[i] for i in indices]
        logger.info(f"  Downsampled to {len(positions)} vertices ({downsample*100:.0f}%)")

    # Convert vertices to Gaussians with adaptive scale
    g_positions, g_scales, g_rotations, g_opacities, g_sh_dc = vertices_to_gaussians(
        positions, normals, base_scale, use_adaptive_scale=use_adaptive_scale
    )
    logger.info(f"  Created {len(g_positions)} Gaussians (adaptive_scale={use_adaptive_scale})")

    # Sample texture colors
    logger.info(f"  Sampling texture colors...")
    rgb_colors = sample_texture_colors(avatar, mesh_vertex_indices)
    g_sh_dc = rgb_to_sh_dc(rgb_colors)
    logger.info(f"  Applied {len(avatar.textures)} textures")

    # Create RadianceAsset
    asset = RadianceAsset()
    asset.positions = g_positions
    asset.scales = g_scales
    asset.rotations = g_rotations
    asset.opacities = g_opacities
    asset.sh_dc = g_sh_dc

    # Transfer skeleton
    asset.skeleton = RadianceSkeleton()
    for bone in avatar.skeleton.bones:
        asset.skeleton.bones.append(RadianceBone(
            name=bone.name,
            parent_index=bone.parent_index,
            position=(bone.transform.position.x, bone.transform.position.y, bone.transform.position.z),
            rotation=(bone.transform.rotation.x, bone.transform.rotation.y,
                     bone.transform.rotation.z, bone.transform.rotation.w),
            scale=(bone.transform.scale.x, bone.transform.scale.y, bone.transform.scale.z),
        ))
    asset.skeleton.humanoid_map = dict(avatar.skeleton.humanoid_map)
    logger.info(f"  Transferred skeleton: {len(asset.skeleton.bones)} bones")

    # Transfer spring bones
    for chain in avatar.spring_bones.chains:
        asset.spring_chains.append(SpringChain(
            name=chain.name,
            bone_indices=list(chain.bone_indices),
            stiffness=chain.stiffness,
            gravity_power=chain.gravity_power,
            gravity_dir=(chain.gravity_dir.x, chain.gravity_dir.y, chain.gravity_dir.z),
            drag_force=chain.drag_force,
            hit_radius=chain.hit_radius,
        ))

    for collider in avatar.spring_bones.colliders:
        asset.spring_colliders.append(SpringCollider(
            bone_index=collider.bone_index,
            offset=(collider.offset.x, collider.offset.y, collider.offset.z),
            radius=collider.radius,
        ))
    logger.info(f"  Transferred spring bones: {len(asset.spring_chains)} chains, "
               f"{len(asset.spring_colliders)} colliders")

    # Transfer skinning weights
    bone_indices, bone_weights, primary_bones, semantic_labels, body_regions = transfer_skinning(
        avatar, g_positions
    )
    asset.skin_bone_indices = bone_indices
    asset.skin_bone_weights = bone_weights
    asset.semantic_labels = semantic_labels
    asset.body_regions = np.array([body_region_to_enum(r) for r in body_regions], dtype=np.uint8)
    logger.info(f"  Transferred skinning: {len(set(primary_bones))} unique bone assignments")

    # Set metadata
    asset.metadata.entity_type = "noodling"
    asset.metadata.entity_id = entity_id or vrm_path.stem
    asset.metadata.display_name = display_name or avatar.metadata.title or vrm_path.stem
    asset.metadata.author = avatar.metadata.author
    asset.metadata.created = datetime.now().isoformat()
    asset.compute_bounds()

    # Region distribution stats
    region_counts = {}
    for region in body_regions:
        region_counts[region] = region_counts.get(region, 0) + 1
    logger.info(f"  Body regions: {region_counts}")

    # Save if output path provided
    if output_path:
        asset.save(output_path)
        logger.info(f"  Saved: {output_path}")

    return asset


def ply_to_radiance_filtered(
    ply_path: str,
    output_path: str,
    entity_id: str = "",
    display_name: str = "",
    filter_gaussians: bool = True,
    min_opacity: float = 0.8,
    max_scale: float = 0.05,
    max_sh_brightness: float = 2.0,
) -> RadianceAsset:
    """
    Convert OpenSplat PLY output to .radiance with optional filtering.

    This handles the PLY format from OpenSplat/3DGS training, which stores
    Gaussians with log-space scales and logit-space opacity.

    Args:
        ply_path: Input PLY file from OpenSplat
        output_path: Output .radiance file
        entity_id: Entity ID for scene protocol
        display_name: Human-readable name
        filter_gaussians: Apply background artifact filtering
        min_opacity: Minimum opacity for filtering (default 0.8)
        max_scale: Maximum scale for filtering (default 0.05)
        max_sh_brightness: Maximum SH brightness for filtering (default 2.0)

    Returns:
        Created RadianceAsset
    """
    from noodlestudio.core.semantic_world.radiance_format import RadianceAsset

    logger.info(f"Converting PLY to Radiance: {ply_path}")

    # Load PLY using radiance_format's built-in loader
    asset = RadianceAsset.from_ply(ply_path)
    logger.info(f"  Loaded {len(asset.positions):,} Gaussians from PLY")

    # Stats before filtering
    logger.info(f"  Scale range: {asset.scales.min():.5f} to {asset.scales.max():.5f}")
    logger.info(f"  Opacity range: {asset.opacities.min():.3f} to {asset.opacities.max():.3f}")
    logger.info(f"  SH DC range: {asset.sh_dc.min():.3f} to {asset.sh_dc.max():.3f}")

    # Apply filtering if requested
    if filter_gaussians:
        asset = filter_trained_gaussians(
            asset,
            min_opacity=min_opacity,
            max_scale=max_scale,
            max_sh_brightness=max_sh_brightness,
        )

    # Set metadata
    asset.metadata.entity_type = "noodling"
    asset.metadata.entity_id = entity_id or Path(ply_path).stem
    asset.metadata.display_name = display_name or Path(ply_path).stem
    asset.metadata.created = datetime.now().isoformat()
    asset.compute_bounds()

    # Save
    asset.save(output_path)
    logger.info(f"  Saved: {output_path}")

    return asset


def main():
    parser = argparse.ArgumentParser(
        description="Convert VRM avatar or trained PLY to semantic Gaussian splat (.radiance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic VRM conversion
    python -m noodlestudio.tools.vrm_to_radiance avatar.vrm

    # With custom output and name
    python -m noodlestudio.tools.vrm_to_radiance avatar.vrm -o red.radiance --name "Red Fire Anklebiter"

    # Convert OpenSplat PLY with filtering (removes white-background artifacts)
    python -m noodlestudio.tools.vrm_to_radiance trained.ply --filter

    # PLY with custom filter thresholds
    python -m noodlestudio.tools.vrm_to_radiance trained.ply --filter --min-opacity 0.9 --max-scale 0.03

    # PLY without filtering (keep all Gaussians)
    python -m noodlestudio.tools.vrm_to_radiance trained.ply --no-filter

    # Downsampled VRM for faster preview
    python -m noodlestudio.tools.vrm_to_radiance avatar.vrm --downsample 0.1 -o preview.radiance
        """
    )

    parser.add_argument("input_path", help="Path to input VRM or PLY file")
    parser.add_argument("-o", "--output", help="Output .radiance file path")
    parser.add_argument("--name", help="Entity display name")
    parser.add_argument("--entity-id", help="Entity ID for scene protocol")

    # VRM-specific options
    vrm_group = parser.add_argument_group("VRM options")
    vrm_group.add_argument("--downsample", type=float, default=1.0,
                          help="Vertex downsampling ratio (default: 1.0 = all vertices)")
    vrm_group.add_argument("--scale", type=float, default=0.005,
                          help="Base Gaussian scale (default: 0.005)")
    vrm_group.add_argument("--no-densify", action="store_true",
                          help="Disable densification (use original vertices only)")
    vrm_group.add_argument("--no-face-centers", action="store_true",
                          help="Skip adding face center Gaussians")
    vrm_group.add_argument("--no-edge-midpoints", action="store_true",
                          help="Skip adding edge midpoint Gaussians")
    vrm_group.add_argument("--no-adaptive-scale", action="store_true",
                          help="Use fixed scale instead of adaptive")

    # PLY filtering options (for trained Gaussians)
    ply_group = parser.add_argument_group("PLY filtering options (for OpenSplat output)")
    ply_group.add_argument("--filter", action="store_true",
                          help="Apply background artifact filtering to PLY (default for .ply)")
    ply_group.add_argument("--no-filter", action="store_true",
                          help="Disable filtering (keep all Gaussians from PLY)")
    ply_group.add_argument("--min-opacity", type=float, default=0.8,
                          help="Minimum opacity threshold for filtering (default: 0.8)")
    ply_group.add_argument("--max-scale", type=float, default=0.05,
                          help="Maximum scale threshold for filtering (default: 0.05)")
    ply_group.add_argument("--max-brightness", type=float, default=2.0,
                          help="Maximum SH brightness for filtering (default: 2.0)")

    parser.add_argument("--preview", action="store_true",
                       help="Generate preview PLY alongside .radiance")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s"
    )

    # Determine input type and output path
    input_path = Path(args.input_path)
    is_ply = input_path.suffix.lower() == '.ply'

    if args.output:
        output_path = args.output
    else:
        output_path = input_path.with_suffix('.radiance')

    # Convert based on input type
    try:
        if is_ply:
            # PLY from OpenSplat training
            # Default to filtering unless --no-filter specified
            do_filter = not args.no_filter

            asset = ply_to_radiance_filtered(
                ply_path=str(input_path),
                output_path=str(output_path),
                entity_id=args.entity_id or "",
                display_name=args.name or "",
                filter_gaussians=do_filter,
                min_opacity=args.min_opacity,
                max_scale=args.max_scale,
                max_sh_brightness=args.max_brightness,
            )

            # Summary for PLY
            print(f"\nPLY conversion complete!")
            print(f"  Gaussians: {asset.gaussian_count:,}")
            print(f"  Filtering: {'enabled' if do_filter else 'disabled'}")
            if do_filter:
                print(f"    min_opacity: {args.min_opacity}")
                print(f"    max_scale: {args.max_scale}")
                print(f"    max_brightness: {args.max_brightness}")

        else:
            # VRM avatar
            asset = vrm_to_radiance(
                vrm_path=str(input_path),
                output_path=str(output_path),
                entity_id=args.entity_id or "",
                display_name=args.name or "",
                downsample=args.downsample,
                base_scale=args.scale,
                densify=not args.no_densify,
                add_face_centers=not args.no_face_centers,
                add_edge_midpoints=not args.no_edge_midpoints,
                use_adaptive_scale=not args.no_adaptive_scale,
            )

            # Summary for VRM
            print(f"\nVRM conversion complete!")
            print(f"  Gaussians: {asset.gaussian_count:,}")
            print(f"  Bones: {asset.bone_count}")
            print(f"  Spring chains: {len(asset.spring_chains)}")

        # Generate preview PLY if requested
        if args.preview:
            preview_ply_path = Path(output_path).with_suffix('.preview.ply')
            asset.export_ply(str(preview_ply_path))
            print(f"  Preview PLY: {preview_ply_path}")

        print(f"  Output: {output_path}")

        # File size
        if Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
