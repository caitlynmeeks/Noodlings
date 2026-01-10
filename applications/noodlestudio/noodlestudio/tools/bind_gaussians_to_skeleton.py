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
#   Bind Trained Gaussians to VRM Skeleton.
#
#   Takes trained Gaussians from OpenSplat and binds them to ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tools.bind_gaussians_to_skeleton
# PURPOSE:  Bind Trained Gaussians to VRM Skeleton.
# LAYER:    Studio / Tools
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   load_gaussian_ply(), transfer_skinning_weights(), bind_gaussians_to_skeleton()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def load_gaussian_ply(ply_path: str) -> dict:
    """
    Load trained Gaussians from OpenSplat PLY output.

    Returns dict with:
        positions: (N, 3) float32
        scales: (N, 3) float32 (log scale)
        rotations: (N, 4) float32 quaternion (wxyz)
        opacities: (N,) float32 (logit)
        sh_dc: (N, 3) float32 (DC component)
        sh_rest: (N, K, 3) float32 (higher order SH, optional)
    """
    import plyfile

    plydata = plyfile.PlyData.read(ply_path)
    vertex = plydata['vertex']

    n = len(vertex['x'])
    logger.info(f"Loading {n:,} Gaussians from {ply_path}")

    # Positions
    positions = np.stack([
        vertex['x'], vertex['y'], vertex['z']
    ], axis=-1).astype(np.float32)

    # Scales (stored as log)
    scales = np.stack([
        vertex['scale_0'], vertex['scale_1'], vertex['scale_2']
    ], axis=-1).astype(np.float32)

    # Rotations (stored as wxyz quaternion)
    rotations = np.stack([
        vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']
    ], axis=-1).astype(np.float32)

    # Opacities (stored as logit)
    opacities = vertex['opacity'].astype(np.float32)

    # Spherical harmonics DC component
    sh_dc = np.stack([
        vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
    ], axis=-1).astype(np.float32)

    # Check for higher-order SH
    sh_rest = None
    sh_keys = [k for k in vertex.data.dtype.names if k.startswith('f_rest_')]
    if sh_keys:
        n_sh = len(sh_keys) // 3
        sh_rest_list = []
        for i in range(n_sh):
            sh_rest_list.append(np.stack([
                vertex[f'f_rest_{i*3}'],
                vertex[f'f_rest_{i*3+1}'],
                vertex[f'f_rest_{i*3+2}']
            ], axis=-1))
        sh_rest = np.stack(sh_rest_list, axis=1).astype(np.float32)
        logger.info(f"  Loaded {n_sh} additional SH bands")

    return {
        'positions': positions,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'sh_dc': sh_dc,
        'sh_rest': sh_rest,
    }


def transfer_skinning_weights(
    gaussian_positions: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_joint_indices: np.ndarray,
    mesh_joint_weights: np.ndarray,
    k_neighbors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transfer skinning weights from mesh vertices to Gaussians.

    Uses inverse distance weighting from K nearest mesh vertices.

    Args:
        gaussian_positions: (N, 3) Gaussian positions
        mesh_vertices: (M, 3) mesh vertex positions
        mesh_joint_indices: (M, 4) bone indices per vertex
        mesh_joint_weights: (M, 4) bone weights per vertex
        k_neighbors: number of neighbors to interpolate from

    Returns:
        joint_indices: (N, 4) bone indices for each Gaussian
        joint_weights: (N, 4) bone weights for each Gaussian
    """
    from scipy.spatial import cKDTree

    n_gaussians = len(gaussian_positions)
    logger.info(f"Transferring skinning weights to {n_gaussians:,} Gaussians...")

    # Build KD-tree of mesh vertices
    tree = cKDTree(mesh_vertices)

    # Query K nearest neighbors for each Gaussian
    distances, indices = tree.query(gaussian_positions, k=k_neighbors)

    # Output arrays
    out_joint_indices = np.zeros((n_gaussians, 4), dtype=np.int32)
    out_joint_weights = np.zeros((n_gaussians, 4), dtype=np.float32)

    for i in range(n_gaussians):
        # Get neighbor vertex data
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]

        # Inverse distance weights (avoid div by zero)
        neighbor_distances = np.maximum(neighbor_distances, 1e-6)
        inv_dist_weights = 1.0 / neighbor_distances
        inv_dist_weights = inv_dist_weights / inv_dist_weights.sum()

        # Collect all bone influences from neighbors
        bone_weight_sum = {}
        for j, (v_idx, dist_weight) in enumerate(zip(neighbor_indices, inv_dist_weights)):
            v_bones = mesh_joint_indices[v_idx]
            v_weights = mesh_joint_weights[v_idx]

            for bone_idx, bone_weight in zip(v_bones, v_weights):
                if bone_weight > 0:
                    key = int(bone_idx)
                    bone_weight_sum[key] = bone_weight_sum.get(key, 0.0) + bone_weight * dist_weight

        # Sort by weight and take top 4
        sorted_bones = sorted(bone_weight_sum.items(), key=lambda x: -x[1])[:4]

        # Normalize weights
        total = sum(w for _, w in sorted_bones)
        if total > 0:
            for j, (bone_idx, weight) in enumerate(sorted_bones):
                out_joint_indices[i, j] = bone_idx
                out_joint_weights[i, j] = weight / total

        if (i + 1) % 50000 == 0:
            logger.info(f"  Processed {i+1:,}/{n_gaussians:,}")

    logger.info(f"  Transferred skinning to {n_gaussians:,} Gaussians")
    return out_joint_indices, out_joint_weights


def bind_gaussians_to_skeleton(
    gaussian_ply_path: str,
    vrm_path: str,
    output_path: str,
    entity_id: str = "",
    display_name: str = "",
) -> 'RadianceAsset':
    """
    Bind trained Gaussians to VRM skeleton.

    Args:
        gaussian_ply_path: Path to trained Gaussians (.ply from OpenSplat)
        vrm_path: Path to original VRM (for skeleton)
        output_path: Output .radiance file path
        entity_id: Entity ID for scene protocol
        display_name: Human-readable name

    Returns:
        Created RadianceAsset
    """
    from ..core.semantic_world.radiance_format import RadianceAsset, save_radiance
    from ..core.semantic_world.vrm_parser import parse_vrm

    gaussian_ply_path = Path(gaussian_ply_path)
    vrm_path = Path(vrm_path)
    output_path = Path(output_path)

    logger.info(f"Binding Gaussians to skeleton:")
    logger.info(f"  Gaussians: {gaussian_ply_path}")
    logger.info(f"  VRM: {vrm_path}")
    logger.info(f"  Output: {output_path}")

    # Load trained Gaussians
    gaussians = load_gaussian_ply(str(gaussian_ply_path))
    n_gaussians = len(gaussians['positions'])

    # Parse VRM for skeleton and skinning
    avatar = parse_vrm(str(vrm_path))
    logger.info(f"  VRM: {avatar.bone_count} bones, {avatar.vertex_count} vertices")

    # Collect all mesh vertices and skinning data
    all_vertices = []
    all_joint_indices = []
    all_joint_weights = []

    for mesh in avatar.meshes:
        all_vertices.append(mesh.vertices)
        if mesh.joint_indices is not None:
            all_joint_indices.append(mesh.joint_indices)
            all_joint_weights.append(mesh.joint_weights)

    mesh_vertices = np.vstack(all_vertices)

    if all_joint_indices:
        mesh_joint_indices = np.vstack(all_joint_indices)
        mesh_joint_weights = np.vstack(all_joint_weights)
    else:
        logger.warning("  No skinning data in VRM, using identity weights")
        mesh_joint_indices = np.zeros((len(mesh_vertices), 4), dtype=np.int32)
        mesh_joint_weights = np.zeros((len(mesh_vertices), 4), dtype=np.float32)
        mesh_joint_weights[:, 0] = 1.0

    # Transfer skinning weights
    joint_indices, joint_weights = transfer_skinning_weights(
        gaussians['positions'],
        mesh_vertices,
        mesh_joint_indices,
        mesh_joint_weights,
    )

    # Convert Gaussian data to radiance format
    # Scales: OpenSplat stores log(scale), we need actual scale
    scales = np.exp(gaussians['scales'])

    # Rotations: OpenSplat stores wxyz, radiance uses xyzw
    rotations_wxyz = gaussians['rotations']
    rotations_xyzw = np.zeros_like(rotations_wxyz)
    rotations_xyzw[:, 0] = rotations_wxyz[:, 1]  # x
    rotations_xyzw[:, 1] = rotations_wxyz[:, 2]  # y
    rotations_xyzw[:, 2] = rotations_wxyz[:, 3]  # z
    rotations_xyzw[:, 3] = rotations_wxyz[:, 0]  # w

    # Opacities: OpenSplat stores logit, convert to sigmoid
    opacities = 1.0 / (1.0 + np.exp(-gaussians['opacities']))

    # SH to RGB (DC component only for now)
    # SH DC formula: color = 0.5 + SH_C0 * sh_dc
    SH_C0 = 0.28209479177387814
    rgb = 0.5 + SH_C0 * gaussians['sh_dc']
    rgb = np.clip(rgb, 0, 1)

    # Create RadianceAsset
    asset = RadianceAsset(
        entity_id=entity_id or Path(output_path).stem,
        display_name=display_name or Path(output_path).stem,
    )

    asset.positions = gaussians['positions']
    asset.scales = scales
    asset.rotations = rotations_xyzw
    asset.opacities = opacities
    asset.sh_dc = gaussians['sh_dc']

    # Store higher-order SH if available
    if gaussians['sh_rest'] is not None:
        asset.sh_rest = gaussians['sh_rest']

    # Store skeleton
    asset.bones = avatar.bones
    asset.bone_names = [b.name for b in avatar.bones] if avatar.bones else []

    # Store skinning
    asset.joint_indices = joint_indices
    asset.joint_weights = joint_weights

    # Add semantic labels based on bone assignments
    asset.semantic_labels = _compute_body_regions(
        joint_indices, joint_weights, avatar.bones
    )

    # Save
    save_radiance(asset, str(output_path))
    logger.info(f"  Saved: {output_path} ({n_gaussians:,} Gaussians)")

    return asset


def _compute_body_regions(
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    bones: list,
) -> np.ndarray:
    """Compute body region labels from bone assignments."""
    n = len(joint_indices)
    labels = np.zeros(n, dtype=np.uint8)

    if not bones:
        return labels

    # Build bone name to region mapping
    bone_to_region = {}
    for i, bone in enumerate(bones):
        name_lower = bone.name.lower()
        if 'head' in name_lower or 'neck' in name_lower or 'eye' in name_lower:
            bone_to_region[i] = 1  # head
        elif 'spine' in name_lower or 'chest' in name_lower or 'hips' in name_lower:
            bone_to_region[i] = 2  # torso
        elif 'left' in name_lower and ('arm' in name_lower or 'shoulder' in name_lower):
            bone_to_region[i] = 3  # left_arm
        elif 'right' in name_lower and ('arm' in name_lower or 'shoulder' in name_lower):
            bone_to_region[i] = 4  # right_arm
        elif 'left' in name_lower and ('leg' in name_lower or 'foot' in name_lower or 'toe' in name_lower):
            bone_to_region[i] = 5  # left_leg
        elif 'right' in name_lower and ('leg' in name_lower or 'foot' in name_lower or 'toe' in name_lower):
            bone_to_region[i] = 6  # right_leg
        elif 'left' in name_lower and 'hand' in name_lower:
            bone_to_region[i] = 7  # left_hand
        elif 'right' in name_lower and 'hand' in name_lower:
            bone_to_region[i] = 8  # right_hand
        else:
            bone_to_region[i] = 0  # other

    # Assign region based on dominant bone
    for i in range(n):
        dominant_bone = joint_indices[i, 0]
        labels[i] = bone_to_region.get(dominant_bone, 0)

    return labels


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Bind trained Gaussians to VRM skeleton"
    )
    parser.add_argument("gaussian_ply", help="Trained Gaussians PLY from OpenSplat")
    parser.add_argument("vrm", help="Original VRM file (for skeleton)")
    parser.add_argument("-o", "--output", help="Output .radiance file")
    parser.add_argument("--name", help="Entity display name")
    parser.add_argument("--entity-id", help="Entity ID for scene protocol")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(message)s'
    )

    gaussian_path = Path(args.gaussian_ply)
    vrm_path = Path(args.vrm)

    if args.output:
        output_path = args.output
    else:
        output_path = str(gaussian_path.with_suffix('.radiance'))

    asset = bind_gaussians_to_skeleton(
        gaussian_ply_path=str(gaussian_path),
        vrm_path=str(vrm_path),
        output_path=output_path,
        entity_id=args.entity_id or "",
        display_name=args.name or "",
    )

    print(f"Created: {output_path}")
    print(f"  Gaussians: {asset.gaussian_count:,}")
    print(f"  Bones: {len(asset.bones) if asset.bones else 0}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
