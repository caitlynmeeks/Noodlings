"""
Spring Bone Simulation - Physics simulation for VRM spring bones.

Simulates pendulum-like motion for hair, cloth, accessories using:
- Verlet integration (stable, position-based)
- Stiffness: spring force toward rest pose
- Gravity: external force
- Drag: velocity damping
- Collision: spherical colliders

Based on VRM SecondaryAnimation / VRMC_springBone specification.

Author: Caitlyn + Claude
Date: December 2025
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np

from .vrm_parser import (
    Vector3,
    Quaternion,
    Transform,
    Bone,
    Skeleton,
    SpringBoneChain,
    SpringBoneCollider,
    SpringBoneSystem,
    VRMAvatar,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Math Utilities
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, returning zero if length is zero."""
    length = np.linalg.norm(v)
    if length < 1e-10:
        return np.zeros_like(v)
    return v / length


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (x, y, z, w format)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quaternion_from_to_rotation(from_dir: np.ndarray, to_dir: np.ndarray) -> np.ndarray:
    """Create quaternion that rotates from_dir to to_dir."""
    from_dir = normalize(from_dir)
    to_dir = normalize(to_dir)

    dot = np.dot(from_dir, to_dir)

    if dot > 0.9999:
        return np.array([0, 0, 0, 1], dtype=np.float32)

    if dot < -0.9999:
        # 180 degree rotation - find perpendicular axis
        axis = np.cross(np.array([1, 0, 0]), from_dir)
        if np.linalg.norm(axis) < 0.001:
            axis = np.cross(np.array([0, 1, 0]), from_dir)
        axis = normalize(axis)
        return np.array([axis[0], axis[1], axis[2], 0], dtype=np.float32)

    axis = np.cross(from_dir, to_dir)
    s = math.sqrt((1 + dot) * 2)
    inv_s = 1 / s

    return np.array([
        axis[0] * inv_s,
        axis[1] * inv_s,
        axis[2] * inv_s,
        s * 0.5
    ], dtype=np.float32)


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion."""
    # q * v * q^-1 where v is treated as quaternion (v.x, v.y, v.z, 0)
    qx, qy, qz, qw = q

    # Cross product part
    cx = qw * v[0] + qy * v[2] - qz * v[1]
    cy = qw * v[1] + qz * v[0] - qx * v[2]
    cz = qw * v[2] + qx * v[1] - qy * v[0]
    cw = -qx * v[0] - qy * v[1] - qz * v[2]

    # Second multiply
    return np.array([
        cx * qw + cw * -qx + cy * -qz - cz * -qy,
        cy * qw + cw * -qy + cz * -qx - cx * -qz,
        cz * qw + cw * -qz + cx * -qy - cy * -qx,
    ])


def mat4_transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Transform a point by a 4x4 matrix."""
    p = np.append(point, 1.0)
    result = matrix @ p
    return result[:3]


def mat4_transform_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Transform a vector by a 4x4 matrix (ignores translation)."""
    v = np.append(vector, 0.0)
    result = matrix @ v
    return result[:3]


# =============================================================================
# Simulation State
# =============================================================================

@dataclass
class SpringJoint:
    """State for a single spring bone joint."""
    bone_index: int
    # Rest state
    rest_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rest_length: float = 0.0
    # Dynamic state
    current_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    previous_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # For transform computation
    initial_local_rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    bone_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))  # Typically Y-up


@dataclass
class SpringChainState:
    """Simulation state for an entire spring chain."""
    chain: SpringBoneChain
    joints: List[SpringJoint] = field(default_factory=list)
    # Chain-level settings (cached from SpringBoneChain)
    stiffness: float = 1.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, -1, 0]))
    gravity_power: float = 0.0
    drag: float = 0.4
    hit_radius: float = 0.02


@dataclass
class ColliderState:
    """Runtime state for a spherical collider."""
    collider: SpringBoneCollider
    world_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.1


# =============================================================================
# Spring Bone Simulator
# =============================================================================

class SpringBoneSimulator:
    """
    Physics simulation for VRM spring bones.

    Uses Verlet integration with collision detection for stable,
    realistic hair/cloth secondary animation.

    Usage:
        sim = SpringBoneSimulator(avatar)
        sim.initialize()

        # Each frame:
        sim.update(dt, bone_transforms)
        transforms = sim.get_bone_transforms()
    """

    def __init__(self, avatar: VRMAvatar):
        """
        Initialize simulator from VRM avatar.

        Args:
            avatar: Parsed VRM avatar with spring bone definitions
        """
        self.avatar = avatar
        self.skeleton = avatar.skeleton
        self.spring_system = avatar.spring_bones

        # Simulation state
        self.chain_states: List[SpringChainState] = []
        self.collider_states: List[ColliderState] = []

        # Bone transforms (updated each frame by animation system)
        self.bone_world_matrices: Dict[int, np.ndarray] = {}

        # Output: modified bone rotations
        self.bone_rotations: Dict[int, np.ndarray] = {}

        # Simulation parameters
        self.gravity = np.array([0, -9.8, 0], dtype=np.float32)
        self.fixed_timestep = 1.0 / 60.0
        self.accumulated_time = 0.0

        self._initialized = False

    def initialize(self):
        """
        Initialize simulation state from rest pose.

        Call once after loading avatar, before starting simulation.
        """
        logger.info(f"Initializing spring bone simulation: {len(self.spring_system.chains)} chains")

        # Initialize chain states
        self.chain_states = []
        for chain in self.spring_system.chains:
            state = self._create_chain_state(chain)
            self.chain_states.append(state)

        # Initialize collider states
        self.collider_states = []
        for collider in self.spring_system.colliders:
            state = ColliderState(
                collider=collider,
                radius=collider.radius,
            )
            self.collider_states.append(state)

        self._initialized = True
        logger.info(f"Initialized {len(self.chain_states)} chains, {len(self.collider_states)} colliders")

    def _create_chain_state(self, chain: SpringBoneChain) -> SpringChainState:
        """Create simulation state for a spring chain."""
        state = SpringChainState(
            chain=chain,
            stiffness=chain.stiffness,
            gravity=chain.gravity_dir.to_array() if chain.gravity_dir else np.array([0, -1, 0]),
            gravity_power=chain.gravity_power,
            drag=chain.drag_force,
            hit_radius=chain.hit_radius,
        )

        # Create joints for each bone in chain
        for i, bone_idx in enumerate(chain.bone_indices):
            if bone_idx >= len(self.skeleton.bones):
                continue

            bone = self.skeleton.bones[bone_idx]

            joint = SpringJoint(
                bone_index=bone_idx,
                rest_position=bone.transform.position.to_array(),
                initial_local_rotation=bone.transform.rotation.to_array(),
            )

            # Compute rest length (distance to parent)
            if i > 0:
                # Parent is previous bone in chain
                parent_bone_idx = chain.bone_indices[i - 1]
                if parent_bone_idx < len(self.skeleton.bones):
                    parent_bone = self.skeleton.bones[parent_bone_idx]
                    joint.rest_length = np.linalg.norm(
                        joint.rest_position - parent_bone.transform.position.to_array()
                    )
            else:
                # First joint in chain - parent is bone's actual parent
                if bone.parent_index >= 0 and bone.parent_index < len(self.skeleton.bones):
                    parent_bone = self.skeleton.bones[bone.parent_index]
                    joint.rest_length = np.linalg.norm(
                        joint.rest_position - parent_bone.transform.position.to_array()
                    )
                else:
                    # No parent - use a small default length
                    joint.rest_length = 0.1

            # Set initial positions
            joint.current_position = joint.rest_position.copy()
            joint.previous_position = joint.rest_position.copy()

            # Compute bone axis (direction from parent to this bone)
            if i > 0:
                parent_bone_idx = chain.bone_indices[i - 1]
                parent_bone = self.skeleton.bones[parent_bone_idx]
                direction = joint.rest_position - parent_bone.transform.position.to_array()
            elif bone.parent_index >= 0 and bone.parent_index < len(self.skeleton.bones):
                parent_bone = self.skeleton.bones[bone.parent_index]
                direction = joint.rest_position - parent_bone.transform.position.to_array()
            else:
                direction = np.array([0, -1, 0])  # Default: hanging down

            joint.bone_axis = normalize(direction)

            state.joints.append(joint)

        return state

    def set_bone_transforms(self, transforms: Dict[int, np.ndarray]):
        """
        Update bone world transforms from animation system.

        Args:
            transforms: Dict mapping bone index to 4x4 world matrix
        """
        self.bone_world_matrices = transforms

        # Update collider world positions
        for collider_state in self.collider_states:
            bone_idx = collider_state.collider.bone_index
            if bone_idx in transforms:
                matrix = transforms[bone_idx]
                offset = collider_state.collider.offset.to_array()
                collider_state.world_position = mat4_transform_point(matrix, offset)

    def update(self, dt: float, bone_transforms: Optional[Dict[int, np.ndarray]] = None):
        """
        Step simulation forward.

        Args:
            dt: Delta time in seconds
            bone_transforms: Optional updated bone transforms
        """
        if not self._initialized:
            logger.warning("SpringBoneSimulator not initialized")
            return

        if bone_transforms:
            self.set_bone_transforms(bone_transforms)

        # Fixed timestep simulation for stability
        self.accumulated_time += dt

        while self.accumulated_time >= self.fixed_timestep:
            self._simulation_step(self.fixed_timestep)
            self.accumulated_time -= self.fixed_timestep

    def _simulation_step(self, dt: float):
        """Single physics simulation step."""
        for chain_state in self.chain_states:
            self._simulate_chain(chain_state, dt)

    def _simulate_chain(self, state: SpringChainState, dt: float):
        """Simulate a single spring chain."""
        for i, joint in enumerate(state.joints):
            # Get parent world position
            if i > 0:
                # Use previous joint's simulated position
                parent_pos = state.joints[i - 1].current_position
            else:
                # First joint in chain - get parent bone's animated world position
                bone = self.skeleton.bones[joint.bone_index] if joint.bone_index < len(self.skeleton.bones) else None
                parent_bone_idx = bone.parent_index if bone else -1

                if parent_bone_idx >= 0 and parent_bone_idx in self.bone_world_matrices:
                    parent_pos = self.bone_world_matrices[parent_bone_idx][:3, 3].copy()
                elif joint.bone_index in self.bone_world_matrices:
                    # Use this bone's position as reference
                    parent_pos = self.bone_world_matrices[joint.bone_index][:3, 3].copy()
                else:
                    parent_pos = joint.rest_position.copy()

            # Compute target (rest) position relative to current parent
            target_pos = self._compute_rest_world_position(joint, parent_pos, i, state)

            # Verlet integration
            velocity = joint.current_position - joint.previous_position

            # Apply drag
            velocity *= (1.0 - state.drag)

            # Compute forces
            force = np.zeros(3, dtype=np.float32)

            # Gravity
            force += state.gravity * state.gravity_power * 9.8

            # Stiffness (spring force toward target/rest position)
            stiffness_force = (target_pos - joint.current_position) * state.stiffness
            force += stiffness_force

            # Store previous position
            joint.previous_position = joint.current_position.copy()

            # Integrate
            joint.current_position = joint.current_position + velocity + force * dt * dt

            # Constraint: maintain bone length from parent
            if joint.rest_length > 0:
                direction = joint.current_position - parent_pos
                length = np.linalg.norm(direction)
                if length > 1e-6:
                    direction = direction / length
                else:
                    direction = joint.bone_axis
                joint.current_position = parent_pos + direction * joint.rest_length

            # Collision detection
            self._apply_collisions(joint, state.hit_radius)

    def _compute_rest_world_position(
        self,
        joint: SpringJoint,
        parent_pos: np.ndarray,
        joint_index: int,
        state: SpringChainState,
    ) -> np.ndarray:
        """
        Compute target position (where joint wants to be relative to parent).

        For spring bones, this is the position offset from parent along bone axis,
        maintaining rest length but allowing the chain to follow parent motion.
        """
        # Target is: parent position + bone direction * bone length
        # The bone direction starts as the rest axis but can be influenced by animation

        if joint.rest_length > 0:
            # Offset from parent in rest direction
            return parent_pos + joint.bone_axis * joint.rest_length
        else:
            # No rest length defined - use bone world position if available
            if joint.bone_index in self.bone_world_matrices:
                return self.bone_world_matrices[joint.bone_index][:3, 3].copy()
            return parent_pos

    def _apply_collisions(self, joint: SpringJoint, hit_radius: float):
        """Apply collision response against all colliders."""
        for collider_state in self.collider_states:
            # Sphere-sphere collision
            delta = joint.current_position - collider_state.world_position
            distance = np.linalg.norm(delta)

            min_distance = hit_radius + collider_state.radius

            if distance < min_distance and distance > 1e-6:
                # Push joint out of collider
                direction = delta / distance
                joint.current_position = (
                    collider_state.world_position + direction * min_distance
                )

    def get_bone_rotations(self) -> Dict[int, np.ndarray]:
        """
        Get modified bone rotations for rendering.

        Returns:
            Dict mapping bone index to quaternion (x, y, z, w)
        """
        rotations = {}

        for chain_state in self.chain_states:
            for i, joint in enumerate(chain_state.joints):
                # Compute rotation from bone's initial direction to current direction
                bone_idx = joint.bone_index

                if i < len(chain_state.joints) - 1:
                    # Direction to next joint
                    next_joint = chain_state.joints[i + 1]
                    current_dir = normalize(next_joint.current_position - joint.current_position)
                else:
                    # Use stored bone axis for leaf joints
                    current_dir = joint.bone_axis

                # Initial direction (rest pose)
                initial_dir = joint.bone_axis

                # Compute rotation from initial to current
                rotation = quaternion_from_to_rotation(initial_dir, current_dir)

                # Combine with initial local rotation
                final_rotation = quaternion_multiply(rotation, joint.initial_local_rotation)

                rotations[bone_idx] = final_rotation

        return rotations

    def get_joint_positions(self) -> Dict[int, np.ndarray]:
        """
        Get current joint world positions.

        Returns:
            Dict mapping bone index to position (x, y, z)
        """
        positions = {}

        for chain_state in self.chain_states:
            for joint in chain_state.joints:
                positions[joint.bone_index] = joint.current_position.copy()

        return positions

    def reset(self):
        """Reset all joints to rest pose."""
        for chain_state in self.chain_states:
            for joint in chain_state.joints:
                joint.current_position = joint.rest_position.copy()
                joint.previous_position = joint.rest_position.copy()


# =============================================================================
# Gaussian Deformation
# =============================================================================

class GaussianSpringDeformer:
    """
    Apply spring bone transforms to skinned Gaussians.

    Combines:
    - LBS (Linear Blend Skinning) from skeleton
    - Spring bone secondary animation
    - Blend shapes (morph targets)
    """

    def __init__(
        self,
        gaussian_positions: np.ndarray,
        bone_weights: np.ndarray,
        bone_indices: np.ndarray,
    ):
        """
        Initialize deformer.

        Args:
            gaussian_positions: (N, 3) rest pose positions
            bone_weights: (N, 4) skinning weights per Gaussian
            bone_indices: (N, 4) bone indices per Gaussian
        """
        self.rest_positions = gaussian_positions.copy()
        self.bone_weights = bone_weights
        self.bone_indices = bone_indices.astype(np.int32)

        self.num_gaussians = gaussian_positions.shape[0]
        self.deformed_positions = gaussian_positions.copy()

    def apply_transforms(
        self,
        bone_matrices: Dict[int, np.ndarray],
        spring_rotations: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Apply skeleton + spring bone transforms.

        Args:
            bone_matrices: Dict mapping bone index to 4x4 world matrix
            spring_rotations: Optional spring bone rotation overrides

        Returns:
            (N, 3) deformed positions
        """
        # If spring rotations provided, modify the bone matrices
        if spring_rotations:
            bone_matrices = self._apply_spring_rotations(bone_matrices, spring_rotations)

        # Linear Blend Skinning
        for i in range(self.num_gaussians):
            pos = np.zeros(3, dtype=np.float32)

            for j in range(4):
                bone_idx = self.bone_indices[i, j]
                weight = self.bone_weights[i, j]

                if weight > 0 and bone_idx in bone_matrices:
                    matrix = bone_matrices[bone_idx]
                    transformed = mat4_transform_point(matrix, self.rest_positions[i])
                    pos += transformed * weight

            self.deformed_positions[i] = pos

        return self.deformed_positions

    def _apply_spring_rotations(
        self,
        bone_matrices: Dict[int, np.ndarray],
        spring_rotations: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """Apply spring bone rotation overrides to bone matrices."""
        modified = dict(bone_matrices)

        for bone_idx, rotation in spring_rotations.items():
            if bone_idx in modified:
                # Replace rotation component of matrix
                matrix = modified[bone_idx].copy()

                # Convert quaternion to rotation matrix
                rot_matrix = self._quaternion_to_matrix(rotation)

                # Keep translation, replace rotation
                matrix[:3, :3] = rot_matrix
                modified[bone_idx] = matrix

        return modified

    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
        x, y, z, w = q

        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        yy = y * y
        yz = y * z
        yw = y * w
        zz = z * z
        zw = z * w

        return np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)],
        ], dtype=np.float32)


# =============================================================================
# High-Level Integration
# =============================================================================

def create_spring_simulation(avatar: VRMAvatar) -> SpringBoneSimulator:
    """
    Create and initialize a spring bone simulator.

    Args:
        avatar: Parsed VRM avatar

    Returns:
        Initialized SpringBoneSimulator
    """
    sim = SpringBoneSimulator(avatar)
    sim.initialize()
    return sim


def create_gaussian_deformer(
    avatar: VRMAvatar,
    gaussian_positions: np.ndarray,
) -> Optional[GaussianSpringDeformer]:
    """
    Create a Gaussian deformer from VRM skinning data.

    Args:
        avatar: Parsed VRM avatar with skinning
        gaussian_positions: (N, 3) Gaussian positions

    Returns:
        GaussianSpringDeformer or None if no skinning data
    """
    # Collect all skinning weights from meshes
    all_weights = []
    all_indices = []

    for mesh in avatar.meshes:
        if mesh.joint_weights is not None and mesh.joint_indices is not None:
            all_weights.append(mesh.joint_weights)
            all_indices.append(mesh.joint_indices)

    if not all_weights:
        logger.warning("No skinning data found in avatar")
        return None

    weights = np.vstack(all_weights)
    indices = np.vstack(all_indices)

    # Ensure we have weights for all Gaussians
    num_gaussians = gaussian_positions.shape[0]
    if weights.shape[0] != num_gaussians:
        logger.warning(f"Skinning data mismatch: {weights.shape[0]} weights, {num_gaussians} Gaussians")
        # Pad or truncate
        if weights.shape[0] < num_gaussians:
            pad_count = num_gaussians - weights.shape[0]
            weights = np.vstack([weights, np.zeros((pad_count, 4))])
            indices = np.vstack([indices, np.zeros((pad_count, 4), dtype=np.int32)])
        else:
            weights = weights[:num_gaussians]
            indices = indices[:num_gaussians]

    return GaussianSpringDeformer(gaussian_positions, weights, indices)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    # Test with a simple simulated skeleton
    print("Spring Bone Simulation Test")
    print("=" * 40)

    # Create a minimal test skeleton
    from vrm_parser import VRMAvatar, Skeleton, Bone, SpringBoneSystem, SpringBoneChain, Transform

    avatar = VRMAvatar()

    # Add some bones
    for i in range(5):
        bone = Bone(
            name=f"hair_{i}",
            index=i,
            parent_index=i - 1 if i > 0 else -1,
            transform=Transform(
                position=Vector3(0, 1 - i * 0.2, 0),
                rotation=Quaternion(0, 0, 0, 1),
            ),
        )
        avatar.skeleton.bones.append(bone)

    # Add a spring chain
    chain = SpringBoneChain(
        name="hair_chain",
        bone_indices=[0, 1, 2, 3, 4],
        stiffness=0.8,
        gravity_power=0.1,
        gravity_dir=Vector3(0, -1, 0),
        drag_force=0.3,
        hit_radius=0.02,
    )
    avatar.spring_bones.chains.append(chain)

    # Create simulator
    sim = create_spring_simulation(avatar)

    # Run simulation for a few frames
    print("\nSimulating 60 frames...")
    for frame in range(60):
        sim.update(1.0 / 60.0)

    # Print joint positions
    positions = sim.get_joint_positions()
    print("\nFinal joint positions:")
    for bone_idx, pos in sorted(positions.items()):
        print(f"  Bone {bone_idx}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print("\nSimulation test complete!")
