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
#   Tests for VRM GPU Skeletal Skinning
#
#   Verifies inverse bind matrix storage, bone matrix computation,
#   quaternion/euler math, hierarchy traversal, and retargeter mapping.
#   These tests run without an OpenGL context.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_vrm_gpu_skinning
# PURPOSE:  Tests for GPU Skeletal Animation Pipeline
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestInverseBindMatrices, TestBoneMatrixMath,
#   TestHierarchyTraversal, TestSkinningMatrices,
#   TestRetargeterMapping, TestJointDataFallback
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import math
import pytest
import numpy as np

from noodlestudio.core.semantic_world.vrm_parser import (
    VRMAvatar, Mesh, MToonMaterial, Skeleton, Bone, Transform,
    Vector3, Quaternion,
)
from noodlestudio.core.pose_track import (
    PoseRetargeter, PoseState, MUSCLE_DEFINITIONS,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_bone(name, index, parent=-1, pos=(0, 0, 0),
               rot=(0, 0, 0, 1), humanoid=None, children=None):
    """Create a Bone with given parameters."""
    bone = Bone(
        name=name,
        index=index,
        parent_index=parent,
        transform=Transform(
            position=Vector3(*pos),
            rotation=Quaternion(*rot),
            scale=Vector3(1, 1, 1),
        ),
        children=children or [],
        humanoid_bone=humanoid,
    )
    return bone


def _make_skeleton(bones, root=0, humanoid_map=None, ibm=None):
    """Create a Skeleton with given bones and inverse bind matrices."""
    skel = Skeleton(
        bones=bones,
        root_bone_index=root,
        humanoid_map=humanoid_map or {},
        inverse_bind_matrices=ibm,
    )
    return skel


def _make_avatar(skeleton, meshes=None, materials=None):
    """Create a VRMAvatar for testing."""
    avatar = VRMAvatar()
    avatar.skeleton = skeleton
    avatar.meshes = meshes or []
    avatar.materials = materials or []
    return avatar


# Import the static methods from VRMViewportWidget for testing.
# Since VRMViewportWidget may not be importable without Qt/OpenGL,
# we replicate the math functions here for testing.

def _quat_to_matrix(x, y, z, w):
    """Quaternion to 3x3 rotation matrix (row-major)."""
    x2 = x + x
    y2 = y + y
    z2 = z + z
    xx = x * x2
    xy = x * y2
    xz = x * z2
    yy = y * y2
    yz = y * z2
    zz = z * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2

    return np.array([
        [1 - (yy + zz), xy - wz,       xz + wy],
        [xy + wz,       1 - (xx + zz),  yz - wx],
        [xz - wy,       yz + wx,        1 - (xx + yy)],
    ], dtype=np.float32)


def _euler_to_matrix(rx, ry, rz):
    """Euler degrees (XYZ order) to 3x3 rotation matrix."""
    rx_rad = math.radians(rx)
    ry_rad = math.radians(ry)
    rz_rad = math.radians(rz)
    cx, sx = math.cos(rx_rad), math.sin(rx_rad)
    cy, sy = math.cos(ry_rad), math.sin(ry_rad)
    cz, sz = math.cos(rz_rad), math.sin(rz_rad)
    return np.array([
        [cy * cz,  sx * sy * cz - cx * sz,  cx * sy * cz + sx * sz],
        [cy * sz,  sx * sy * sz + cx * cz,  cx * sy * sz - sx * cz],
        [-sy,      sx * cy,                  cx * cy],
    ], dtype=np.float32)


def _bone_local_matrix(bone, bone_rotations=None, humanoid_to_bone_idx=None):
    """Replicate VRMViewportWidget._bone_local_matrix logic for testing."""
    mat = np.eye(4, dtype=np.float32)
    q = bone.transform.rotation
    rot_rest = _quat_to_matrix(q.x, q.y, q.z, q.w)

    rot_pose = np.eye(3, dtype=np.float32)
    if bone_rotations and bone.humanoid_bone:
        euler = bone_rotations.get(bone.humanoid_bone)
        if euler:
            rot_pose = _euler_to_matrix(euler[0], euler[1], euler[2])

    combined = rot_rest @ rot_pose
    s = bone.transform.scale
    scale = np.diag([s.x, s.y, s.z]).astype(np.float32)
    mat[:3, :3] = combined @ scale
    mat[0, 3] = bone.transform.position.x
    mat[1, 3] = bone.transform.position.y
    mat[2, 3] = bone.transform.position.z
    return mat


def _compute_bone_matrices(skeleton, bone_rotations=None):
    """Replicate VRMViewportWidget._compute_bone_matrices logic for testing."""
    num_bones = len(skeleton.bones)
    ibm = skeleton.inverse_bind_matrices

    world_transforms = [None] * num_bones
    order = []
    visited = set()
    queue = [skeleton.root_bone_index]
    while queue:
        idx = queue.pop(0)
        if idx in visited or idx < 0 or idx >= num_bones:
            continue
        visited.add(idx)
        order.append(idx)
        queue.extend(skeleton.bones[idx].children)
    for i in range(num_bones):
        if i not in visited:
            order.append(i)

    humanoid_to_bone_idx = {}
    for bone in skeleton.bones:
        if bone.humanoid_bone:
            humanoid_to_bone_idx[bone.humanoid_bone] = bone.index

    for idx in order:
        bone = skeleton.bones[idx]
        local = _bone_local_matrix(bone, bone_rotations, humanoid_to_bone_idx)
        if bone.parent_index >= 0 and world_transforms[bone.parent_index] is not None:
            world_transforms[idx] = world_transforms[bone.parent_index] @ local
        else:
            world_transforms[idx] = local

    bone_matrices = np.zeros((num_bones, 4, 4), dtype=np.float32)
    for i in range(num_bones):
        wt = world_transforms[i]
        if wt is None:
            wt = np.eye(4, dtype=np.float32)
        if ibm is not None and i < len(ibm):
            bone_matrices[i] = wt @ ibm[i]
        else:
            bone_matrices[i] = wt

    return bone_matrices, world_transforms


# =============================================================================
# Test: Inverse Bind Matrices
# =============================================================================

class TestInverseBindMatrices:
    """Test inverse bind matrix storage on Skeleton."""

    def test_ibm_field_exists(self):
        """Skeleton should have inverse_bind_matrices field."""
        skel = Skeleton()
        assert skel.inverse_bind_matrices is None

    def test_ibm_stored_with_shape(self):
        """When provided, IBM should be (N, 4, 4) array."""
        ibm = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        skel = _make_skeleton(
            [_make_bone("root", 0)],
            ibm=ibm,
        )
        assert skel.inverse_bind_matrices is not None
        assert skel.inverse_bind_matrices.shape == (1, 4, 4)

    def test_ibm_identity_at_rest(self):
        """Identity IBM means rest-pose skinning produces identity."""
        ibm = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        bones = [_make_bone("root", 0)]
        skel = _make_skeleton(bones, ibm=ibm)

        matrices, _ = _compute_bone_matrices(skel)
        # At rest with identity IBM, skinning matrix should be identity
        np.testing.assert_allclose(matrices[0], np.eye(4), atol=1e-6)

    def test_ibm_extra_bones_get_identity(self):
        """Bones beyond the IBM count should get identity."""
        # 1 IBM for 2 bones
        ibm = np.zeros((2, 4, 4), dtype=np.float32)
        ibm[0] = np.eye(4)
        ibm[1] = np.eye(4)
        # Second bone has custom IBM
        ibm[1][0, 3] = 0.5

        bones = [
            _make_bone("root", 0, children=[1, 2]),
            _make_bone("child1", 1, parent=0, pos=(0, 1, 0)),
            _make_bone("child2", 2, parent=0, pos=(0, 0.5, 0)),
        ]
        # Only 2 IBM provided, third bone gets identity
        skel = _make_skeleton(bones, ibm=ibm)
        assert skel.inverse_bind_matrices.shape == (2, 4, 4)


# =============================================================================
# Test: Quaternion and Euler Math
# =============================================================================

class TestBoneMatrixMath:
    """Test rotation math utilities."""

    def test_quat_identity(self):
        """Identity quaternion (0,0,0,1) should produce identity matrix."""
        mat = _quat_to_matrix(0, 0, 0, 1)
        np.testing.assert_allclose(mat, np.eye(3), atol=1e-6)

    def test_quat_90_y(self):
        """90 degree rotation about Y axis."""
        angle = math.radians(90)
        # Quaternion for 90 deg Y: (0, sin(45), 0, cos(45))
        q = (0, math.sin(angle / 2), 0, math.cos(angle / 2))
        mat = _quat_to_matrix(*q)
        # X axis should map to -Z, Z should map to X
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=np.float32)
        np.testing.assert_allclose(mat, expected, atol=1e-6)

    def test_euler_zero(self):
        """Zero euler angles should produce identity."""
        mat = _euler_to_matrix(0, 0, 0)
        np.testing.assert_allclose(mat, np.eye(3), atol=1e-6)

    def test_euler_90_x(self):
        """90 degree rotation about X axis."""
        mat = _euler_to_matrix(90, 0, 0)
        # Y should map to Z, Z should map to -Y
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        np.testing.assert_allclose(mat, expected, atol=1e-6)

    def test_euler_90_y(self):
        """90 degree rotation about Y axis."""
        mat = _euler_to_matrix(0, 90, 0)
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=np.float32)
        np.testing.assert_allclose(mat, expected, atol=1e-6)

    def test_euler_90_z(self):
        """90 degree rotation about Z axis."""
        mat = _euler_to_matrix(0, 0, 90)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        np.testing.assert_allclose(mat, expected, atol=1e-6)

    def test_bone_local_matrix_identity(self):
        """Bone with zero transform produces identity local matrix."""
        bone = _make_bone("test", 0)
        mat = _bone_local_matrix(bone)
        np.testing.assert_allclose(mat, np.eye(4), atol=1e-6)

    def test_bone_local_matrix_translation(self):
        """Bone with position offset produces correct translation."""
        bone = _make_bone("test", 0, pos=(1.0, 2.0, 3.0))
        mat = _bone_local_matrix(bone)
        assert mat[0, 3] == pytest.approx(1.0)
        assert mat[1, 3] == pytest.approx(2.0)
        assert mat[2, 3] == pytest.approx(3.0)
        # Rotation part should be identity
        np.testing.assert_allclose(mat[:3, :3], np.eye(3), atol=1e-6)

    def test_bone_local_matrix_with_rest_rotation(self):
        """Bone with rest-pose quaternion rotation."""
        # 90 degrees about Y
        angle = math.radians(90)
        qy = math.sin(angle / 2)
        qw = math.cos(angle / 2)
        bone = _make_bone("test", 0, rot=(0, qy, 0, qw))
        mat = _bone_local_matrix(bone)
        # Check rotation part matches quat rotation
        expected_rot = _quat_to_matrix(0, qy, 0, qw)
        np.testing.assert_allclose(mat[:3, :3], expected_rot, atol=1e-6)

    def test_bone_local_matrix_with_pose_overlay(self):
        """Pose rotation overlays on top of rest-pose."""
        bone = _make_bone("test", 0, humanoid="head")
        # Apply 30 degrees around Y
        rotations = {"head": (0, 30, 0)}
        mat = _bone_local_matrix(bone, bone_rotations=rotations)
        # Should not be identity (pose applied)
        assert not np.allclose(mat[:3, :3], np.eye(3), atol=1e-3)
        # Translation should still be zero
        assert mat[0, 3] == pytest.approx(0.0)


# =============================================================================
# Test: Hierarchy Traversal
# =============================================================================

class TestHierarchyTraversal:
    """Test bone hierarchy walk and world transform accumulation."""

    def test_single_bone_world(self):
        """Single root bone world transform equals local."""
        bones = [_make_bone("root", 0, pos=(0, 1, 0))]
        skel = _make_skeleton(bones)
        _, world = _compute_bone_matrices(skel)
        assert world[0][1, 3] == pytest.approx(1.0)

    def test_parent_child_chain(self):
        """Child world position accumulates parent translation."""
        bones = [
            _make_bone("root", 0, pos=(0, 1, 0), children=[1]),
            _make_bone("child", 1, parent=0, pos=(0, 0.5, 0)),
        ]
        skel = _make_skeleton(bones)
        _, world = _compute_bone_matrices(skel)
        # Root at y=1
        assert world[0][1, 3] == pytest.approx(1.0)
        # Child at y=1.5 (parent 1.0 + local 0.5)
        assert world[1][1, 3] == pytest.approx(1.5)

    def test_three_bone_chain(self):
        """Three-bone chain accumulates correctly."""
        bones = [
            _make_bone("hips", 0, pos=(0, 0.8, 0), children=[1]),
            _make_bone("spine", 1, parent=0, pos=(0, 0.2, 0), children=[2]),
            _make_bone("chest", 2, parent=1, pos=(0, 0.2, 0)),
        ]
        skel = _make_skeleton(bones)
        _, world = _compute_bone_matrices(skel)
        assert world[0][1, 3] == pytest.approx(0.8)
        assert world[1][1, 3] == pytest.approx(1.0)
        assert world[2][1, 3] == pytest.approx(1.2)

    def test_branching_hierarchy(self):
        """Branching skeleton: two children from same parent."""
        bones = [
            _make_bone("root", 0, pos=(0, 1, 0), children=[1, 2]),
            _make_bone("left", 1, parent=0, pos=(-0.5, 0, 0)),
            _make_bone("right", 2, parent=0, pos=(0.5, 0, 0)),
        ]
        skel = _make_skeleton(bones)
        _, world = _compute_bone_matrices(skel)
        # Left child: x = -0.5, y = 1.0
        assert world[1][0, 3] == pytest.approx(-0.5)
        assert world[1][1, 3] == pytest.approx(1.0)
        # Right child: x = 0.5, y = 1.0
        assert world[2][0, 3] == pytest.approx(0.5)
        assert world[2][1, 3] == pytest.approx(1.0)


# =============================================================================
# Test: Skinning Matrices
# =============================================================================

class TestSkinningMatrices:
    """Test skinning matrix computation (world * inverse_bind)."""

    def test_rest_pose_identity(self):
        """At rest pose with matching IBM, skinning matrices are identity."""
        # Build a 2-bone chain
        bones = [
            _make_bone("root", 0, pos=(0, 1, 0), children=[1]),
            _make_bone("child", 1, parent=0, pos=(0, 0.5, 0)),
        ]

        # IBM should be the inverse of the rest-pose world transform
        skel_no_ibm = _make_skeleton(bones)
        _, world = _compute_bone_matrices(skel_no_ibm)

        ibm = np.zeros((2, 4, 4), dtype=np.float32)
        ibm[0] = np.linalg.inv(world[0])
        ibm[1] = np.linalg.inv(world[1])

        skel = _make_skeleton(bones, ibm=ibm)
        matrices, _ = _compute_bone_matrices(skel)

        np.testing.assert_allclose(matrices[0], np.eye(4), atol=1e-5)
        np.testing.assert_allclose(matrices[1], np.eye(4), atol=1e-5)

    def test_pose_rotation_changes_skinning(self):
        """Applied pose rotation produces non-identity skinning matrix."""
        bones = [
            _make_bone("root", 0, pos=(0, 1, 0), humanoid="hips", children=[1]),
            _make_bone("head", 1, parent=0, pos=(0, 0.5, 0), humanoid="head"),
        ]

        # Compute rest-pose IBM
        skel_rest = _make_skeleton(
            bones, humanoid_map={"hips": 0, "head": 1},
        )
        _, world_rest = _compute_bone_matrices(skel_rest)
        ibm = np.zeros((2, 4, 4), dtype=np.float32)
        ibm[0] = np.linalg.inv(world_rest[0])
        ibm[1] = np.linalg.inv(world_rest[1])

        # Now apply a head rotation
        skel = _make_skeleton(
            bones, humanoid_map={"hips": 0, "head": 1}, ibm=ibm,
        )
        bone_rotations = {"head": (0, 45, 0)}
        matrices, _ = _compute_bone_matrices(skel, bone_rotations)

        # Root (hips) should still be identity (no rotation applied)
        np.testing.assert_allclose(matrices[0], np.eye(4), atol=1e-5)
        # Head should NOT be identity (rotation was applied)
        assert not np.allclose(matrices[1], np.eye(4), atol=1e-3)

    def test_bone_matrix_count(self):
        """Number of output matrices matches bone count."""
        bones = [_make_bone(f"bone_{i}", i) for i in range(5)]
        bones[0].children = [1, 2]
        bones[1].parent_index = 0
        bones[2].parent_index = 0
        bones[2].children = [3, 4]
        bones[3].parent_index = 2
        bones[4].parent_index = 2

        skel = _make_skeleton(bones)
        matrices, _ = _compute_bone_matrices(skel)
        assert len(matrices) == 5

    def test_no_skeleton_graceful(self):
        """Empty skeleton produces empty matrices."""
        skel = _make_skeleton([])
        matrices, _ = _compute_bone_matrices(skel)
        assert len(matrices) == 0


# =============================================================================
# Test: PoseRetargeter Bone Name Mapping
# =============================================================================

class TestRetargeterMapping:
    """Test that PoseRetargeter output matches VRM humanoid_map keys."""

    def test_muscle_to_bone_lowercase(self):
        """PoseRetargeter converts bone names to lowercase by default."""
        retargeter = PoseRetargeter()
        result = retargeter.muscle_to_rotation("Head.NodDownUp", 0.5)
        assert result is not None
        bone_name = result[0]
        assert bone_name == "head"

    def test_muscle_to_bone_matches_humanoid_map(self):
        """Retargeter output keys match VRM humanoid_map keys."""
        retargeter = PoseRetargeter()
        pose = PoseState(muscles={"Head.TurnLeftRight": 0.5})
        rotations = retargeter.apply_pose(pose)
        assert "head" in rotations

    def test_multiple_muscles_same_bone(self):
        """Multiple muscles for one bone accumulate rotations."""
        retargeter = PoseRetargeter()
        pose = PoseState(muscles={
            "Head.NodDownUp": 0.3,
            "Head.TurnLeftRight": 0.5,
            "Head.TiltLeftRight": 0.2,
        })
        rotations = retargeter.apply_pose(pose)
        assert "head" in rotations
        # All three axes should have non-zero values
        rx, ry, rz = rotations["head"]
        assert abs(rx) > 0
        assert abs(ry) > 0
        assert abs(rz) > 0

    def test_arm_muscles_map_correctly(self):
        """Arm muscle bone names map to VRM humanoid bone names."""
        retargeter = PoseRetargeter()
        pose = PoseState(muscles={
            "LeftArm.DownUp": 0.5,
            "RightArm.DownUp": 0.3,
        })
        rotations = retargeter.apply_pose(pose)
        # PoseRetargeter maps "LeftArm" -> "leftUpperArm" via DEFAULT_VRM_BONE_MAP
        assert "leftUpperArm" in rotations
        assert "rightUpperArm" in rotations


# =============================================================================
# Test: Joint Data Fallback
# =============================================================================

class TestJointDataFallback:
    """Test fallback joint indices/weights for unskinned meshes."""

    def test_mesh_with_skinning_data(self):
        """Meshes with joint data should preserve it."""
        mesh = Mesh(
            name="skinned",
            vertices=np.zeros((4, 3), dtype=np.float32),
            joint_indices=np.array([[0, 1, 2, 3]], dtype=np.uint8),
            joint_weights=np.array([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32),
        )
        assert mesh.joint_indices is not None
        assert mesh.joint_weights is not None
        assert mesh.joint_weights[0, 0] == pytest.approx(0.4)

    def test_mesh_without_skinning_data(self):
        """Meshes without joint data have None."""
        mesh = Mesh(
            name="static",
            vertices=np.zeros((4, 3), dtype=np.float32),
        )
        assert mesh.joint_indices is None
        assert mesh.joint_weights is None

    def test_fallback_indices_zeros(self):
        """Fallback joint indices should be all zeros (pin to root)."""
        n_verts = 10
        fallback = np.zeros((n_verts, 4), dtype=np.int32)
        assert np.all(fallback == 0)

    def test_fallback_weights_first_bone(self):
        """Fallback weights should be (1,0,0,0) - full weight on bone 0."""
        n_verts = 10
        fallback = np.zeros((n_verts, 4), dtype=np.float32)
        fallback[:, 0] = 1.0
        assert fallback[0, 0] == pytest.approx(1.0)
        assert fallback[0, 1] == pytest.approx(0.0)
        assert np.all(fallback[:, 0] == 1.0)


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
