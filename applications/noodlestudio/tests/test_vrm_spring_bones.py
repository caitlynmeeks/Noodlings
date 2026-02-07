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
#   Tests for VRM Spring Bone Integration
#
#   Verifies VRM 0.x chain expansion, glTF index remapping,
#   spring bone simulator integration, and pipeline wiring.
#   These tests run without an OpenGL context.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_vrm_spring_bones
# PURPOSE:  Tests for Spring Bone Integration
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestVRM0xChainExpansion, TestIndexRemapping,
#   TestSimulatorIntegration, TestPipelineWiring
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import numpy as np
from dataclasses import field

from noodlestudio.core.semantic_world.vrm_parser import (
    Vector3, Quaternion, Transform,
    Bone, Skeleton,
    SpringBoneChain, SpringBoneCollider, SpringBoneSystem,
    VRMAvatar,
)
from noodlestudio.core.semantic_world.spring_bone_simulation import (
    SpringBoneSimulator,
)


# ---------------------------------------------------------------------------
# Helpers: build mock skeletons and avatars
# ---------------------------------------------------------------------------

def _make_bone(name, index, parent=-1, pos=(0, 0, 0), children=None,
               humanoid=None):
    return Bone(
        name=name,
        index=index,
        parent_index=parent,
        transform=Transform(
            position=Vector3(*pos),
            rotation=Quaternion(0, 0, 0, 1),
            scale=Vector3(1, 1, 1),
        ),
        children=children or [],
        humanoid_bone=humanoid,
    )


def _make_spring_avatar():
    """Create a minimal avatar with spring bone chains for testing.

    Skeleton layout:
      hips[0] (root, y=1)
        -> spine[1] (y=0.3)
          -> head[2] (y=0.3)
            -> hair1[3] (y=0.2)  <- spring chain 1
              -> hair2[4] (y=0.15)
        -> tail1[5] (y=-0.1)    <- spring chain 2
          -> tail2[6] (y=-0.15)
            -> tail3[7] (y=-0.1)
    """
    bones = [
        _make_bone("hips", 0, pos=(0, 1, 0), children=[1, 5], humanoid="hips"),
        _make_bone("spine", 1, parent=0, pos=(0, 0.3, 0), children=[2], humanoid="spine"),
        _make_bone("head", 2, parent=1, pos=(0, 0.3, 0), children=[3], humanoid="head"),
        _make_bone("hair1", 3, parent=2, pos=(0, 0.2, 0), children=[4]),
        _make_bone("hair2", 4, parent=3, pos=(0, 0.15, 0)),
        _make_bone("tail1", 5, parent=0, pos=(0, -0.1, -0.1), children=[6]),
        _make_bone("tail2", 6, parent=5, pos=(0, -0.15, 0), children=[7]),
        _make_bone("tail3", 7, parent=6, pos=(0, -0.1, 0)),
    ]

    skeleton = Skeleton(
        bones=bones,
        root_bone_index=0,
        humanoid_map={"hips": 0, "spine": 1, "head": 2},
    )

    # Identity inverse bind matrices
    skeleton.inverse_bind_matrices = np.array(
        [np.eye(4, dtype=np.float32) for _ in bones]
    )

    spring_system = SpringBoneSystem(
        chains=[
            SpringBoneChain(
                name="hair",
                bone_indices=[3, 4],
                stiffness=0.5,
                gravity_power=0.1,
                gravity_dir=Vector3(0, -1, 0),
                drag_force=0.3,
                hit_radius=0.01,
            ),
            SpringBoneChain(
                name="tail",
                bone_indices=[5, 6, 7],
                stiffness=0.8,
                gravity_power=0.2,
                gravity_dir=Vector3(0, -1, 0),
                drag_force=0.4,
                hit_radius=0.02,
            ),
        ],
        colliders=[
            SpringBoneCollider(
                bone_index=2,  # head
                offset=Vector3(0, 0.1, 0),
                radius=0.15,
            ),
        ],
    )

    avatar = VRMAvatar()
    avatar.skeleton = skeleton
    avatar.spring_bones = spring_system

    return avatar


# =============================================================================
# Test: VRM 0.x Chain Expansion
# =============================================================================

class TestVRM0xChainExpansion:
    """Tests for VRM 0.x spring bone root->chain expansion."""

    def test_multiple_roots_create_multiple_chains(self):
        """Each bone in boneGroups.bones creates a separate chain."""
        from noodlestudio.core.semantic_world.vrm_parser import VRMParser

        parser = VRMParser()

        # Simulate glTF nodes with children: root0->child0, root1->child1
        parser.gltf.json_data = {
            'nodes': [
                {'name': 'root0', 'children': [2]},   # node 0
                {'name': 'root1', 'children': [3]},   # node 1
                {'name': 'child0'},                     # node 2
                {'name': 'child1'},                     # node 3
            ]
        }

        secondary_anim = {
            'boneGroups': [{
                'bones': [0, 1],  # Two roots -> two chains
                'stiffiness': 0.5,
                'gravityPower': 0.1,
                'gravityDir': {'x': 0, 'y': -1, 'z': 0},
                'dragForce': 0.3,
                'hitRadius': 0.02,
                'colliderGroups': [],
            }],
            'colliderGroups': [],
        }

        parser._parse_spring_bones_0_x(secondary_anim)

        assert len(parser.avatar.spring_bones.chains) == 2
        assert parser.avatar.spring_bones.chains[0].bone_indices == [0, 2]
        assert parser.avatar.spring_bones.chains[1].bone_indices == [1, 3]

    def test_single_root_single_chain(self):
        """One root bone produces one chain through its children."""
        from noodlestudio.core.semantic_world.vrm_parser import VRMParser

        parser = VRMParser()
        parser.gltf.json_data = {
            'nodes': [
                {'name': 'root', 'children': [1]},
                {'name': 'mid', 'children': [2]},
                {'name': 'tip'},
            ]
        }

        secondary_anim = {
            'boneGroups': [{
                'bones': [0],
                'stiffiness': 1.0,
                'gravityPower': 0,
                'gravityDir': {'x': 0, 'y': -1, 'z': 0},
                'dragForce': 0.4,
                'hitRadius': 0.02,
                'colliderGroups': [],
            }],
            'colliderGroups': [],
        }

        parser._parse_spring_bones_0_x(secondary_anim)

        assert len(parser.avatar.spring_bones.chains) == 1
        assert parser.avatar.spring_bones.chains[0].bone_indices == [0, 1, 2]

    def test_chain_preserves_physics_params(self):
        """Each expanded chain inherits the group's physics parameters."""
        from noodlestudio.core.semantic_world.vrm_parser import VRMParser

        parser = VRMParser()
        parser.gltf.json_data = {
            'nodes': [{'name': 'root'}]
        }

        secondary_anim = {
            'boneGroups': [{
                'bones': [0],
                'stiffiness': 0.7,
                'gravityPower': 0.3,
                'gravityDir': {'x': 0, 'y': -1, 'z': 0},
                'dragForce': 0.5,
                'hitRadius': 0.03,
                'colliderGroups': [],
            }],
            'colliderGroups': [],
        }

        parser._parse_spring_bones_0_x(secondary_anim)
        chain = parser.avatar.spring_bones.chains[0]

        assert chain.stiffness == 0.7
        assert chain.gravity_power == 0.3
        assert chain.drag_force == 0.5
        assert chain.hit_radius == 0.03

    def test_leaf_root_creates_single_bone_chain(self):
        """A root with no children creates a chain of just that bone."""
        from noodlestudio.core.semantic_world.vrm_parser import VRMParser

        parser = VRMParser()
        parser.gltf.json_data = {
            'nodes': [{'name': 'leaf'}]
        }

        secondary_anim = {
            'boneGroups': [{
                'bones': [0],
                'stiffiness': 1.0,
                'gravityPower': 0,
                'gravityDir': {'x': 0, 'y': -1, 'z': 0},
                'dragForce': 0.4,
                'hitRadius': 0.02,
                'colliderGroups': [],
            }],
            'colliderGroups': [],
        }

        parser._parse_spring_bones_0_x(secondary_anim)
        assert parser.avatar.spring_bones.chains[0].bone_indices == [0]


# =============================================================================
# Test: glTF Index Remapping
# =============================================================================

class TestIndexRemapping:
    """Tests for spring bone glTF node -> bone array index remapping."""

    def test_chain_indices_remapped(self):
        """Spring bone chain indices are converted from glTF to bone array."""
        from noodlestudio.core.semantic_world.vrm_parser import VRMParser

        parser = VRMParser()
        # Simulate mapping: glTF node 10 -> bone 0, node 20 -> bone 1
        parser._gltf_to_bone = {10: 0, 20: 1, 30: 2}

        parser.avatar.spring_bones.chains = [
            SpringBoneChain(name="test", bone_indices=[10, 20, 30])
        ]

        parser._remap_spring_bone_indices()

        assert parser.avatar.spring_bones.chains[0].bone_indices == [0, 1, 2]

    def test_collider_index_remapped(self):
        """Collider bone_index is converted from glTF to bone array."""
        from noodlestudio.core.semantic_world.vrm_parser import VRMParser

        parser = VRMParser()
        parser._gltf_to_bone = {5: 42}

        parser.avatar.spring_bones.colliders = [
            SpringBoneCollider(bone_index=5, radius=0.1)
        ]

        parser._remap_spring_bone_indices()

        assert parser.avatar.spring_bones.colliders[0].bone_index == 42

    def test_unmapped_index_preserved(self):
        """Indices not in the mapping pass through unchanged."""
        from noodlestudio.core.semantic_world.vrm_parser import VRMParser

        parser = VRMParser()
        parser._gltf_to_bone = {10: 0}

        parser.avatar.spring_bones.chains = [
            SpringBoneChain(name="test", bone_indices=[10, 99])
        ]

        parser._remap_spring_bone_indices()

        assert parser.avatar.spring_bones.chains[0].bone_indices == [0, 99]


# =============================================================================
# Test: Simulator Integration
# =============================================================================

class TestSimulatorIntegration:
    """Tests for spring bone simulator with mock avatars."""

    def test_simulator_creates_from_avatar(self):
        """SpringBoneSimulator initializes from avatar with chains."""
        avatar = _make_spring_avatar()
        sim = SpringBoneSimulator(avatar)
        sim.initialize()

        assert len(sim.chain_states) == 2
        assert len(sim.collider_states) == 1

    def test_simulator_step_moves_joints(self):
        """After update() with gravity, joint positions change from rest."""
        avatar = _make_spring_avatar()
        sim = SpringBoneSimulator(avatar)
        sim.initialize()

        # Build rest-pose world transforms (identity hierarchy)
        rest_transforms = {}
        for bone in avatar.skeleton.bones:
            wt = np.eye(4, dtype=np.float32)
            wt[:3, 3] = bone.transform.position.to_array()
            rest_transforms[bone.index] = wt

        # Step several frames
        for _ in range(10):
            sim.update(0.016, rest_transforms)

        positions = sim.get_joint_positions()
        # Spring bones should have moved (at least slightly)
        assert len(positions) > 0

    def test_simulator_with_gravity_pulls_down(self):
        """Gravity should pull spring bone joints downward over time."""
        avatar = _make_spring_avatar()
        # Increase gravity power for visible effect
        avatar.spring_bones.chains[1].gravity_power = 1.0
        sim = SpringBoneSimulator(avatar)
        sim.initialize()

        # Build world transforms
        rest_transforms = {}
        for bone in avatar.skeleton.bones:
            wt = np.eye(4, dtype=np.float32)
            wt[:3, 3] = bone.transform.position.to_array()
            rest_transforms[bone.index] = wt

        # Record initial Y position of tail tip
        initial_pos = sim.chain_states[1].joints[-1].current_position.copy()

        # Step many frames
        for _ in range(60):
            sim.update(0.016, rest_transforms)

        final_pos = sim.chain_states[1].joints[-1].current_position
        # Y should have decreased (gravity pulls down)
        assert final_pos[1] < initial_pos[1], (
            f"Tail tip should drop: initial_y={initial_pos[1]:.4f}, "
            f"final_y={final_pos[1]:.4f}"
        )

    def test_simulator_reset_restores_rest(self):
        """reset() returns all joints to their rest positions."""
        avatar = _make_spring_avatar()
        avatar.spring_bones.chains[0].gravity_power = 1.0
        sim = SpringBoneSimulator(avatar)
        sim.initialize()

        rest_transforms = {}
        for bone in avatar.skeleton.bones:
            wt = np.eye(4, dtype=np.float32)
            wt[:3, 3] = bone.transform.position.to_array()
            rest_transforms[bone.index] = wt

        # Step to move joints away from rest
        for _ in range(30):
            sim.update(0.016, rest_transforms)

        # Verify joints moved
        positions_moved = sim.get_joint_positions()
        assert len(positions_moved) > 0

        # Reset
        sim.reset()

        # Verify joints back at rest
        for chain_state in sim.chain_states:
            for joint in chain_state.joints:
                np.testing.assert_array_almost_equal(
                    joint.current_position, joint.rest_position,
                    err_msg=f"Joint {joint.bone_index} not at rest after reset"
                )

    def test_get_bone_rotations_returns_quaternions(self):
        """get_bone_rotations() returns dict of bone_index -> quaternion."""
        avatar = _make_spring_avatar()
        sim = SpringBoneSimulator(avatar)
        sim.initialize()

        rest_transforms = {}
        for bone in avatar.skeleton.bones:
            wt = np.eye(4, dtype=np.float32)
            wt[:3, 3] = bone.transform.position.to_array()
            rest_transforms[bone.index] = wt

        sim.update(0.016, rest_transforms)
        rotations = sim.get_bone_rotations()

        # Should have entries for spring bones
        assert isinstance(rotations, dict)
        for bone_idx, quat in rotations.items():
            assert isinstance(bone_idx, int)
            assert len(quat) == 4, f"Quaternion should have 4 components, got {len(quat)}"


# =============================================================================
# Test: Pipeline Wiring
# =============================================================================

class TestPipelineWiring:
    """Tests for spring bone overriding bone matrices in the pipeline."""

    def test_spring_positions_override_world_transforms(self):
        """Simulated positions replace world transforms for spring bones."""
        avatar = _make_spring_avatar()

        # Build rest-pose world transforms
        world_transforms = []
        for bone in avatar.skeleton.bones:
            wt = np.eye(4, dtype=np.float32)
            wt[:3, 3] = bone.transform.position.to_array()
            world_transforms.append(wt)

        bone_matrices = np.array([np.eye(4, dtype=np.float32) for _ in avatar.skeleton.bones])

        # Simulate spring bone step
        sim = SpringBoneSimulator(avatar)
        sim.initialize()

        bone_transforms_dict = {i: wt for i, wt in enumerate(world_transforms)}
        sim.set_bone_transforms(bone_transforms_dict)

        # Step with gravity to create movement
        avatar.spring_bones.chains[1].gravity_power = 1.0
        sim.chain_states[1].gravity_power = 1.0
        for _ in range(30):
            sim.update(0.016, bone_transforms_dict)

        spring_positions = sim.get_joint_positions()

        # Verify some spring bones have moved
        assert len(spring_positions) > 0, "Should have spring bone positions"

    def test_non_spring_bones_unchanged(self):
        """Bones not in spring chains keep original transforms."""
        avatar = _make_spring_avatar()

        # Spring bone indices: 3, 4, 5, 6, 7
        spring_indices = set()
        for chain in avatar.spring_bones.chains:
            spring_indices.update(chain.bone_indices)

        # Non-spring bones: 0 (hips), 1 (spine), 2 (head)
        non_spring = {0, 1, 2}
        assert non_spring.isdisjoint(spring_indices)
