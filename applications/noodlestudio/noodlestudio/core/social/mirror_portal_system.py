"""
Mirror and Portal Rendering System

Gaussians make mirrors and portals trivial - just render from a different camera!
No stencil buffers, no complex scene traversal. Pure elegance.

VRChat users congregate around mirrors. This is our secret weapon.

Author: Caitlyn + Claude
Date: December 2025
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Math Utilities
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    length = np.linalg.norm(v)
    if length < 1e-10:
        return np.zeros_like(v)
    return v / length


def reflect_vector(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect vector v across plane with given normal."""
    normal = normalize(normal)
    return v - 2 * np.dot(v, normal) * normal


def reflect_point(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Reflect a point across a plane."""
    normal = normalize(plane_normal)
    d = np.dot(point - plane_point, normal)
    return point - 2 * d * normal


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create quaternion from axis-angle (returns x, y, z, w)."""
    axis = normalize(axis)
    half_angle = angle * 0.5
    s = math.sin(half_angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half_angle)])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion."""
    qx, qy, qz, qw = q
    # q * v * q^-1
    cx = qw * v[0] + qy * v[2] - qz * v[1]
    cy = qw * v[1] + qz * v[0] - qx * v[2]
    cz = qw * v[2] + qx * v[1] - qy * v[0]
    cw = -qx * v[0] - qy * v[1] - qz * v[2]
    return np.array([
        cx * qw + cw * -qx + cy * -qz - cz * -qy,
        cy * qw + cw * -qy + cz * -qx - cx * -qz,
        cz * qw + cw * -qz + cx * -qy - cy * -qx,
    ])


# =============================================================================
# Camera
# =============================================================================

@dataclass
class Camera:
    """Virtual camera for rendering."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # Quaternion
    fov: float = 60.0  # Degrees
    aspect: float = 16.0 / 9.0
    near: float = 0.1
    far: float = 1000.0

    @property
    def forward(self) -> np.ndarray:
        """Get forward direction (negative Z in camera space)."""
        return rotate_vector(np.array([0, 0, -1]), self.rotation)

    @property
    def up(self) -> np.ndarray:
        """Get up direction."""
        return rotate_vector(np.array([0, 1, 0]), self.rotation)

    @property
    def right(self) -> np.ndarray:
        """Get right direction."""
        return rotate_vector(np.array([1, 0, 0]), self.rotation)

    def look_at(self, target: np.ndarray, up: np.ndarray = None):
        """Orient camera to look at target."""
        if up is None:
            up = np.array([0, 1, 0])

        forward = normalize(target - self.position)
        right = normalize(np.cross(forward, up))
        up = np.cross(right, forward)

        # Convert to quaternion
        trace = right[0] + up[1] - forward[2] + 1
        if trace > 0.0001:
            s = 0.5 / math.sqrt(trace)
            w = 0.25 / s
            x = (up[2] + forward[1]) * s
            y = (forward[0] + right[2]) * s
            z = (right[1] - up[0]) * s
        else:
            # Handle edge cases
            if right[0] > up[1] and right[0] > -forward[2]:
                s = 2.0 * math.sqrt(1.0 + right[0] - up[1] + forward[2])
                w = (up[2] + forward[1]) / s
                x = 0.25 * s
                y = (right[1] + up[0]) / s
                z = (forward[0] + right[2]) / s
            elif up[1] > -forward[2]:
                s = 2.0 * math.sqrt(1.0 + up[1] - right[0] + forward[2])
                w = (forward[0] + right[2]) / s
                x = (right[1] + up[0]) / s
                y = 0.25 * s
                z = (up[2] + forward[1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 - forward[2] - right[0] - up[1])
                w = (right[1] - up[0]) / s
                x = (forward[0] + right[2]) / s
                y = (up[2] + forward[1]) / s
                z = 0.25 * s

        self.rotation = normalize(np.array([x, y, z, w]))

    def get_view_matrix(self) -> np.ndarray:
        """Get 4x4 view matrix."""
        # Rotation matrix from quaternion
        x, y, z, w = self.rotation
        rot = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 0],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 0],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), 0],
            [0, 0, 0, 1]
        ])

        # Translation
        trans = np.eye(4)
        trans[:3, 3] = -self.position

        return rot @ trans


# =============================================================================
# Mirror Surface
# =============================================================================

class MirrorType(Enum):
    FLAT = "flat"           # Simple planar reflection
    CURVED = "curved"       # Curved surface (sample multiple points)
    PORTAL = "portal"       # Shows different location
    TIME_DELAY = "time"     # Shows past frames


@dataclass
class MirrorSurface:
    """
    A reflective surface that renders the scene from a reflected viewpoint.

    With Gaussian splatting, this is trivially efficient - just render
    from the reflected camera position. No stencil buffers needed!
    """
    id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    size: Tuple[float, float] = (2.0, 3.0)  # Width, height in meters
    mirror_type: MirrorType = MirrorType.FLAT
    render_resolution: Tuple[int, int] = (1024, 1024)

    # For portal mirrors
    destination_position: Optional[np.ndarray] = None
    destination_rotation: Optional[np.ndarray] = None

    # For time-delay mirrors
    delay_frames: int = 30  # ~0.5 seconds at 60fps

    # Render state
    render_texture: Optional[any] = None  # Platform-specific texture handle
    last_rendered_frame: int = -1

    def get_reflected_camera(self, viewer_camera: Camera) -> Camera:
        """
        Compute the reflected camera position for rendering.

        This is the magic - Gaussians render from any viewpoint,
        so we just flip the camera across the mirror plane.
        """
        if self.mirror_type == MirrorType.PORTAL and self.destination_position is not None:
            return self._get_portal_camera(viewer_camera)

        # Reflect camera position across mirror plane
        reflected_pos = reflect_point(
            viewer_camera.position,
            self.position,
            self.normal
        )

        # Reflect camera forward direction
        reflected_forward = reflect_vector(viewer_camera.forward, self.normal)

        # Create reflected camera
        reflected_camera = Camera(
            position=reflected_pos,
            fov=viewer_camera.fov,
            aspect=viewer_camera.aspect,
            near=viewer_camera.near,
            far=viewer_camera.far,
        )
        reflected_camera.look_at(reflected_pos + reflected_forward)

        return reflected_camera

    def _get_portal_camera(self, viewer_camera: Camera) -> Camera:
        """Get camera at portal destination."""
        if self.destination_position is None:
            return viewer_camera

        # Transform viewer position relative to this mirror
        local_pos = viewer_camera.position - self.position

        # Project onto mirror plane and compute offset
        # The portal shows what you'd see if you were on the other side
        portal_camera = Camera(
            position=self.destination_position + local_pos,
            rotation=self.destination_rotation if self.destination_rotation is not None else viewer_camera.rotation,
            fov=viewer_camera.fov,
            aspect=viewer_camera.aspect,
        )

        return portal_camera

    def is_visible(self, viewer_position: np.ndarray) -> bool:
        """Check if viewer can see the mirror (facing toward it)."""
        to_viewer = viewer_position - self.position
        return np.dot(to_viewer, self.normal) > 0

    def get_corners(self) -> np.ndarray:
        """Get world-space corner positions of mirror quad."""
        # Compute local axes
        up = np.array([0, 1, 0])
        if abs(np.dot(self.normal, up)) > 0.99:
            up = np.array([1, 0, 0])

        right = normalize(np.cross(up, self.normal))
        up = normalize(np.cross(self.normal, right))

        hw, hh = self.size[0] / 2, self.size[1] / 2

        return np.array([
            self.position - right * hw - up * hh,  # Bottom-left
            self.position + right * hw - up * hh,  # Bottom-right
            self.position + right * hw + up * hh,  # Top-right
            self.position - right * hw + up * hh,  # Top-left
        ])


# =============================================================================
# Portal System
# =============================================================================

@dataclass
class Portal:
    """
    A portal that shows and teleports to another location.

    Portal pairs are linked - looking through A shows B's view,
    walking through A teleports to B.
    """
    id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # Quaternion
    size: Tuple[float, float] = (2.0, 3.0)  # Width, height
    linked_portal: Optional['Portal'] = None
    render_resolution: Tuple[int, int] = (1024, 1024)
    max_recursion_depth: int = 2  # How many portals-in-portals to render

    # Render state
    render_texture: Optional[any] = None

    @property
    def forward(self) -> np.ndarray:
        """Portal's forward direction (what you see through it)."""
        return rotate_vector(np.array([0, 0, -1]), self.rotation)

    @property
    def normal(self) -> np.ndarray:
        """Portal surface normal (facing the viewer)."""
        return rotate_vector(np.array([0, 0, 1]), self.rotation)

    def world_to_local(self, point: np.ndarray) -> np.ndarray:
        """Transform world point to portal's local space."""
        # Inverse rotation
        q = self.rotation.copy()
        q[:3] = -q[:3]  # Conjugate
        local = rotate_vector(point - self.position, q)
        return local

    def local_to_world(self, point: np.ndarray) -> np.ndarray:
        """Transform local point to world space."""
        return self.position + rotate_vector(point, self.rotation)

    def get_destination_camera(self, viewer_camera: Camera) -> Optional[Camera]:
        """
        Get the camera that shows what's through the portal.

        The viewer's position relative to this portal is transformed
        to the linked portal's space, then flipped 180 degrees
        (because you're looking "out" of the destination portal).
        """
        if self.linked_portal is None:
            return None

        # Transform viewer to this portal's local space
        local_pos = self.world_to_local(viewer_camera.position)
        local_forward = rotate_vector(
            viewer_camera.forward,
            np.array([-self.rotation[0], -self.rotation[1], -self.rotation[2], self.rotation[3]])
        )

        # Flip Z (coming out the other side, facing opposite direction)
        local_pos[2] = -local_pos[2]
        local_forward[2] = -local_forward[2]

        # Transform to linked portal's world space
        dest_pos = self.linked_portal.local_to_world(local_pos)
        dest_forward = rotate_vector(local_forward, self.linked_portal.rotation)

        # Create destination camera
        dest_camera = Camera(
            position=dest_pos,
            fov=viewer_camera.fov,
            aspect=viewer_camera.aspect,
        )
        dest_camera.look_at(dest_pos + dest_forward)

        return dest_camera

    def check_crossing(
        self,
        prev_position: np.ndarray,
        curr_position: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Check if something crossed through the portal.

        Returns destination position if crossed, None otherwise.
        """
        if self.linked_portal is None:
            return None

        # Check if crossed the portal plane
        prev_local = self.world_to_local(prev_position)
        curr_local = self.world_to_local(curr_position)

        # Crossed if Z changed sign and within portal bounds
        if prev_local[2] * curr_local[2] > 0:
            return None  # Same side

        # Check if within portal bounds
        hw, hh = self.size[0] / 2, self.size[1] / 2
        cross_point = (prev_local + curr_local) / 2  # Approximate

        if abs(cross_point[0]) > hw or abs(cross_point[1]) > hh:
            return None  # Outside portal bounds

        # Calculate destination position
        # Flip Z and transform to linked portal's space
        dest_local = curr_local.copy()
        dest_local[2] = -dest_local[2]

        return self.linked_portal.local_to_world(dest_local)

    def get_teleport_transform(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        velocity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate position, rotation, and velocity after teleporting.

        Preserves momentum relative to portal orientation.
        """
        if self.linked_portal is None:
            return position, rotation, velocity

        # Transform to local, flip Z, transform to destination
        local_pos = self.world_to_local(position)
        local_pos[2] = -local_pos[2]
        dest_pos = self.linked_portal.local_to_world(local_pos)

        # Transform rotation (compose with portal transform difference)
        # This preserves the relative orientation
        q_inv = np.array([-self.rotation[0], -self.rotation[1], -self.rotation[2], self.rotation[3]])
        local_rot = quaternion_multiply(q_inv, rotation)

        # Flip 180 on Y axis for coming out the other side
        flip = quaternion_from_axis_angle(np.array([0, 1, 0]), math.pi)
        local_rot = quaternion_multiply(flip, local_rot)

        dest_rot = quaternion_multiply(self.linked_portal.rotation, local_rot)

        # Transform velocity
        local_vel = rotate_vector(velocity, q_inv)
        local_vel[2] = -local_vel[2]  # Flip Z
        dest_vel = rotate_vector(local_vel, self.linked_portal.rotation)

        return dest_pos, dest_rot, dest_vel


# =============================================================================
# Render Manager
# =============================================================================

class MirrorPortalManager:
    """
    Manages all mirrors and portals in a scene.

    Coordinates rendering order, handles recursive portals,
    and provides the reflected/destination cameras for the
    Gaussian renderer.
    """

    def __init__(self):
        self.mirrors: Dict[str, MirrorSurface] = {}
        self.portals: Dict[str, Portal] = {}
        self.frame_counter: int = 0

        # Render callback - called with (camera, render_target)
        self.render_callback: Optional[Callable] = None

        # Time-delay buffer for time mirrors
        self.frame_buffer: List[any] = []
        self.max_buffer_frames: int = 60

    def add_mirror(self, mirror: MirrorSurface):
        """Add a mirror to the scene."""
        self.mirrors[mirror.id] = mirror
        logger.info(f"Added mirror: {mirror.id}")

    def add_portal(self, portal: Portal):
        """Add a portal to the scene."""
        self.portals[portal.id] = portal
        logger.info(f"Added portal: {portal.id}")

    def link_portals(self, portal_a_id: str, portal_b_id: str):
        """Link two portals together."""
        if portal_a_id in self.portals and portal_b_id in self.portals:
            self.portals[portal_a_id].linked_portal = self.portals[portal_b_id]
            self.portals[portal_b_id].linked_portal = self.portals[portal_a_id]
            logger.info(f"Linked portals: {portal_a_id} <-> {portal_b_id}")

    def update(self, viewer_camera: Camera):
        """
        Update all mirrors and portals for current frame.

        Renders each visible mirror/portal from reflected viewpoint.
        """
        self.frame_counter += 1

        # Render mirrors
        for mirror_id, mirror in self.mirrors.items():
            if mirror.is_visible(viewer_camera.position):
                self._render_mirror(mirror, viewer_camera)

        # Render portals (may be recursive)
        for portal_id, portal in self.portals.items():
            self._render_portal(portal, viewer_camera, depth=0)

    def _render_mirror(self, mirror: MirrorSurface, viewer_camera: Camera):
        """Render a mirror's view."""
        if mirror.last_rendered_frame == self.frame_counter:
            return  # Already rendered this frame

        reflected_camera = mirror.get_reflected_camera(viewer_camera)

        if self.render_callback:
            self.render_callback(reflected_camera, mirror.render_texture)

        mirror.last_rendered_frame = self.frame_counter

    def _render_portal(self, portal: Portal, viewer_camera: Camera, depth: int):
        """Render a portal's view, possibly recursively."""
        if depth > portal.max_recursion_depth:
            return

        if portal.linked_portal is None:
            return

        dest_camera = portal.get_destination_camera(viewer_camera)
        if dest_camera is None:
            return

        # First, render any portals visible through this portal (recursion)
        if depth < portal.max_recursion_depth:
            for other_id, other_portal in self.portals.items():
                if other_portal.id != portal.id:
                    # Check if other portal is in view from destination camera
                    # (Simplified: always render nested portals)
                    self._render_portal(other_portal, dest_camera, depth + 1)

        # Now render the main portal view
        if self.render_callback:
            # For recursive portals, reduce resolution
            scale = 1.0 / (2 ** depth)
            self.render_callback(dest_camera, portal.render_texture)

    def check_portal_crossings(
        self,
        entity_id: str,
        prev_position: np.ndarray,
        curr_position: np.ndarray,
        rotation: np.ndarray,
        velocity: np.ndarray
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Check if an entity crossed through any portal.

        Returns (portal_id, new_position, new_rotation, new_velocity) if crossed.
        """
        for portal_id, portal in self.portals.items():
            dest_pos = portal.check_crossing(prev_position, curr_position)
            if dest_pos is not None:
                new_pos, new_rot, new_vel = portal.get_teleport_transform(
                    curr_position, rotation, velocity
                )
                logger.info(f"Entity {entity_id} crossed portal {portal_id}")
                return (portal_id, new_pos, new_rot, new_vel)

        return None

    def get_visible_mirrors(self, viewer_position: np.ndarray) -> List[MirrorSurface]:
        """Get all mirrors visible from viewer position."""
        return [m for m in self.mirrors.values() if m.is_visible(viewer_position)]

    def get_render_cameras(self, viewer_camera: Camera) -> List[Tuple[str, Camera]]:
        """
        Get all cameras needed for mirror/portal rendering.

        Returns list of (surface_id, camera) tuples for the Gaussian renderer.
        This is the key integration point - the renderer just needs to
        render from these cameras.
        """
        cameras = []

        # Mirror cameras
        for mirror_id, mirror in self.mirrors.items():
            if mirror.is_visible(viewer_camera.position):
                cameras.append((mirror_id, mirror.get_reflected_camera(viewer_camera)))

        # Portal cameras
        for portal_id, portal in self.portals.items():
            dest_camera = portal.get_destination_camera(viewer_camera)
            if dest_camera:
                cameras.append((portal_id, dest_camera))

        return cameras


# =============================================================================
# Factory Functions
# =============================================================================

def create_mirror(
    id: str,
    position: Tuple[float, float, float],
    normal: Tuple[float, float, float] = (0, 0, 1),
    size: Tuple[float, float] = (2.0, 3.0),
    mirror_type: MirrorType = MirrorType.FLAT,
) -> MirrorSurface:
    """Create a mirror surface."""
    return MirrorSurface(
        id=id,
        position=np.array(position, dtype=np.float32),
        normal=normalize(np.array(normal, dtype=np.float32)),
        size=size,
        mirror_type=mirror_type,
    )


def create_portal_pair(
    id_a: str,
    position_a: Tuple[float, float, float],
    rotation_a: Tuple[float, float, float, float],
    id_b: str,
    position_b: Tuple[float, float, float],
    rotation_b: Tuple[float, float, float, float],
    size: Tuple[float, float] = (2.0, 3.0),
) -> Tuple[Portal, Portal]:
    """Create a linked pair of portals."""
    portal_a = Portal(
        id=id_a,
        position=np.array(position_a, dtype=np.float32),
        rotation=np.array(rotation_a, dtype=np.float32),
        size=size,
    )
    portal_b = Portal(
        id=id_b,
        position=np.array(position_b, dtype=np.float32),
        rotation=np.array(rotation_b, dtype=np.float32),
        size=size,
    )

    portal_a.linked_portal = portal_b
    portal_b.linked_portal = portal_a

    return portal_a, portal_b


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Mirror/Portal System Test")
    print("=" * 40)

    # Create a mirror on the wall
    mirror = create_mirror(
        id="bathroom_mirror",
        position=(0, 1.5, 5),  # On far wall
        normal=(0, 0, -1),     # Facing toward origin
        size=(2, 2),
    )

    # Create viewer camera
    viewer = Camera(position=np.array([0, 1.5, 0]))
    viewer.look_at(np.array([0, 1.5, 5]))  # Looking at mirror

    print(f"Viewer at: {viewer.position}")
    print(f"Mirror at: {mirror.position}, normal: {mirror.normal}")
    print(f"Mirror visible: {mirror.is_visible(viewer.position)}")

    # Get reflected camera
    reflected = mirror.get_reflected_camera(viewer)
    print(f"\nReflected camera:")
    print(f"  Position: {reflected.position}")
    print(f"  Forward: {reflected.forward}")

    # Create portal pair
    portal_a, portal_b = create_portal_pair(
        id_a="portal_room1",
        position_a=(5, 1.5, 0),
        rotation_a=(0, 0, 0, 1),
        id_b="portal_room2",
        position_b=(100, 1.5, 0),
        rotation_b=(0, 1, 0, 0),  # Rotated 180
        size=(2, 3),
    )

    print(f"\nPortal A at: {portal_a.position}")
    print(f"Portal B at: {portal_b.position}")

    # Get destination camera when looking through portal A
    viewer_at_portal = Camera(position=np.array([3, 1.5, 0]))
    viewer_at_portal.look_at(np.array([5, 1.5, 0]))

    dest_cam = portal_a.get_destination_camera(viewer_at_portal)
    if dest_cam:
        print(f"\nDestination camera (through portal A):")
        print(f"  Position: {dest_cam.position}")
        print(f"  Forward: {dest_cam.forward}")

    # Test portal crossing
    crossed = portal_a.check_crossing(
        prev_position=np.array([4.5, 1.5, 0]),
        curr_position=np.array([5.5, 1.5, 0]),
    )
    print(f"\nPortal crossing test: {'CROSSED!' if crossed is not None else 'not crossed'}")
    if crossed is not None:
        print(f"  Destination: {crossed}")

    print("\nTest complete!")
