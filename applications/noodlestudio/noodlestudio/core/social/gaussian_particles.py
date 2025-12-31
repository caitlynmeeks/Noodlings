"""
Gaussian Particle System

Particles rendered as small Gaussian splats - they already have
soft blending built in! Perfect for fire, smoke, sparkles, magic.

This is a key advantage over traditional particle systems that
use billboard quads - Gaussians blend naturally.

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
# Particle Properties
# =============================================================================

class EmitterShape(Enum):
    POINT = "point"         # Emit from single point
    SPHERE = "sphere"       # Random within sphere
    CONE = "cone"           # Cone/jet pattern
    BOX = "box"             # Random within box
    CIRCLE = "circle"       # Emit from circle (good for rings)


class BlendMode(Enum):
    ADDITIVE = "additive"   # Fire, magic, glow
    ALPHA = "alpha"         # Smoke, fog
    MULTIPLY = "multiply"   # Shadows, dark effects


@dataclass
class ColorGradient:
    """Color over lifetime - key/value pairs."""
    keys: List[Tuple[float, np.ndarray]] = field(default_factory=list)

    def sample(self, t: float) -> np.ndarray:
        """Sample color at time t (0-1)."""
        if not self.keys:
            return np.array([1, 1, 1, 1])

        if t <= self.keys[0][0]:
            return self.keys[0][1]
        if t >= self.keys[-1][0]:
            return self.keys[-1][1]

        # Find surrounding keys
        for i in range(len(self.keys) - 1):
            if self.keys[i][0] <= t <= self.keys[i + 1][0]:
                k0, c0 = self.keys[i]
                k1, c1 = self.keys[i + 1]
                blend = (t - k0) / (k1 - k0)
                return c0 * (1 - blend) + c1 * blend

        return self.keys[-1][1]


@dataclass
class Curve:
    """Float value over lifetime."""
    keys: List[Tuple[float, float]] = field(default_factory=list)

    def sample(self, t: float) -> float:
        """Sample value at time t (0-1)."""
        if not self.keys:
            return 1.0

        if t <= self.keys[0][0]:
            return self.keys[0][1]
        if t >= self.keys[-1][0]:
            return self.keys[-1][1]

        for i in range(len(self.keys) - 1):
            if self.keys[i][0] <= t <= self.keys[i + 1][0]:
                k0, v0 = self.keys[i]
                k1, v1 = self.keys[i + 1]
                blend = (t - k0) / (k1 - k0)
                return v0 * (1 - blend) + v1 * blend

        return self.keys[-1][1]


# =============================================================================
# Particle
# =============================================================================

@dataclass
class GaussianParticle:
    """
    A single particle represented as a Gaussian splat.

    Each particle has position, velocity, lifetime, and appearance
    that can change over time.
    """
    # Physics
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Lifetime
    age: float = 0.0
    lifetime: float = 1.0
    is_alive: bool = True

    # Appearance (Gaussian properties)
    color: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 1]))  # RGBA
    scale: float = 0.1  # Gaussian radius
    rotation: float = 0.0  # For elongated particles

    # Original spawn values (for lifetime curves)
    spawn_scale: float = 0.1
    spawn_color: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 1]))

    @property
    def normalized_age(self) -> float:
        """Age as 0-1 fraction of lifetime."""
        if self.lifetime <= 0:
            return 1.0
        return min(1.0, self.age / self.lifetime)

    def to_gaussian_data(self) -> Dict:
        """
        Export as Gaussian splat data for rendering.

        Returns dict compatible with Gaussian renderer.
        """
        return {
            "position": self.position.tolist(),
            "scale": [self.scale, self.scale, self.scale],
            "color": self.color[:3].tolist(),
            "opacity": float(self.color[3]),
            "rotation": [0, 0, math.sin(self.rotation/2), math.cos(self.rotation/2)],
        }


# =============================================================================
# Particle Emitter
# =============================================================================

@dataclass
class ParticleEmitter:
    """
    Spawns and manages particles.

    Configurable emission rate, shape, initial velocity, etc.
    """
    id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # Quaternion

    # Emission
    emission_rate: float = 10.0  # Particles per second
    max_particles: int = 1000
    is_emitting: bool = True

    # Spawn shape
    emitter_shape: EmitterShape = EmitterShape.POINT
    shape_radius: float = 0.5
    shape_size: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1]))
    cone_angle: float = 30.0  # Degrees

    # Initial velocity
    initial_speed: Tuple[float, float] = (1.0, 2.0)  # Min, max
    speed_spread: float = 0.5  # Random spread

    # Lifetime
    lifetime: Tuple[float, float] = (1.0, 2.0)  # Min, max seconds

    # Physics
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, -9.8, 0]))
    drag: float = 0.1

    # Appearance over lifetime
    color_over_lifetime: Optional[ColorGradient] = None
    scale_over_lifetime: Optional[Curve] = None
    opacity_over_lifetime: Optional[Curve] = None

    # Initial appearance
    start_color: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 1]))
    start_scale: Tuple[float, float] = (0.05, 0.1)

    # Blend mode
    blend_mode: BlendMode = BlendMode.ADDITIVE

    # Runtime state
    particles: List[GaussianParticle] = field(default_factory=list)
    _emission_accumulator: float = 0.0

    def update(self, dt: float):
        """
        Update all particles and emit new ones.

        Args:
            dt: Delta time in seconds
        """
        # Update existing particles
        for particle in self.particles:
            if not particle.is_alive:
                continue

            # Physics
            particle.velocity += self.gravity * dt
            particle.velocity *= (1.0 - self.drag * dt)
            particle.position += particle.velocity * dt

            # Age
            particle.age += dt
            if particle.age >= particle.lifetime:
                particle.is_alive = False
                continue

            # Update appearance over lifetime
            t = particle.normalized_age
            self._update_particle_appearance(particle, t)

        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive]

        # Emit new particles
        if self.is_emitting:
            self._emit_particles(dt)

    def _emit_particles(self, dt: float):
        """Emit new particles based on emission rate."""
        self._emission_accumulator += self.emission_rate * dt

        while self._emission_accumulator >= 1.0 and len(self.particles) < self.max_particles:
            self._emission_accumulator -= 1.0
            particle = self._spawn_particle()
            self.particles.append(particle)

    def _spawn_particle(self) -> GaussianParticle:
        """Create a new particle with randomized initial state."""
        # Random position within emitter shape
        position = self._get_spawn_position()

        # Random velocity
        velocity = self._get_spawn_velocity()

        # Random lifetime
        lifetime = np.random.uniform(self.lifetime[0], self.lifetime[1])

        # Random scale
        scale = np.random.uniform(self.start_scale[0], self.start_scale[1])

        particle = GaussianParticle(
            position=position,
            velocity=velocity,
            lifetime=lifetime,
            scale=scale,
            spawn_scale=scale,
            color=self.start_color.copy(),
            spawn_color=self.start_color.copy(),
        )

        return particle

    def _get_spawn_position(self) -> np.ndarray:
        """Get random spawn position based on emitter shape."""
        if self.emitter_shape == EmitterShape.POINT:
            return self.position.copy()

        elif self.emitter_shape == EmitterShape.SPHERE:
            # Random point in sphere
            phi = np.random.uniform(0, 2 * math.pi)
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = math.sqrt(1 - cos_theta**2)
            r = self.shape_radius * np.random.uniform(0, 1) ** (1/3)
            return self.position + np.array([
                r * sin_theta * math.cos(phi),
                r * cos_theta,
                r * sin_theta * math.sin(phi),
            ])

        elif self.emitter_shape == EmitterShape.BOX:
            return self.position + np.array([
                np.random.uniform(-self.shape_size[0]/2, self.shape_size[0]/2),
                np.random.uniform(-self.shape_size[1]/2, self.shape_size[1]/2),
                np.random.uniform(-self.shape_size[2]/2, self.shape_size[2]/2),
            ])

        elif self.emitter_shape == EmitterShape.CIRCLE:
            angle = np.random.uniform(0, 2 * math.pi)
            r = self.shape_radius * np.random.uniform(0, 1) ** 0.5
            return self.position + np.array([
                r * math.cos(angle),
                0,
                r * math.sin(angle),
            ])

        elif self.emitter_shape == EmitterShape.CONE:
            return self.position.copy()

        return self.position.copy()

    def _get_spawn_velocity(self) -> np.ndarray:
        """Get random initial velocity."""
        speed = np.random.uniform(self.initial_speed[0], self.initial_speed[1])

        if self.emitter_shape == EmitterShape.CONE:
            # Cone-shaped emission
            angle = math.radians(self.cone_angle)
            phi = np.random.uniform(0, 2 * math.pi)
            cos_theta = np.random.uniform(math.cos(angle), 1)
            sin_theta = math.sqrt(1 - cos_theta**2)

            # Default up direction, transform by emitter rotation
            local_dir = np.array([
                sin_theta * math.cos(phi),
                cos_theta,
                sin_theta * math.sin(phi),
            ])

            return local_dir * speed

        else:
            # Random spread
            spread = self.speed_spread
            direction = np.array([
                np.random.uniform(-spread, spread),
                1 + np.random.uniform(-spread, spread),
                np.random.uniform(-spread, spread),
            ])
            direction = direction / np.linalg.norm(direction)
            return direction * speed

    def _update_particle_appearance(self, particle: GaussianParticle, t: float):
        """Update particle appearance based on lifetime curves."""
        # Color
        if self.color_over_lifetime:
            particle.color = self.color_over_lifetime.sample(t)
        else:
            # Default fade out
            particle.color = particle.spawn_color.copy()
            particle.color[3] = particle.spawn_color[3] * (1.0 - t)

        # Scale
        if self.scale_over_lifetime:
            particle.scale = particle.spawn_scale * self.scale_over_lifetime.sample(t)

        # Opacity override
        if self.opacity_over_lifetime:
            particle.color[3] = particle.spawn_color[3] * self.opacity_over_lifetime.sample(t)

    def get_gaussian_data(self) -> List[Dict]:
        """Get all particles as Gaussian splat data for rendering."""
        return [p.to_gaussian_data() for p in self.particles if p.is_alive]


# =============================================================================
# Particle System
# =============================================================================

class ParticleSystem:
    """
    Manages multiple particle emitters.

    Provides unified update and render interface.
    """

    def __init__(self):
        self.emitters: Dict[str, ParticleEmitter] = {}
        self.time: float = 0.0

    def add_emitter(self, emitter: ParticleEmitter):
        """Add an emitter to the system."""
        self.emitters[emitter.id] = emitter
        logger.debug(f"Added particle emitter: {emitter.id}")

    def remove_emitter(self, emitter_id: str):
        """Remove an emitter."""
        if emitter_id in self.emitters:
            del self.emitters[emitter_id]

    def update(self, dt: float):
        """Update all emitters."""
        self.time += dt
        for emitter in self.emitters.values():
            emitter.update(dt)

    def get_all_gaussian_data(self) -> List[Dict]:
        """Get all particles from all emitters as Gaussian data."""
        all_particles = []
        for emitter in self.emitters.values():
            all_particles.extend(emitter.get_gaussian_data())
        return all_particles

    @property
    def total_particle_count(self) -> int:
        """Total particles across all emitters."""
        return sum(len(e.particles) for e in self.emitters.values())


# =============================================================================
# Preset Emitters
# =============================================================================

def create_fire_emitter(id: str, position: Tuple[float, float, float]) -> ParticleEmitter:
    """Create a fire effect emitter."""
    emitter = ParticleEmitter(
        id=id,
        position=np.array(position, dtype=np.float32),
        emission_rate=50.0,
        max_particles=500,
        emitter_shape=EmitterShape.CIRCLE,
        shape_radius=0.2,
        initial_speed=(1.0, 3.0),
        lifetime=(0.5, 1.5),
        gravity=np.array([0, 2, 0]),  # Upward (fire rises)
        drag=0.1,
        start_scale=(0.1, 0.2),
        blend_mode=BlendMode.ADDITIVE,
    )

    # Orange to red to black
    emitter.color_over_lifetime = ColorGradient(keys=[
        (0.0, np.array([1.0, 0.9, 0.3, 1.0])),   # Yellow
        (0.3, np.array([1.0, 0.5, 0.0, 0.8])),   # Orange
        (0.6, np.array([1.0, 0.2, 0.0, 0.5])),   # Red
        (1.0, np.array([0.2, 0.0, 0.0, 0.0])),   # Fade out
    ])

    # Shrink over time
    emitter.scale_over_lifetime = Curve(keys=[
        (0.0, 1.0),
        (0.5, 0.8),
        (1.0, 0.3),
    ])

    return emitter


def create_smoke_emitter(id: str, position: Tuple[float, float, float]) -> ParticleEmitter:
    """Create a smoke effect emitter."""
    emitter = ParticleEmitter(
        id=id,
        position=np.array(position, dtype=np.float32),
        emission_rate=20.0,
        max_particles=200,
        emitter_shape=EmitterShape.CIRCLE,
        shape_radius=0.3,
        initial_speed=(0.5, 1.0),
        lifetime=(2.0, 4.0),
        gravity=np.array([0, 0.5, 0]),  # Slow rise
        drag=0.3,
        start_scale=(0.2, 0.4),
        blend_mode=BlendMode.ALPHA,
    )

    # Gray, fading
    emitter.color_over_lifetime = ColorGradient(keys=[
        (0.0, np.array([0.3, 0.3, 0.3, 0.5])),
        (0.5, np.array([0.5, 0.5, 0.5, 0.3])),
        (1.0, np.array([0.7, 0.7, 0.7, 0.0])),
    ])

    # Expand over time
    emitter.scale_over_lifetime = Curve(keys=[
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 1.5),
    ])

    return emitter


def create_sparkle_emitter(id: str, position: Tuple[float, float, float]) -> ParticleEmitter:
    """Create a sparkle/magic effect emitter."""
    emitter = ParticleEmitter(
        id=id,
        position=np.array(position, dtype=np.float32),
        emission_rate=30.0,
        max_particles=300,
        emitter_shape=EmitterShape.SPHERE,
        shape_radius=0.5,
        initial_speed=(0.2, 0.5),
        speed_spread=1.0,
        lifetime=(0.5, 1.0),
        gravity=np.array([0, -0.5, 0]),  # Slight fall
        drag=0.5,
        start_scale=(0.02, 0.05),
        start_color=np.array([1.0, 1.0, 1.0, 1.0]),
        blend_mode=BlendMode.ADDITIVE,
    )

    # Bright flash then fade
    emitter.opacity_over_lifetime = Curve(keys=[
        (0.0, 0.0),
        (0.1, 1.0),  # Quick flash on
        (0.3, 1.0),
        (1.0, 0.0),  # Fade out
    ])

    return emitter


def create_snow_emitter(id: str, position: Tuple[float, float, float], area_size: float = 10.0) -> ParticleEmitter:
    """Create a snow effect emitter (spawn from above)."""
    emitter = ParticleEmitter(
        id=id,
        position=np.array(position, dtype=np.float32),
        emission_rate=100.0,
        max_particles=2000,
        emitter_shape=EmitterShape.BOX,
        shape_size=np.array([area_size, 0.5, area_size]),
        initial_speed=(0.0, 0.2),
        speed_spread=0.3,
        lifetime=(5.0, 10.0),
        gravity=np.array([0, -1.0, 0]),
        drag=0.8,  # High drag for floating
        start_scale=(0.02, 0.04),
        start_color=np.array([1.0, 1.0, 1.0, 0.9]),
        blend_mode=BlendMode.ALPHA,
    )

    return emitter


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    print("Gaussian Particle System Test")
    print("=" * 40)

    system = ParticleSystem()

    # Add fire emitter
    fire = create_fire_emitter("campfire", (0, 0, 0))
    system.add_emitter(fire)

    # Add smoke emitter above fire
    smoke = create_smoke_emitter("campfire_smoke", (0, 1, 0))
    system.add_emitter(smoke)

    # Simulate 3 seconds
    print("\nSimulating 3 seconds...")
    dt = 1.0 / 60.0
    for frame in range(180):
        system.update(dt)

        if frame % 60 == 0:
            count = system.total_particle_count
            print(f"  Frame {frame}: {count} particles")

    # Get final state
    gaussians = system.get_all_gaussian_data()
    print(f"\nFinal state: {len(gaussians)} Gaussians")

    if gaussians:
        sample = gaussians[0]
        print(f"\nSample particle:")
        print(f"  Position: {sample['position']}")
        print(f"  Scale: {sample['scale']}")
        print(f"  Color: {sample['color']}")
        print(f"  Opacity: {sample['opacity']:.2f}")

    print("\nTest complete!")
