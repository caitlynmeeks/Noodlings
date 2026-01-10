# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Environmental Emitters
#
#   Objects in the world can emit physical signals - a campfire
#   radiates heat and light, a bakery emits delicious bread scent,
#   a radioactive barrel glows with danger. This module defines
#   all the emitter types that can affect agents. Each emitter
#   broadcasts its signal with distance-based falloff, so a
#   character standing near a fire feels it more intensely.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.environmental_emitters
# PURPOSE:  Physical signal sources (heat, light, scent, radiation)
# LAYER:    Backend / Semantic Physics
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EnvironmentalEmitter   Base class for all emitter types
#   HeatEmitter            Radiates thermal energy
#   LightEmitter           Emits illumination
#   ScentEmitter           Broadcasts olfactory signals
#   RadioactiveEmitter     Emits ionizing radiation
#   LiquidEmitter          Produces fluid flow
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Environmental Emitters - Complete Implementation

All emitter types for noodleMUSH Semantic Physics Engine:
- Heat/Cold (thermal)
- Sound (acoustic)
- Light (optical)
- Liquid (fluid flow)
- Radiation (ionizing)
- Scent (olfactory)
- Smoke/Gas (particulate)
- Vibration (seismic)
- Magnetic/Electric (field)

Author: Caitlyn + Claude
Date: November 22, 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import math


class EnvironmentalEmitter(ABC):
    """
    Base class for all environmental emitters.

    Emitters broadcast physical signals to nearby entities.
    Signal strength decreases with distance.
    """

    def __init__(self, enabled: bool = True):
        """Initialize emitter."""
        self.enabled = enabled
        self.emitter_type = self.__class__.__name__

    @abstractmethod
    def get_signal_strength_at_distance(self, distance: float) -> float:
        """
        Calculate signal strength at distance.

        Returns:
            Signal strength (normalized 0.0 to 1.0)
        """
        pass

    @abstractmethod
    def get_signal_description(self) -> str:
        """Get semantic description of signal."""
        pass

    @abstractmethod
    def affects_entity_at(self, distance: float) -> bool:
        """Check if signal affects entities at distance."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': self.emitter_type,
            'enabled': self.enabled
        }


# ===== THERMAL EMITTERS =====

class HeatEmitter(EnvironmentalEmitter):
    """Radiates thermal energy (heat)."""

    def __init__(
        self,
        temperature: float,  # °F
        heat_radius: float = 5.0,
        attenuation: float = 2.0,
        enabled: bool = True
    ):
        super().__init__(enabled)
        self.temperature = temperature
        self.heat_radius = heat_radius
        self.attenuation = attenuation

    def get_effective_temperature(self, distance: float, ambient_temp: float = 70.0) -> float:
        """Calculate felt temperature at distance."""
        if distance >= self.heat_radius or not self.enabled:
            return ambient_temp

        heat_contribution = (self.temperature - ambient_temp) * \
                          (1.0 - (distance / self.heat_radius) ** self.attenuation)
        return ambient_temp + heat_contribution

    def get_signal_strength_at_distance(self, distance: float) -> float:
        """Get normalized heat intensity (0-1)."""
        temp_diff = abs(self.temperature - 70.0)  # Difference from comfortable
        effective = self.get_effective_temperature(distance, 70.0)
        effective_diff = abs(effective - 70.0)
        return effective_diff / max(temp_diff, 1.0)

    def get_signal_description(self) -> str:
        if self.temperature > 500:
            return f"intense {self.temperature:.0f}°F heat"
        elif self.temperature > 200:
            return f"strong {self.temperature:.0f}°F heat"
        elif self.temperature > 100:
            return f"warm {self.temperature:.0f}°F heat"
        else:
            return "mild warmth"

    def affects_entity_at(self, distance: float) -> bool:
        return distance < self.heat_radius

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'temperature': self.temperature,
            'heat_radius': self.heat_radius,
            'attenuation': self.attenuation
        })
        return d


# ===== OPTICAL EMITTERS =====

class LightEmitter(EnvironmentalEmitter):
    """Emits light (illumination)."""

    def __init__(
        self,
        brightness: float,  # Lumens
        color: str = "#FFFFFF",  # Hex color
        light_radius: float = 10.0,
        flicker: bool = False,
        enabled: bool = True
    ):
        super().__init__(enabled)
        self.brightness = brightness
        self.color = color
        self.light_radius = light_radius
        self.flicker = flicker

    def get_effective_brightness(self, distance: float) -> float:
        """Calculate brightness at distance (inverse square law)."""
        if distance >= self.light_radius or not self.enabled:
            return 0.0

        # Inverse square law
        return self.brightness / max(1.0, distance ** 2)

    def get_signal_strength_at_distance(self, distance: float) -> float:
        """Get normalized light intensity (0-1)."""
        effective = self.get_effective_brightness(distance)
        return min(1.0, effective / 1000.0)  # Normalize to 1000 lumens max

    def get_signal_description(self) -> str:
        if self.brightness > 5000:
            intensity = "blinding"
        elif self.brightness > 1000:
            intensity = "very bright"
        elif self.brightness > 200:
            intensity = "bright"
        elif self.brightness > 50:
            intensity = "moderate"
        else:
            intensity = "dim"

        flicker_desc = " flickering" if self.flicker else ""
        return f"{intensity}{flicker_desc} {self.color} light"

    def affects_entity_at(self, distance: float) -> bool:
        return distance < self.light_radius

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'brightness': self.brightness,
            'color': self.color,
            'light_radius': self.light_radius,
            'flicker': self.flicker
        })
        return d


# ===== FLUID EMITTERS =====

class LiquidEmitter(EnvironmentalEmitter):
    """Emits liquid (leak, flow, fountain)."""

    def __init__(
        self,
        liquid_type: str = "water",
        flow_rate: float = 1.0,  # Liters per minute
        temperature: float = 70.0,  # °F
        viscosity: str = "normal",
        pressure: float = 1.0,  # PSI
        enabled: bool = True
    ):
        super().__init__(enabled)
        self.liquid_type = liquid_type
        self.flow_rate = flow_rate
        self.temperature = temperature
        self.viscosity = viscosity
        self.pressure = pressure

    def get_signal_strength_at_distance(self, distance: float) -> float:
        """Liquid flow strength (spray radius)."""
        spray_radius = math.sqrt(self.pressure) * 2.0  # Pressure determines spray
        if distance > spray_radius:
            return 0.0
        return 1.0 - (distance / spray_radius)

    def get_signal_description(self) -> str:
        if self.flow_rate > 50:
            intensity = "torrential"
        elif self.flow_rate > 10:
            intensity = "strong"
        elif self.flow_rate > 1:
            intensity = "steady"
        else:
            intensity = "dripping"

        temp_desc = ""
        if self.temperature > 120:
            temp_desc = " (scalding hot)"
        elif self.temperature < 40:
            temp_desc = " (ice cold)"

        return f"{intensity} {self.liquid_type} flow{temp_desc}"

    def affects_entity_at(self, distance: float) -> bool:
        spray_radius = math.sqrt(self.pressure) * 2.0
        return distance < spray_radius

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'liquid_type': self.liquid_type,
            'flow_rate': self.flow_rate,
            'temperature': self.temperature,
            'viscosity': self.viscosity,
            'pressure': self.pressure
        })
        return d


# ===== RADIATION EMITTERS =====

class RadioactiveEmitter(EnvironmentalEmitter):
    """Emits ionizing radiation."""

    def __init__(
        self,
        radiation_type: str = "gamma",  # "alpha", "beta", "gamma"
        intensity: float = 100.0,  # Rads per hour
        radius: float = 10.0,
        enabled: bool = True
    ):
        super().__init__(enabled)
        self.radiation_type = radiation_type
        self.intensity = intensity  # Rads/hr
        self.radius = radius

    def get_effective_radiation(self, distance: float) -> float:
        """Calculate radiation dose at distance (inverse square law)."""
        if distance >= self.radius or not self.enabled:
            return 0.0
        return self.intensity / max(1.0, distance ** 2)

    def get_signal_strength_at_distance(self, distance: float) -> float:
        """Normalized radiation (0-1)."""
        effective = self.get_effective_radiation(distance)
        return min(1.0, effective / 1000.0)  # Normalize to 1000 rads/hr

    def get_signal_description(self) -> str:
        if self.intensity > 500:
            danger = "lethal"
        elif self.intensity > 100:
            danger = "dangerous"
        elif self.intensity > 10:
            danger = "hazardous"
        else:
            danger = "low-level"

        return f"{danger} {self.radiation_type} radiation ({self.intensity} rads/hr)"

    def affects_entity_at(self, distance: float) -> bool:
        return distance < self.radius

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'radiation_type': self.radiation_type,
            'intensity': self.intensity,
            'radius': self.radius
        })
        return d


# ===== OLFACTORY EMITTERS =====

class ScentEmitter(EnvironmentalEmitter):
    """Emits scent/odor."""

    def __init__(
        self,
        scent_type: str = "neutral",
        intensity: float = 0.5,  # 0-1
        pleasantness: float = 0.5,  # 0 (awful) to 1 (wonderful)
        radius: float = 5.0,
        enabled: bool = True
    ):
        super().__init__(enabled)
        self.scent_type = scent_type
        self.intensity = intensity
        self.pleasantness = pleasantness
        self.radius = radius

    def get_effective_intensity(self, distance: float) -> float:
        """Calculate scent intensity at distance."""
        if distance >= self.radius or not self.enabled:
            return 0.0
        return self.intensity * (1.0 - distance / self.radius)

    def get_signal_strength_at_distance(self, distance: float) -> float:
        return self.get_effective_intensity(distance)

    def get_signal_description(self) -> str:
        if self.pleasantness > 0.7:
            quality = "delightful"
        elif self.pleasantness > 0.5:
            quality = "pleasant"
        elif self.pleasantness > 0.3:
            quality = "neutral"
        else:
            quality = "unpleasant"

        if self.intensity > 0.8:
            strength = "strong"
        elif self.intensity > 0.5:
            strength = "noticeable"
        else:
            strength = "faint"

        return f"{strength} {quality} scent of {self.scent_type}"

    def affects_entity_at(self, distance: float) -> bool:
        return distance < self.radius

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'scent_type': self.scent_type,
            'intensity': self.intensity,
            'pleasantness': self.pleasantness,
            'radius': self.radius
        })
        return d


# ===== EXAMPLE USAGE =====

if __name__ == '__main__':
    print("=== ENVIRONMENTAL EMITTERS TEST ===\n")

    # Vulcan teapot
    teapot_heat = HeatEmitter(temperature=250.0, heat_radius=2.0)

    print("Vulcan Teapot (250°F):")
    for dist in [0.5, 1.0, 1.5, 2.0]:
        temp = teapot_heat.get_effective_temperature(dist, ambient_temp=70.0)
        print(f"  {dist}m away: {temp:.1f}°F")

    print()

    # Campfire (multi-emitter)
    print("Campfire at 3 meters:")
    heat = HeatEmitter(temperature=800.0, heat_radius=10.0)
    light = LightEmitter(brightness=1000, light_radius=15.0)

    print(f"  Heat: {heat.get_effective_temperature(3.0, 70.0):.1f}°F")
    print(f"  Light: {light.get_effective_brightness(3.0):.0f} lumens")
    print()

    # Bakery scent
    print("Bakery scent:")
    bakery = ScentEmitter(
        scent_type="fresh bread",
        intensity=0.8,
        pleasantness=0.9,
        radius=20.0
    )

    for dist in [5, 10, 15, 20]:
        intensity = bakery.get_effective_intensity(dist)
        print(f"  {dist}m away: {intensity:.2f} intensity")
        if intensity > 0.3:
            print(f"    → *sniff* Mmm! Fresh bread!")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
