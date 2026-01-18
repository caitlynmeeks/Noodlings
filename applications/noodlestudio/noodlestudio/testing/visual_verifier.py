# ──────────────────────────────────────────────────────────────
#
#   Visual Verifier - Baseline comparison for UI testing
#
#   Compares screenshots against saved baselines using SSIM
#   to detect visual regressions.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.testing.visual_verifier
# PURPOSE:  Visual regression testing
# LAYER:    Studio / Testing
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from PIL import Image
import numpy as np


@dataclass
class VerificationResult:
    """Result of visual verification."""
    passed: bool
    similarity: float  # 0.0 to 1.0
    diff_image: Optional[bytes] = None  # PNG bytes highlighting differences
    message: str = ""


class VisualVerifier:
    """
    Visual verification using baseline comparison.

    Supports:
    - Pixel-perfect comparison
    - SSIM (Structural Similarity Index) for fuzzy matching
    - Region-based comparison (ignore dynamic areas)
    - Diff image generation for debugging
    """

    def __init__(self, baselines_dir: Path = None):
        # Default to tests/ui/baselines relative to noodlestudio
        if baselines_dir is None:
            noodlestudio_path = Path(__file__).parent.parent
            baselines_dir = noodlestudio_path / "tests" / "ui" / "baselines"

        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

        # Default threshold - 95% similarity required
        self.threshold = 0.95

    def capture_baseline(self, name: str, screenshot_b64: str) -> Path:
        """
        Save a screenshot as a baseline.

        Human runs this once to establish "known good" state.

        Args:
            name: Baseline name (without .png extension)
            screenshot_b64: Base64-encoded PNG screenshot

        Returns:
            Path to saved baseline file
        """
        png_bytes = base64.b64decode(screenshot_b64)
        path = self.baselines_dir / f"{name}.png"
        path.write_bytes(png_bytes)
        print(f"[VisualVerifier] Baseline saved: {path}")
        return path

    def verify(
        self,
        name: str,
        screenshot_b64: str,
        threshold: float = None,
        ignore_regions: List[Tuple[int, int, int, int]] = None
    ) -> VerificationResult:
        """
        Compare current screenshot to baseline.

        Args:
            name: Baseline name (without .png)
            screenshot_b64: Current screenshot as base64 PNG
            threshold: Similarity threshold (0.0-1.0), default 0.95
            ignore_regions: List of (x, y, w, h) regions to ignore

        Returns:
            VerificationResult with pass/fail and similarity score
        """
        threshold = threshold or self.threshold
        baseline_path = self.baselines_dir / f"{name}.png"

        if not baseline_path.exists():
            return VerificationResult(
                passed=False,
                similarity=0.0,
                message=f"Baseline not found: {name}. Run capture_baseline first."
            )

        # Load images
        baseline_img = Image.open(baseline_path)
        current_bytes = base64.b64decode(screenshot_b64)
        current_img = Image.open(io.BytesIO(current_bytes))

        # Resize current to match baseline if needed
        if current_img.size != baseline_img.size:
            current_img = current_img.resize(
                baseline_img.size,
                Image.Resampling.LANCZOS
            )

        # Apply ignore regions (fill with baseline content to neutralize differences)
        if ignore_regions:
            for x, y, w, h in ignore_regions:
                region = baseline_img.crop((x, y, x + w, y + h))
                current_img.paste(region, (x, y))

        # Calculate similarity using SSIM
        similarity = self._calculate_ssim(baseline_img, current_img)

        passed = similarity >= threshold

        # Generate diff image if failed
        diff_image = None
        if not passed:
            diff_image = self._generate_diff(baseline_img, current_img)

        return VerificationResult(
            passed=passed,
            similarity=similarity,
            diff_image=diff_image,
            message=f"Similarity: {similarity:.1%} (threshold: {threshold:.1%})"
        )

    def _calculate_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate combined similarity score.

        Uses a combination of:
        - SSIM (Structural Similarity Index) for structure
        - Mean color difference for absolute color changes

        This hybrid approach catches both structural changes (moved elements)
        and color changes (wrong textures, theme issues).
        """
        # Convert to RGB arrays for color comparison
        arr1_rgb = np.array(img1.convert('RGB'), dtype=np.float64)
        arr2_rgb = np.array(img2.convert('RGB'), dtype=np.float64)

        # Calculate mean absolute color difference (normalized to 0-1)
        color_diff = np.abs(arr1_rgb - arr2_rgb).mean() / 255.0
        color_similarity = 1.0 - color_diff

        # Convert to grayscale for structural comparison
        arr1 = np.array(img1.convert('L'), dtype=np.float64)
        arr2 = np.array(img2.convert('L'), dtype=np.float64)

        # SSIM constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        # Compute means
        mu1 = arr1.mean()
        mu2 = arr2.mean()

        # Compute variances
        sigma1_sq = arr1.var()
        sigma2_sq = arr2.var()

        # Compute covariance
        sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()

        # SSIM formula
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))

        # Combine SSIM and color similarity
        # Weight: 50% SSIM (structure) + 50% color similarity
        combined = 0.5 * float(ssim) + 0.5 * color_similarity

        return combined

    def _generate_diff(self, img1: Image.Image, img2: Image.Image) -> bytes:
        """
        Generate a diff image highlighting differences.

        Pixels that differ are shown in red on a grayscale base.
        """
        # Convert to RGB arrays
        arr1 = np.array(img1.convert('RGB'))
        arr2 = np.array(img2.convert('RGB'))

        # Calculate per-pixel difference
        diff = np.abs(arr1.astype(int) - arr2.astype(int))

        # Create diff visualization
        # Red where different, grayscale original where same
        diff_sum = diff.sum(axis=2)
        threshold = 30  # Difference threshold per pixel

        # Start with grayscale version of baseline
        gray = np.array(img1.convert('L'))
        result = np.stack([gray, gray, gray], axis=2)

        # Mark differences in red
        mask = diff_sum > threshold
        result[mask] = [255, 0, 0]  # Red for differences

        # Convert to PNG bytes
        diff_img = Image.fromarray(result.astype(np.uint8))
        buffer = io.BytesIO()
        diff_img.save(buffer, format='PNG')
        return buffer.getvalue()

    def list_baselines(self) -> List[str]:
        """List all available baseline names."""
        baselines = []
        for path in self.baselines_dir.glob("*.png"):
            baselines.append(path.stem)
        return sorted(baselines)

    def delete_baseline(self, name: str) -> bool:
        """Delete a baseline by name."""
        path = self.baselines_dir / f"{name}.png"
        if path.exists():
            path.unlink()
            print(f"[VisualVerifier] Deleted baseline: {name}")
            return True
        return False


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
