# Quick Start: NoodleStudio Gaussian Tests

## Launch
```bash
cd applications/noodlestudio
./launch_with_log.sh
```

## Available Panels
- **Gaussian Viewer** - 3D viewport for .radiance assets
- **Radiance Inspector** - Material properties (tint, scale, alpha)
- **Assets Panel** - File browser
- **Facets Editor** - Node-based cognitive architecture
- **Console** - Script execution

## Test Assets (ready to load)

| Asset | Path | Description |
|-------|------|-------------|
| Fire Imp (rigged) | `external/obj/Fire Imp/fire_imp_rigged_final.radiance` | Textured, 22-bone skeleton |
| Alicia (VRM) | `external/vrm_samples/alicia_densified_tuned.radiance` | 137K Gaussians, best quality |
| Alicia (black BG) | `external/datasets/alicia_black/alicia_black_30k.radiance` | 30K trained Gaussians |
| Alicia (white BG) | `external/datasets/alicia_views/alicia_30k_clean.radiance` | 9K trained (sparse) |

## Quick Tests

### Test 1: Load Fire Imp
1. Open Gaussian Viewer panel
2. Load `fire_imp_rigged_final.radiance`
3. Expected: Orange imp with flame hands
4. Try orbit (left-drag), zoom (scroll)

### Test 2: Adjust Properties
1. Open Radiance Inspector
2. Change Scale to 1.5x
3. Change Tint to blue
4. Expected: Imp gets bigger and blue-tinted

### Test 3: Compare Quality
1. Load each Alicia asset in sequence
2. Rate quality 1-5
3. Note any artifacts or missing features

## Report Back
Tell Claude:
- FPS you're getting
- Any panels that don't open
- Any crashes (check `logs/noodlestudio_*.log`)
- What feels good vs clunky
