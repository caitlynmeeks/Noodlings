# NoodleStudio UX Test Plan

**Date:** December 24, 2025
**Tester:** Caitlyn
**Purpose:** Validate Gaussian rendering, auto-rigging, and animation features

---

## Test Setup

1. Launch NoodleStudio:
   ```bash
   cd applications/noodlestudio
   ./launch_with_log.sh
   ```

2. Note your starting state:
   - [ ] App launches without errors
   - [ ] Main window appears
   - [ ] Status bar shows at bottom

---

## TEST 1: Gaussian Viewer Panel

**Goal:** Verify GPU-accelerated Gaussian rendering works

### Steps:
1. Go to **View > Gaussian Viewer** (or find the panel)
2. Click **Load Asset** button
3. Navigate to: `external/vrm_samples/alicia_densified_tuned.radiance`
4. The character should appear in the viewport

### Report:
- [ ] Asset loaded successfully
- [ ] Character visible in viewport
- [ ] FPS counter shows (should be ~60-120 FPS)
- Actual FPS: ______
- Any errors: ______

### Camera Controls (test each):
- [ ] Left-drag: Orbit camera works
- [ ] Right-drag: Pan camera works
- [ ] Scroll: Zoom works
- [ ] Press F: Focus/reset view works

---

## TEST 2: Fire Imp Auto-Rigged Asset

**Goal:** Test loading auto-rigged Gaussian asset

### Steps:
1. In Gaussian Viewer, click **Load Asset**
2. Navigate to: `external/obj/Fire Imp/fire_imp_rigged_final.radiance`
3. Character should appear (orange fire imp)

### Report:
- [ ] Fire Imp loads successfully
- [ ] Colors correct (orange/red/yellow)
- [ ] Full body visible (ears, hands, feet)
- Any issues: ______

---

## TEST 3: Radiance Inspector

**Goal:** Test material property adjustments

### Steps:
1. With Fire Imp loaded, open **View > Radiance Inspector** (or Properties panel)
2. Try adjusting:
   - **Scale multiplier**: Drag slider (0.5x to 2.0x)
   - **Tint color**: Change to blue or green
   - **Alpha/Opacity**: Reduce to 0.5

### Report:
- [ ] Scale slider responsive
- [ ] Tint changes visible in real-time
- [ ] Alpha changes visible
- Latency feels: [ ] Instant [ ] Slight delay [ ] Laggy

---

## TEST 4: Asset Comparison

**Goal:** Compare trained vs VRM-converted assets

### Steps:
1. Load: `external/datasets/alicia_views/alicia_30k_clean.radiance` (white BG trained)
2. Note quality
3. Load: `external/datasets/alicia_black/alicia_black_30k.radiance` (black BG trained)
4. Note quality
5. Load: `external/vrm_samples/alicia_densified_tuned.radiance` (VRM reference)
6. Note quality

### Report:
| Asset | Loads OK | Quality (1-5) | Notes |
|-------|----------|---------------|-------|
| White BG trained | [ ] | ___ | |
| Black BG trained | [ ] | ___ | |
| VRM reference | [ ] | ___ | |

---

## TEST 5: Neural Canvas (if available)

**Goal:** Test node-based cognitive architecture editor

### Steps:
1. Go to **View > Neural Canvas** or **Facets Editor**
2. Try creating a simple node graph
3. Connect nodes together

### Report:
- [ ] Canvas opens
- [ ] Can add nodes
- [ ] Can connect nodes
- [ ] Nodes execute without errors
- Any issues: ______

---

## TEST 6: Performance Under Load

**Goal:** Test with multiple assets

### Steps:
1. Open multiple Gaussian Viewer panels (if possible)
2. Load different assets in each
3. Orbit all cameras simultaneously

### Report:
- [ ] Multiple panels work
- [ ] No crashes
- FPS with multiple assets: ______
- Memory usage (Activity Monitor): ______ GB

---

## TEST 7: Animation/Posing (Scripting)

**Goal:** Test if scripting API can pose rigged assets

### Steps:
1. Open **View > Script Console** (if available)
2. Try running:
   ```javascript
   // Get loaded asset
   let asset = context.noodle.viewer.currentAsset;
   console.log("Gaussians:", asset.gaussianCount);
   console.log("Has skeleton:", asset.hasSkeleton);
   ```

### Report:
- [ ] Console opens
- [ ] Script executes
- Output: ______

---

## General Observations

### What worked well:
1.
2.
3.

### What felt clunky or confusing:
1.
2.
3.

### Features you expected but couldn't find:
1.
2.
3.

### Crashes or errors (paste any error messages):
```

```

---

## System Info

- macOS version: ______
- Memory: ______ GB
- GPU: ______
- NoodleStudio version: ______

---

**Thank you for testing! Return this filled form to Claude.**
