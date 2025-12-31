# Asset System Design

## Current Issues
1. Inspector shows "skeleton: No" even when asset has skeleton
2. Inspector doesn't update when different asset loaded
3. Assets panel says "Gaussians" - should say "Radiances"
4. No connection between Load button and Assets panel

## Proposed Unity-Style Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ASSETS PANEL                              │
├─────────────────────────────────────────────────────────────────┤
│  v Noodlings                                                     │
│      Red Fire Anklebiter                                        │
│      Yuki Cyberfox                                              │
│  v Radiances           <-- renamed from "Gaussians"             │
│      fire_imp_rigged_final.radiance  [selected]                 │
│      alicia_densified_tuned.radiance                            │
│      alicia_black_30k.radiance                                  │
│  v VRMs                 <-- NEW                                  │
│      AliciaSolid.vrm                                            │
│  v Meshes               <-- NEW                                  │
│      Fire Imp.obj                                                │
│  v Scripts                                                       │
│  v Stages                                                        │
└─────────────────────────────────────────────────────────────────┘
          │
          │ selection
          v
┌─────────────────────────────────────────────────────────────────┐
│               GAUSSIAN VIEWER PANEL                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │              [3D Viewport]                               │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│  Status: fire_imp_rigged_final | 1,644 Gaussians | 120 FPS     │
└─────────────────────────────────────────────────────────────────┘
          │
          │ auto-updates
          v
┌─────────────────────────────────────────────────────────────────┐
│                 INSPECTOR PANEL                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Radiance Properties                                      │   │
│  │ Name: fire_imp_rigged_final                             │   │
│  │ Path: external/obj/Fire Imp/...                         │   │
│  │ Gaussians: 1,644                                         │   │
│  │ Skeleton: Yes (22 bones)    <-- should show details     │   │
│  │ Regions: body, head, ...                                │   │
│  │                                                          │   │
│  │ Display                                                  │   │
│  │ Scale: [====|====] 1.0x                                 │   │
│  │ Tint: [■] 1.00, 1.00, 1.00                              │   │
│  │ Alpha: [========|] 100%                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Fix Inspector Communication
1. Add logging to verify signal emission
2. Fix RadianceInspector to show skeleton bone count
3. Ensure inspector updates on every asset load

### Phase 2: Rename Gaussians -> Radiances
1. Update AssetsPanel category name
2. Update related signals and methods

### Phase 3: Connect Load -> Assets Panel
1. When "Load Radiance" clicked and file loaded:
   - Add to Assets panel under "Radiances"
   - Select it in the tree
2. When asset selected in Assets panel:
   - Load into Gaussian Viewer
   - Show in Inspector

### Phase 4: Add VRM and Mesh Asset Types
1. Add "VRMs" category to Assets panel
2. Add "Meshes" category (OBJ, FBX, GLTF)
3. Double-click VRM -> convert to Radiance -> view
4. Double-click Mesh -> auto-rig -> view

## Data Flow

```
User clicks "Load Radiance"
    │
    v
GaussianViewerPanel.load_radiance(path)
    │
    ├──> radianceLoaded.emit(path, component)
    │         │
    │         ├──> MainWindow._on_radiance_loaded()
    │         │         │
    │         │         └──> Inspector.load_entity('radiance', data)
    │         │
    │         └──> AssetsPanel.add_radiance(path)  <-- NEW
    │
    └──> selectionChanged.emit(component)

User clicks asset in Assets Panel
    │
    v
AssetsPanel.assetSelected.emit('radiance', path)
    │
    └──> MainWindow._on_asset_selected()
              │
              └──> GaussianViewerPanel.load_radiance(path)
```

## File Changes Required

1. `assets_panel.py`:
   - Rename "Gaussians" to "Radiances"
   - Add `add_radiance(path, name)` method
   - Add VRMs and Meshes categories

2. `gaussian_viewer_panel.py`:
   - Emit signal to AssetsPanel when asset loaded
   - OR: AssetsPanel listens to same signal

3. `main_window.py`:
   - Connect AssetsPanel.assetSelected to viewer
   - Wire up new asset categories

4. `radiance_inspector.py`:
   - Show bone count: "Yes (22 bones)" instead of just "Yes"
   - Add collapsible skeleton details section

5. `inspector_panel.py`:
   - Ensure RadianceInspector is cleared before reload
