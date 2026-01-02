# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Regression testing infrastructure (pytest + pytest-qt)
- GitHub Actions CI workflow
- Pre-commit hook for test validation
- Testing documentation

---

## [0.1.0-alpha] - 2026-01-01

Initial alpha release of NoodleStudio and NoodleMUSH.

### NoodleStudio

#### Added
- **Facet System** - Visual cognitive architecture editor
  - INCOMING/OUTGOING entry/exit nodes
  - LLMFacet, ScriptedFacet, CharmNetworkFacet
  - ConvergenceFacet for multi-input synthesis
  - 31 utility facet types (math, logic, string, array, data)
- **Gaussian Radiance System**
  - GPU rendering via gsplat-mps (120 FPS on Apple Silicon)
  - RadianceComponent with material overrides
  - Semantic labels and CLIP embeddings
  - VRM to .radiance conversion with densification
- **Neural Canvas** - Visual neural network designer
  - LSTM, GRU, attention heads
  - MLX code generation
  - PyTorch test mode
- **Panels**
  - Stage View with drag-drop hierarchy
  - Assets panel (Unity-style filesystem browser)
  - Inspector with entity-specific modes
  - Facets Editor (node graph)
  - Gaussian Viewer (3D preview)
  - Chat panel
  - Console panel
- **Scripting API** - JavaScript facet scripting
  - context.noodle.affect, models, pose, quantum
- **Cloud Authentication** - Cloudflare Workers backend
- **MCP Integration** - Model Context Protocol support

### NoodleMUSH

#### Added
- WebSocket server for real-time multi-user worlds
- Room/zone spatial system
- Perception filtering (who sees what)
- Token-based authentication
- Web client with URL parameter auth

### Infrastructure
- Project-based organization (Noodlings/, Stages/, Prims/)
- YAML-based configuration throughout
- Undo/redo system with command pattern
- Settings persistence

---

## Version History

| Version | Date | Milestone |
|---------|------|-----------|
| 0.1.0-alpha | 2026-01-01 | Initial alpha release |

---

## Versioning Scheme

- **Major** (X.0.0): Breaking changes, major architecture shifts
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, small improvements
- **Pre-release**: -alpha, -beta, -rc.1 suffixes
