# NOODLE_CODE.md

Instructions and context for the Noodle Code AI assistant embedded in NoodleStudio.

---

## About This File

This file is automatically loaded when a project opens. Add project-specific context,
coding conventions, and instructions here. Noodle Code will use this to understand
your project better and provide more relevant assistance.

---

## NoodleStudio Architecture

NoodleStudio is a PyQt6 desktop IDE for designing cognitive architectures.

### Key Panels (Dockable)
| Panel | Purpose | Key Class |
|-------|---------|-----------|
| **Stage View** | Scene hierarchy tree (zones, noodlings, props) | `scene_hierarchy.py` |
| **Inspector** | Properties for selected entity | `inspector_panel.py` |
| **Assets** | Project file browser | `assets_panel.py` |
| **Facets Editor** | Visual node graph for cognitive pipelines | `facets_editor_panel.py` |
| **Neural Canvas** | Creative AI sandbox | `neural_canvas/` |
| **Console** | Python REPL and logs | `console_panel.py` |
| **Gaussian Viewer** | 3D radiance/splat viewport | `gaussian_viewer_panel.py` |
| **Chat** | noodleMUSH world interaction | `chat_panel.py` |

### Core Systems
| System | Purpose | Key Files |
|--------|---------|-----------|
| **Facet System** | Cognitive node pipelines | `facet_system.py`, `facet_executor.py` |
| **Scene Graph** | Entity hierarchy management | `scene_graph.py`, `scene_node.py` |
| **Radiance** | Gaussian splat rendering | `radiance_component.py`, `gaussian_renderer.py` |
| **Provider Manager** | Multi-LLM backend routing | `provider_manager.py` |
| **Model Labels** | Semantic model routing (Small/Medium/Large) | `model_label_manager.py` |

### Project Structure
```
MyProject/
  project.yaml           # Project manifest
  Noodlings/             # AI characters
    red/
      recipe.yaml        # Character definition
      assembly.yaml      # Facet topology
  Stages/                # Scenes/levels
    main_stage/
      stage.yaml
      hierarchy.yaml     # Entity tree
      zones/
      props/
  Prims/                 # Reusable objects
  Radiances/             # Gaussian splat assets
```

---

## Scripting API

Noodle Code can execute JavaScript in ScriptedFacets. The `context` object provides:

### context.noodle.models
```javascript
// Get model for a label
let model = context.noodle.models.getModelForLabel("Large");

// Call LLM directly
let response = await context.noodle.models.complete({
    label: "Medium",  // or provider: "anthropic", model: "claude-sonnet-4"
    prompt: "Explain quantum computing",
    max_tokens: 500
});
```

### context.noodle.affect
```javascript
// Read current affect state
let affect = context.noodle.affect.current;
// { valence: 0.3, arousal: 0.6, dominance: 0.5, boredom: 0.1, sorrow: 0.0 }

// Nudge affect
context.noodle.affect.nudge({ valence: 0.1, arousal: -0.05 });
```

### context.noodle.pose
```javascript
// Set bone transforms
context.noodle.pose.setBone("head", { rotation: [0, 15, 0] });

// Play animation
context.noodle.pose.playTrack("wave.posetrack");
```

### context.noodle.scene
```javascript
// Find entities
let noodlings = context.noodle.scene.findByType("noodling");
let red = context.noodle.scene.findByName("Red");

// Spatial queries
let nearby = context.noodle.scene.findNear(red.position, radius=5.0);
```

### context.noodle.mcp
```javascript
// Call MCP server tools
let result = await context.noodle.mcp.call("filesystem", "read_file", {
    path: "/tmp/data.json"
});
```

---

## Common Operations

### Creating a New Noodling
1. Right-click in Assets > New > Noodling
2. Or use bash: `mkdir -p Noodlings/mychar && touch Noodlings/mychar/recipe.yaml`
3. Recipe.yaml template:
```yaml
name: MyCharacter
personality:
  openness: 0.7
  conscientiousness: 0.5
  extraversion: 0.6
  agreeableness: 0.8
  neuroticism: 0.3
affect_baseline:
  valence: 0.2
  arousal: 0.4
  dominance: 0.5
assembly: assembly.yaml
```

### Creating a Facet Assembly
1. Open Facets Editor panel
2. Right-click canvas > Add Facet
3. Connect INCOMING -> processing facets -> OUTGOING
4. Save as assembly.yaml

### Hot Reloading Code
After editing Python files, you can hot-reload without restart:
```
hot_reload(module_name="noodlestudio.core.utility_facets")
```
For panel/mixin changes, use `soft_restart(confirm=true)`.

### Using Computer Use
To interact with the UI visually:
```
1. computer_use(action="screenshot")  # See current state
2. Analyze image to find coordinates
3. computer_use(action="left_click", coordinate=[x, y])
4. computer_use(action="screenshot")  # Verify result
```

---

## GitHub Integration

Use `gh` CLI for GitHub operations:
```bash
# Issues
gh issue list
gh issue view 42
gh issue create --title "Bug: description" --label bug

# Pull Requests
gh pr list
gh pr view 123
gh pr create --title "Feature: description" --body "Details..."

# Search
gh search issues "crash" --repo owner/repo
```

---

## Project-Specific Context

<!-- Add your project-specific notes below -->

### Current Focus
(What are you currently working on?)

### Key Files
(Important files for current work)

### Conventions
(Project-specific coding conventions)

### Known Issues
(Current bugs or limitations to be aware of)

---

## Tips for Effective Assistance

1. **Read before editing**: Always use `read_file` before `edit_file`
2. **Search first**: Use `glob` and `grep` to understand existing patterns
3. **Small edits**: Prefer targeted `edit_file` over full `write_file`
4. **Test changes**: Suggest running relevant tests after modifications
5. **Follow patterns**: Match existing code style in the project
6. **Use screenshots**: When debugging UI, take screenshots to understand state
7. **Hot reload**: Use `hot_reload` for quick iteration on tool/facet code

---

*This file is read by Noodle Code on project load. Keep it updated!*
