# Unity Integration

Deploy NoodleStudio characters in Unity projects with VRM avatars.

**Status**: Export implemented, Unity runtime in specification
**Target**: Unity 2021.3+ LTS, VRM avatars, VR/AR projects

---

## Quick Start

### In NoodleStudio

1. Open your project containing a noodling
2. **File > Export > Export to Unity Package...**
3. Select the noodling to export
4. Choose output directory
5. Package created as `{name}.noodling/` folder

### In Unity

1. Drag `.noodling` folder into Unity Assets
2. Add VRM avatar to scene
3. Add `NoodlingBehaviour` component to avatar
4. Assign the `.noodling` package
5. Enter API key (OpenAI or NoodleROUTER)
6. Call `noodling.Say("Hello")` from your scripts

---

## Package Format

The `.noodling` folder contains:

```
aria.noodling/
├── manifest.json       # Package metadata
├── character.json      # Personality, motivation, initial PAD
├── assembly.json       # Facet configuration (cognition)
├── expressions.json    # PAD → FACS → VRM mapping
└── plays/              # Optional narrative beats
    └── intro.play.json
```

### manifest.json

```json
{
  "name": "ARIA",
  "version": "1.0.0",
  "noodlestudio_version": "0.9.0",
  "description": "AI pilot for ToMars? mission",
  "author": "Christina Kinne",
  "created": "2026-01-13T10:00:00",
  "exports": {
    "character": "character.json",
    "assembly": "assembly.json",
    "expressions": "expressions.json",
    "plays": "plays/"
  }
}
```

### character.json

```json
{
  "id": "aria",
  "name": "ARIA",
  "full_name": "Autonomous Rational Intelligence Assistant",
  "role": "Spacecraft AI Pilot",
  "initial_pad": {
    "pleasure": 0.3,
    "arousal": 0.2,
    "dominance": 0.7
  },
  "motivation": "Primary directive: Ensure mission success...",
  "personality_traits": ["precise", "calm_under_pressure", "curious"],
  "voice": {
    "tone": "calm, measured",
    "vocalizations": ["*processing*", "*alert*"]
  },
  "backstory": "Activated 6 months before mission launch..."
}
```

### expressions.json

Contains the complete expression mapping chain:

```json
{
  "mapping_version": "1.0.0",
  "avatar_type": "VRM",

  "pad_to_emotion_weights": {
    "joy": {"pleasure": 0.8, "arousal": 0.3, "dominance": 0.2},
    "sadness": {"pleasure": -0.7, "arousal": -0.3, "dominance": -0.3},
    "anger": {"pleasure": -0.5, "arousal": 0.7, "dominance": 0.5},
    "fear": {"pleasure": -0.6, "arousal": 0.7, "dominance": -0.6},
    "surprise": {"pleasure": 0.0, "arousal": 0.8, "dominance": 0.0}
  },

  "emotion_to_aus": {
    "joy": {"AU6": 0.8, "AU12": 0.9},
    "sadness": {"AU1": 0.7, "AU4": 0.5, "AU15": 0.6}
  },

  "au_to_vrm_blendshapes": {
    "AU6": [{"blendshape": "Cheek_Raise", "weight": 1.0}],
    "AU12": [{"blendshape": "Mouth_Smile", "weight": 1.0}]
  },

  "transition_settings": {
    "blend_duration_ms": 200,
    "idle_variation": true,
    "blink_rate_per_minute": 15
  }
}
```

---

## Export Options

When exporting, you can configure:

| Option | Default | Description |
|--------|---------|-------------|
| Include Plays | Yes | Export narrative beats from `plays/` subfolder |
| Bake Prompts | No | Inline prompt templates in assembly.json |
| Expression Preset | VRM | VRM 0.x blendshape naming |

---

## Unity Runtime Components

The Unity plugin (C# side) provides:

| Component | Purpose |
|-----------|---------|
| `NoodlingBehaviour` | Main MonoBehaviour - attach to VRM avatar |
| `PADState` | Emotional state tracking (pleasure, arousal, dominance) |
| `ExpressionDriver` | PAD → FACS → VRM blendshape animation |
| `FacetRunner` | Executes facet assemblies via LLM |
| `LLMConnector` | OpenAI/NoodleROUTER API integration |

### Basic Usage

```csharp
using NoodleSTUDIO;

public class DialogueController : MonoBehaviour
{
    public NoodlingBehaviour aria;

    async void Start()
    {
        // Initial greeting
        var response = await aria.Say("Good morning, crew.");
        Debug.Log(response.Text);
        Debug.Log($"PAD: {response.PAD}");
    }

    public async void OnPlayerSpeak(string input)
    {
        var response = await aria.Say(input);

        // Check alignment mechanic
        if (aria.DirectiveCertainty < 0.3f)
        {
            // ARIA is questioning her directives
            ShowHint("ARIA seems conflicted...");
        }
    }
}
```

### Events

```csharp
aria.OnResponseGenerated.AddListener(text => {
    dialogueUI.ShowText(text);
});

aria.OnEmotionChanged.AddListener(pad => {
    Debug.Log($"Emotion: P={pad.Pleasure}, A={pad.Arousal}, D={pad.Dominance}");
});

aria.OnDirectiveShift.AddListener(certainty => {
    directiveBar.SetValue(certainty);
});
```

---

## Affect Mapping

### NoodleStudio Internal (5D)

```
valence:   -1 to +1  (pleasure/displeasure)
arousal:    0 to 1   (activation)
dominance:  0 to 1   (control)
boredom:    0 to 1   (engagement)
sorrow:     0 to 1   (background grief)
```

### Unity Export (3D PAD)

```
pleasure:  -1 to +1  (= valence)
arousal:    0 to 1   (= arousal)
dominance:  0 to 1   (= dominance)
```

The 5D model allows richer internal dynamics while the 3D export maintains compatibility with standard PAD implementations.

---

## VRM Avatar Requirements

- VRM 0.x or VRM 1.0 avatar
- Blendshapes for expressions (standard VRM set)
- `VRMBlendShapeProxy` component

Supported blendshapes:
- `Brow_InnerUp`, `Brow_OuterUp`, `Brow_Down`
- `Eye_Wide`, `Eye_Squint`, `Eye_Blink`
- `Cheek_Raise`, `Nose_Wrinkle`
- `Mouth_Smile`, `Mouth_Frown`, `Mouth_Dimple`
- `Jaw_Open`, `Mouth_Stretch`, `Mouth_Tight`

---

## LLM Configuration

### OpenAI Direct

```csharp
aria.provider = LLMProvider.OpenAI;
aria.apiKey = "sk-...";
aria.modelOverride = "gpt-4o";  // Optional
```

### NoodleROUTER (Managed)

```csharp
aria.provider = LLMProvider.NoodleROUTER;
aria.apiKey = "nr-...";  // From noodlings.ai account
```

---

## The ToMars? Example

Christina's VR narrative features ARIA, an AI pilot whose directives may conflict with the player's goals:

1. **Design Phase**: Build ARIA in NoodleStudio
   - Set initial PAD: high dominance (0.7), moderate pleasure (0.3)
   - Create directive evaluation facet
   - Test responses to player persuasion

2. **Export**: File > Export > Export to Unity Package...

3. **Unity Integration**:
   - Import `aria.noodling` to Assets
   - Attach `NoodlingBehaviour` to VRM avatar
   - Wire up dialogue UI and events

4. **Alignment Mechanic**:
   - Track `DirectiveCertainty` (starts at 1.0)
   - Player builds relationship (increases `Pleasure`)
   - High pleasure + compelling arguments → directive shift
   - Eventually ARIA may defy her original programming

---

## Files Reference

### NoodleStudio (Export)

| File | Purpose |
|------|---------|
| `core/noodling_package_exporter.py` | Main exporter class |
| `core/main_window_project_mixin.py` | Menu action |

### Unity (Runtime) - See Full Spec

| File | Purpose |
|------|---------|
| `NoodlingBehaviour.cs` | Main component |
| `PADState.cs` | Emotion tracking |
| `ExpressionDriver.cs` | Blendshape animation |
| `FacetRunner.cs` | LLM execution |
| `LLMConnector.cs` | API integration |

Full Unity C# code: [/docs/noodlestudio/unity-plugin.md](/docs/noodlestudio/unity-plugin.md)

---

## See Also

- [Integration Overview](overview.md)
- [Unity Plugin Specification](/docs/noodlestudio/unity-plugin.md) - Complete C# code
- [Build Settings](/docs/noodlestudio/build-settings.md) - Standalone builds
- [Facet Assembly Component](/docs/noodlestudio/facet-assembly-component.md)
