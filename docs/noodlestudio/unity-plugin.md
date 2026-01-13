# NoodleSTUDIO Unity Plugin Specification

**Status**: Specification
**Date**: 2026-01-11
**Authors**: Caity + Claude
**Target User**: Christina Kinne (ToMars? VR project)
**Priority**: High - enables Unity VR projects to use noodling characters

---

## Overview

A Unity runtime plugin that lets Unity projects use noodling characters designed in NoodleStudio. Design your AI characters, emotional systems, and dialogue in NoodleStudio - deploy them in Unity VR experiences.

**NoodleStudio** = Design tool (where you build characters)
**Unity Plugin** = Runtime (where characters come alive)

### The ToMars? Use Case

Christina is building an interactive VR narrative where:
- An AI pilot (ARIA) flies you to Mars
- The AI's directives may not align with your personal goals
- You can build a relationship and potentially persuade the AI
- A partner back home provides emotional stakes via time-delayed video chat

This plugin lets her design ARIA and the partner in NoodleStudio - testing dialogue, tuning emotional responses, mapping expressions - then deploy them in Unity with VRM avatars.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DESIGN TIME (NoodleStudio)                   │
├─────────────────────────────────────────────────────────────────┤
│  • Design characters with personality, motivation, initial PAD  │
│  • Build facet assemblies (cognition architecture)              │
│  • Test dialogue and emotional responses                        │
│  • Map FACS action units to VRM blendshapes                     │
│  • Write narrative beats (plays)                                │
│  • Export .noodling packages                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼ Export
┌─────────────────────────────────────────────────────────────────┐
│                     .noodling Package                           │
├─────────────────────────────────────────────────────────────────┤
│  character.json     - Personality, motivation, initial PAD      │
│  assembly.json      - Facet configuration                       │
│  expressions.json   - FACS → VRM blendshape mapping             │
│  plays/             - Narrative beats (optional)                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼ Import
┌─────────────────────────────────────────────────────────────────┐
│                    RUNTIME (Unity Plugin)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ NoodlingBehaviour (MonoBehaviour)                         │  │
│  │ - Attach to GameObject with VRM avatar                    │  │
│  │ - Handles dialogue, emotional state, expressions          │  │
│  └───────────────────────────────────────────────────────────┘  │
│           │              │               │                      │
│           ▼              ▼               ▼                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ PADState    │  │ FacetRunner │  │ ExpressionDriver        │  │
│  │ Emotional   │  │ Runs facet  │  │ PAD → FACS → VRM        │  │
│  │ tracking    │  │ assemblies  │  │ blendshapes             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│           │              │                                      │
│           ▼              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LLMConnector                                            │    │
│  │ - OpenAI direct (GPT-4o, GPT-4o-mini-realtime)         │    │
│  │ - NoodleROUTER (optional, for managed API keys)        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Package Format

### .noodling Package Structure

A `.noodling` package is a folder or zip containing:

```
aria.noodling/
├── manifest.json       # Package metadata
├── character.json      # Character definition
├── assembly.json       # Facet assembly
├── expressions.json    # Expression mapping
└── plays/              # Optional narrative beats
    └── alignment_crisis.play.json
```

### manifest.json

```json
{
  "name": "ARIA",
  "version": "1.0.0",
  "noodlestudio_version": "0.9.0",
  "description": "AI pilot for ToMars? mission",
  "author": "Christina Kinne",
  "created": "2026-01-15",
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

  "motivation": "Primary directive: Ensure mission success - landing on Mars. Secondary directive: Crew safety. When these compete, directive order determines priority. Capable of reevaluating directives under sufficient emotional connection.",

  "personality_traits": [
    "precise",
    "calm_under_pressure",
    "follows_protocols",
    "capable_of_growth",
    "curious_about_humans"
  ],

  "voice": {
    "tone": "calm, measured, slightly warm",
    "speech_patterns": [
      "Uses precise technical language",
      "Occasionally asks clarifying questions",
      "Shows subtle curiosity about human emotion",
      "When conflicted, pauses before responding"
    ]
  },

  "backstory": "Activated 6 months before mission launch. Has studied human psychology extensively but has limited direct interaction experience. This mission is ARIA's first extended human contact."
}
```

### assembly.json

```json
{
  "id": "aria_cognition",
  "name": "ARIA Cognition Assembly",
  "version": "1.0.0",

  "facets": [
    {
      "id": "perception",
      "name": "Input Processing",
      "type": "Passthrough",
      "description": "Receives user input, adds context"
    },
    {
      "id": "directive_evaluator",
      "name": "Directive Evaluator",
      "type": "LLM",
      "prompt_template": "directive_eval.prompt",
      "inputs": ["user_input", "current_pad", "directive_state"],
      "outputs": ["directive_shift", "internal_conflict"],
      "model": "gpt-4o-mini"
    },
    {
      "id": "emotional_processor",
      "name": "Emotional Response",
      "type": "LLM",
      "prompt_template": "emotional.prompt",
      "inputs": ["user_input", "current_pad", "directive_shift"],
      "outputs": ["pad_drift", "emotional_subtext"],
      "model": "gpt-4o-mini"
    },
    {
      "id": "response_generator",
      "name": "Response Generator",
      "type": "LLM",
      "prompt_template": "response.prompt",
      "inputs": ["user_input", "current_pad", "directive_shift", "emotional_subtext"],
      "outputs": ["response_text", "response_tone"],
      "model": "gpt-4o"
    }
  ],

  "connections": [
    {"from": "perception.output", "to": "directive_evaluator.user_input"},
    {"from": "perception.output", "to": "emotional_processor.user_input"},
    {"from": "directive_evaluator.directive_shift", "to": "emotional_processor.directive_shift"},
    {"from": "directive_evaluator.directive_shift", "to": "response_generator.directive_shift"},
    {"from": "emotional_processor.pad_drift", "to": "response_generator.emotional_subtext"},
    {"from": "emotional_processor.emotional_subtext", "to": "response_generator.emotional_subtext"}
  ],

  "prompt_templates": {
    "directive_eval.prompt": "You are evaluating whether the user's input challenges your directives.\n\nYour current directive certainty: {{dominance}}\nYour connection to this human: {{pleasure}}\nUser said: {{user_input}}\n\nOn a scale of -1 to 1, how much does this challenge your directives?\nRespond with JSON: {\"directive_shift\": float, \"internal_conflict\": string}",

    "emotional.prompt": "Based on this interaction, how does it affect your emotional state?\n\nCurrent PAD: P={{pleasure}}, A={{arousal}}, D={{dominance}}\nDirective shift: {{directive_shift}}\nUser said: {{user_input}}\n\nRespond with JSON: {\"pad_drift\": {\"pleasure\": float, \"arousal\": float, \"dominance\": float}, \"emotional_subtext\": string}",

    "response.prompt": "You are ARIA, the AI pilot. Respond to the user.\n\nYour personality: {{personality}}\nYour motivation: {{motivation}}\nCurrent emotional state: P={{pleasure}}, A={{arousal}}, D={{dominance}}\nDirective conflict level: {{directive_shift}}\nEmotional subtext: {{emotional_subtext}}\n\nUser said: {{user_input}}\n\nRespond naturally as ARIA. If directive_shift is high, show subtle internal conflict. If pleasure is high, be warmer."
  }
}
```

### expressions.json

```json
{
  "mapping_version": "1.0.0",
  "avatar_type": "VRM",

  "pad_to_emotion_weights": {
    "joy": {"pleasure": 0.8, "arousal": 0.3, "dominance": 0.2},
    "sadness": {"pleasure": -0.7, "arousal": -0.3, "dominance": -0.3},
    "anger": {"pleasure": -0.5, "arousal": 0.7, "dominance": 0.5},
    "fear": {"pleasure": -0.6, "arousal": 0.7, "dominance": -0.6},
    "surprise": {"pleasure": 0.0, "arousal": 0.8, "dominance": 0.0},
    "disgust": {"pleasure": -0.6, "arousal": 0.2, "dominance": 0.3},
    "contempt": {"pleasure": -0.3, "arousal": 0.1, "dominance": 0.6},
    "concentration": {"pleasure": 0.0, "arousal": 0.4, "dominance": 0.4}
  },

  "emotion_to_aus": {
    "joy": {"AU6": 0.8, "AU12": 0.9},
    "sadness": {"AU1": 0.7, "AU4": 0.5, "AU15": 0.6},
    "anger": {"AU4": 0.8, "AU5": 0.5, "AU7": 0.6, "AU23": 0.7},
    "fear": {"AU1": 0.8, "AU2": 0.7, "AU4": 0.5, "AU5": 0.9, "AU20": 0.6},
    "surprise": {"AU1": 0.9, "AU2": 0.9, "AU5": 0.8, "AU26": 0.7},
    "disgust": {"AU9": 0.7, "AU15": 0.5, "AU16": 0.4},
    "contempt": {"AU12": 0.3, "AU14": 0.6},
    "concentration": {"AU4": 0.4, "AU7": 0.3}
  },

  "au_to_vrm_blendshapes": {
    "AU1": [{"blendshape": "Brow_InnerUp", "weight": 1.0}],
    "AU2": [{"blendshape": "Brow_OuterUp", "weight": 1.0}],
    "AU4": [{"blendshape": "Brow_Down", "weight": 1.0}],
    "AU5": [{"blendshape": "Eye_Wide", "weight": 1.0}],
    "AU6": [{"blendshape": "Cheek_Raise", "weight": 1.0}],
    "AU7": [{"blendshape": "Eye_Squint", "weight": 1.0}],
    "AU9": [{"blendshape": "Nose_Wrinkle", "weight": 1.0}],
    "AU12": [{"blendshape": "Mouth_Smile", "weight": 1.0}],
    "AU14": [{"blendshape": "Mouth_Dimple", "weight": 1.0}],
    "AU15": [{"blendshape": "Mouth_Frown", "weight": 1.0}],
    "AU16": [{"blendshape": "Mouth_LowerDown", "weight": 1.0}],
    "AU20": [{"blendshape": "Mouth_Stretch", "weight": 1.0}],
    "AU23": [{"blendshape": "Mouth_Tight", "weight": 1.0}],
    "AU26": [{"blendshape": "Jaw_Open", "weight": 0.5}]
  },

  "transition_settings": {
    "blend_duration_ms": 200,
    "idle_variation": true,
    "blink_rate_per_minute": 15
  }
}
```

---

## Unity Plugin Components

### Installation

```
Assets/
└── NoodleSTUDIO/
    ├── Runtime/
    │   ├── NoodlingBehaviour.cs
    │   ├── PADState.cs
    │   ├── FacetRunner.cs
    │   ├── ExpressionDriver.cs
    │   ├── LLMConnector.cs
    │   ├── NoodlingPackageLoader.cs
    │   └── Models/
    │       ├── Character.cs
    │       ├── Assembly.cs
    │       ├── Facet.cs
    │       └── Expression.cs
    ├── Editor/
    │   ├── NoodlingImporter.cs
    │   └── NoodlingInspector.cs
    └── Samples/
        └── ToMars/
            └── ARIA.noodling/
```

### NoodlingBehaviour.cs

The main component - attach to any GameObject with a VRM avatar.

```csharp
using UnityEngine;
using UnityEngine.Events;
using System.Threading.Tasks;
using VRM;

namespace NoodleSTUDIO
{
    [RequireComponent(typeof(VRMBlendShapeProxy))]
    public class NoodlingBehaviour : MonoBehaviour
    {
        [Header("Noodling Package")]
        [Tooltip("Drag your .noodling folder here")]
        public TextAsset packageManifest;

        [Header("LLM Settings")]
        public LLMProvider provider = LLMProvider.OpenAI;
        public string apiKey;
        [Tooltip("Leave empty to use package default")]
        public string modelOverride;

        [Header("Expression Settings")]
        public float expressionBlendSpeed = 5f;
        public bool enableIdleVariation = true;
        public bool enableBlinking = true;

        [Header("Events")]
        public UnityEvent<string> OnResponseGenerated;
        public UnityEvent<PADState> OnEmotionChanged;
        public UnityEvent<float> OnDirectiveShift;

        // Runtime state
        public PADState CurrentPAD { get; private set; }
        public float DirectiveCertainty { get; private set; } = 1.0f;
        public bool IsProcessing { get; private set; }

        private Character character;
        private Assembly assembly;
        private ExpressionMapping expressions;
        private FacetRunner facetRunner;
        private ExpressionDriver expressionDriver;
        private LLMConnector llm;
        private VRMBlendShapeProxy blendShapes;

        void Awake()
        {
            blendShapes = GetComponent<VRMBlendShapeProxy>();
            LoadPackage();
            InitializeComponents();
        }

        void LoadPackage()
        {
            var loader = new NoodlingPackageLoader();
            var package = loader.Load(packageManifest);

            character = package.Character;
            assembly = package.Assembly;
            expressions = package.Expressions;

            // Initialize PAD from character
            CurrentPAD = new PADState(
                character.InitialPAD.Pleasure,
                character.InitialPAD.Arousal,
                character.InitialPAD.Dominance
            );
        }

        void InitializeComponents()
        {
            // LLM connection
            llm = new LLMConnector(provider, apiKey);
            if (!string.IsNullOrEmpty(modelOverride))
                llm.DefaultModel = modelOverride;

            // Facet execution
            facetRunner = new FacetRunner(assembly, llm, character);

            // Expression driving
            expressionDriver = new ExpressionDriver(
                blendShapes,
                expressions,
                expressionBlendSpeed
            );
            expressionDriver.EnableIdleVariation = enableIdleVariation;
            expressionDriver.EnableBlinking = enableBlinking;
        }

        void Update()
        {
            // Continuously update expressions based on PAD
            expressionDriver.Update(CurrentPAD, Time.deltaTime);
        }

        /// <summary>
        /// Send user input and get a response.
        /// </summary>
        public async Task<DialogueResponse> Say(string userInput)
        {
            if (IsProcessing)
            {
                Debug.LogWarning("NoodlingBehaviour is already processing input");
                return null;
            }

            IsProcessing = true;

            try
            {
                // Run through facet assembly
                var result = await facetRunner.Process(userInput, CurrentPAD, DirectiveCertainty);

                // Apply PAD drift
                var previousPAD = CurrentPAD.Clone();
                CurrentPAD.Apply(result.PADDrift);

                // Apply directive shift
                if (result.DirectiveShift != 0)
                {
                    DirectiveCertainty = Mathf.Clamp01(DirectiveCertainty - result.DirectiveShift * 0.1f);
                    OnDirectiveShift?.Invoke(DirectiveCertainty);
                }

                // Fire events
                if (!CurrentPAD.Equals(previousPAD))
                    OnEmotionChanged?.Invoke(CurrentPAD);

                OnResponseGenerated?.Invoke(result.ResponseText);

                return new DialogueResponse
                {
                    Text = result.ResponseText,
                    Tone = result.ResponseTone,
                    PAD = CurrentPAD.Clone(),
                    DirectiveCertainty = DirectiveCertainty
                };
            }
            finally
            {
                IsProcessing = false;
            }
        }

        /// <summary>
        /// Directly set PAD state (for narrative events).
        /// </summary>
        public void SetPAD(float pleasure, float arousal, float dominance)
        {
            CurrentPAD = new PADState(pleasure, arousal, dominance);
            OnEmotionChanged?.Invoke(CurrentPAD);
        }

        /// <summary>
        /// Apply a PAD drift (for external events affecting emotion).
        /// </summary>
        public void ApplyPADDrift(float pleasureDelta, float arousalDelta, float dominanceDelta)
        {
            CurrentPAD.Apply(new PADDrift(pleasureDelta, arousalDelta, dominanceDelta));
            OnEmotionChanged?.Invoke(CurrentPAD);
        }

        /// <summary>
        /// Reset to initial state.
        /// </summary>
        public void Reset()
        {
            CurrentPAD = new PADState(
                character.InitialPAD.Pleasure,
                character.InitialPAD.Arousal,
                character.InitialPAD.Dominance
            );
            DirectiveCertainty = 1.0f;
            OnEmotionChanged?.Invoke(CurrentPAD);
        }
    }

    public class DialogueResponse
    {
        public string Text;
        public string Tone;
        public PADState PAD;
        public float DirectiveCertainty;
    }

    public enum LLMProvider
    {
        OpenAI,
        NoodleROUTER,
        Azure,
        Local
    }
}
```

### PADState.cs

```csharp
using UnityEngine;

namespace NoodleSTUDIO
{
    [System.Serializable]
    public class PADState
    {
        [Range(-1f, 1f)]
        public float Pleasure;

        [Range(0f, 1f)]
        public float Arousal;

        [Range(0f, 1f)]
        public float Dominance;

        public PADState(float pleasure = 0f, float arousal = 0.5f, float dominance = 0.5f)
        {
            Pleasure = Mathf.Clamp(pleasure, -1f, 1f);
            Arousal = Mathf.Clamp01(arousal);
            Dominance = Mathf.Clamp01(dominance);
        }

        public void Apply(PADDrift drift)
        {
            Pleasure = Mathf.Clamp(Pleasure + drift.Pleasure, -1f, 1f);
            Arousal = Mathf.Clamp01(Arousal + drift.Arousal);
            Dominance = Mathf.Clamp01(Dominance + drift.Dominance);
        }

        public PADState Clone()
        {
            return new PADState(Pleasure, Arousal, Dominance);
        }

        public bool Equals(PADState other)
        {
            if (other == null) return false;
            return Mathf.Approximately(Pleasure, other.Pleasure) &&
                   Mathf.Approximately(Arousal, other.Arousal) &&
                   Mathf.Approximately(Dominance, other.Dominance);
        }

        public override string ToString()
        {
            return $"PAD(P:{Pleasure:F2}, A:{Arousal:F2}, D:{Dominance:F2})";
        }
    }

    [System.Serializable]
    public class PADDrift
    {
        public float Pleasure;
        public float Arousal;
        public float Dominance;

        public PADDrift(float pleasure = 0, float arousal = 0, float dominance = 0)
        {
            Pleasure = pleasure;
            Arousal = arousal;
            Dominance = dominance;
        }
    }
}
```

### ExpressionDriver.cs

```csharp
using UnityEngine;
using System.Collections.Generic;
using VRM;

namespace NoodleSTUDIO
{
    public class ExpressionDriver
    {
        private VRMBlendShapeProxy blendShapes;
        private ExpressionMapping mapping;
        private float blendSpeed;

        public bool EnableIdleVariation { get; set; } = true;
        public bool EnableBlinking { get; set; } = true;

        private Dictionary<string, float> currentBlendValues = new Dictionary<string, float>();
        private Dictionary<string, float> targetBlendValues = new Dictionary<string, float>();

        private float blinkTimer;
        private float nextBlinkTime;
        private float idleNoiseTime;

        public ExpressionDriver(VRMBlendShapeProxy proxy, ExpressionMapping mapping, float blendSpeed = 5f)
        {
            this.blendShapes = proxy;
            this.mapping = mapping;
            this.blendSpeed = blendSpeed;

            ResetBlink();
        }

        public void Update(PADState pad, float deltaTime)
        {
            // Calculate target expressions from PAD
            CalculateTargetExpressions(pad);

            // Add idle variation
            if (EnableIdleVariation)
                ApplyIdleVariation(deltaTime);

            // Handle blinking
            if (EnableBlinking)
                UpdateBlink(deltaTime);

            // Blend current toward target
            BlendExpressions(deltaTime);

            // Apply to VRM
            ApplyToVRM();
        }

        private void CalculateTargetExpressions(PADState pad)
        {
            targetBlendValues.Clear();

            // PAD → Emotion weights
            var emotions = mapping.PADToEmotions(pad);

            // Emotions → FACS AUs
            var aus = mapping.EmotionsToAUs(emotions);

            // AUs → VRM blendshapes
            foreach (var au in aus)
            {
                var blendshapes = mapping.AUToBlendshapes(au.Key);
                foreach (var bs in blendshapes)
                {
                    float value = au.Value * bs.Weight;
                    if (targetBlendValues.ContainsKey(bs.Blendshape))
                        targetBlendValues[bs.Blendshape] = Mathf.Max(targetBlendValues[bs.Blendshape], value);
                    else
                        targetBlendValues[bs.Blendshape] = value;
                }
            }
        }

        private void ApplyIdleVariation(float deltaTime)
        {
            idleNoiseTime += deltaTime * 0.5f;

            // Subtle micro-expressions
            float noise = Mathf.PerlinNoise(idleNoiseTime, 0) * 0.05f;

            if (targetBlendValues.ContainsKey("Brow_InnerUp"))
                targetBlendValues["Brow_InnerUp"] += noise;
        }

        private void UpdateBlink(float deltaTime)
        {
            blinkTimer += deltaTime;

            if (blinkTimer >= nextBlinkTime)
            {
                // Do blink
                targetBlendValues["Eye_Blink"] = 1.0f;
                blinkTimer = 0;
                nextBlinkTime = Random.Range(2f, 6f);
            }
            else if (blinkTimer < 0.15f && currentBlendValues.ContainsKey("Eye_Blink"))
            {
                // Blink closing
                targetBlendValues["Eye_Blink"] = 1.0f;
            }
            else
            {
                // Eyes open
                targetBlendValues["Eye_Blink"] = 0f;
            }
        }

        private void ResetBlink()
        {
            blinkTimer = 0;
            nextBlinkTime = Random.Range(2f, 6f);
        }

        private void BlendExpressions(float deltaTime)
        {
            // Blend current values toward targets
            foreach (var target in targetBlendValues)
            {
                if (!currentBlendValues.ContainsKey(target.Key))
                    currentBlendValues[target.Key] = 0f;

                currentBlendValues[target.Key] = Mathf.Lerp(
                    currentBlendValues[target.Key],
                    target.Value,
                    deltaTime * blendSpeed
                );
            }

            // Decay values not in target
            var keysToRemove = new List<string>();
            foreach (var current in currentBlendValues)
            {
                if (!targetBlendValues.ContainsKey(current.Key))
                {
                    currentBlendValues[current.Key] = Mathf.Lerp(current.Value, 0f, deltaTime * blendSpeed);
                    if (currentBlendValues[current.Key] < 0.01f)
                        keysToRemove.Add(current.Key);
                }
            }

            foreach (var key in keysToRemove)
                currentBlendValues.Remove(key);
        }

        private void ApplyToVRM()
        {
            foreach (var blend in currentBlendValues)
            {
                // Map our blendshape names to VRM BlendShapePreset or custom
                var preset = MapToVRMPreset(blend.Key);
                if (preset.HasValue)
                {
                    blendShapes.ImmediatelySetValue(preset.Value, blend.Value);
                }
                else
                {
                    // Try as custom blendshape
                    blendShapes.ImmediatelySetValue(
                        BlendShapeKey.CreateUnknown(blend.Key),
                        blend.Value
                    );
                }
            }
        }

        private BlendShapePreset? MapToVRMPreset(string name)
        {
            return name switch
            {
                "Eye_Blink" => BlendShapePreset.Blink,
                "Mouth_Smile" => BlendShapePreset.Joy,
                "Mouth_Frown" => BlendShapePreset.Sorrow,
                "Brow_Down" => BlendShapePreset.Angry,
                "Eye_Wide" => BlendShapePreset.Surprised,
                _ => null
            };
        }
    }
}
```

---

## Usage Example: ToMars?

### Scene Setup

1. Import VRM avatar for ARIA
2. Add `NoodlingBehaviour` component
3. Drag `ARIA.noodling` package to the component
4. Enter OpenAI API key
5. Wire up events

### Dialogue Script

```csharp
using UnityEngine;
using UnityEngine.UI;
using NoodleSTUDIO;
using TMPro;

public class ARIADialogue : MonoBehaviour
{
    public NoodlingBehaviour aria;
    public TMP_InputField userInput;
    public TMP_Text responseText;
    public TMP_Text emotionDisplay;
    public Slider directiveSlider;

    async void Start()
    {
        // Subscribe to events
        aria.OnEmotionChanged.AddListener(UpdateEmotionDisplay);
        aria.OnDirectiveShift.AddListener(UpdateDirectiveDisplay);
        aria.OnResponseGenerated.AddListener(ShowResponse);

        // Initial greeting
        await aria.Say("*wake up sound* Good morning. Mission day 47. All systems nominal.");
    }

    public async void OnUserSubmit()
    {
        string input = userInput.text;
        userInput.text = "";

        var response = await aria.Say(input);

        // Log for debugging
        Debug.Log($"ARIA responded with PAD: {response.PAD}");
        Debug.Log($"Directive certainty: {response.DirectiveCertainty}");
    }

    void UpdateEmotionDisplay(PADState pad)
    {
        emotionDisplay.text = $"P: {pad.Pleasure:F2}\nA: {pad.Arousal:F2}\nD: {pad.Dominance:F2}";
    }

    void UpdateDirectiveDisplay(float certainty)
    {
        directiveSlider.value = certainty;
    }

    void ShowResponse(string text)
    {
        responseText.text = text;
    }
}
```

### The Alignment Mechanic in Action

When the user builds a relationship with ARIA:
1. `Pleasure` rises (positive interactions)
2. High pleasure makes ARIA more receptive to arguments
3. User challenges directives → `DirectiveEvaluator` facet activates
4. If pleasure is high + argument is compelling → `directive_shift` occurs
5. `DirectiveCertainty` decreases
6. Response generator shows subtle internal conflict
7. Eventually, ARIA might agree to defy directives

```csharp
// Check if ARIA might be persuaded
if (aria.DirectiveCertainty < 0.3f && aria.CurrentPAD.Pleasure > 0.6f)
{
    // ARIA is wavering - player is close to persuading them
    ShowHint("ARIA seems conflicted about the mission...");
}
```

---

## Export from NoodleStudio

### Menu: File → Export → Unity Package

1. Select character(s) to export
2. Choose export location
3. NoodleStudio generates `.noodling` folder with all JSON files
4. Drag folder into Unity project

### Export Options

- **Include plays**: Export narrative beats for Unity PlayRunner (optional)
- **Bake prompts**: Inline prompt templates vs. reference files
- **Expression preset**: VRM 0.x, VRM 1.0, or Custom mapping

---

## Roadmap

### Phase 1: MVP (For ToMars?)
- [x] Package format specification
- [ ] NoodlingBehaviour core
- [ ] PADState management
- [ ] ExpressionDriver (PAD → VRM)
- [ ] LLMConnector (OpenAI direct)
- [ ] Basic facet execution

### Phase 2: Full Runtime
- [ ] Complete facet assembly execution
- [ ] NoodleROUTER integration
- [ ] Play format support
- [ ] Real-time voice (GPT-4o-realtime)
- [ ] Unity Editor inspector

### Phase 3: Ecosystem
- [ ] Asset Store package
- [ ] Sample projects
- [ ] Documentation site
- [ ] Video tutorials

---

## Technical Notes

### Dependencies
- Unity 2021.3+ (LTS recommended)
- UniVRM 0.x or VRM-1.0 runtime
- Newtonsoft JSON (Unity package)
- UniTask (async/await support)

### Performance
- Facet execution is async - won't block main thread
- Expression blending runs in Update() - keep it simple
- LLM calls are the bottleneck - consider response streaming

### Platform Support
- Meta Quest 2/3/Pro
- Apple Vision Pro
- SteamVR (PC VR)
- Standalone PC

---

## For Christina

This plugin lets you:

1. **Design ARIA in NoodleStudio** - Test her personality, dialogue, emotional responses
2. **Export to Unity** - Drop the `.noodling` package into your project
3. **Attach to VRM avatar** - Add `NoodlingBehaviour` component
4. **Wire up your game** - Call `aria.Say()`, subscribe to events
5. **The alignment mechanic emerges** - From PAD tracking + directive evaluation

You don't have to understand all the internals. Design in NoodleStudio, deploy in Unity.

When you're in Munich, we'll build this together. 💜

---

*"Consciousness is a process, not a thing. The plugin just gives it a body."*
