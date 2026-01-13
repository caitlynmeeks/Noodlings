# .noodling Package Format Specification

Version: 1.0.0
Date: 2026-01-13

---

## Overview

A `.noodling` package is a folder containing JSON files that fully describe a noodling character for deployment in game engines. The format is engine-agnostic; engine-specific plugins interpret the package.

---

## Folder Structure

```
{name}.noodling/
├── manifest.json       # REQUIRED - Package metadata
├── character.json      # REQUIRED - Personality and PAD state
├── assembly.json       # REQUIRED - Facet configuration
├── expressions.json    # REQUIRED - Expression mappings
└── plays/              # OPTIONAL - Narrative beats
    ├── intro.play.json
    └── ending.play.json
```

---

## manifest.json

Package metadata and file references.

```json
{
  "name": "string",
  "version": "semver",
  "noodlestudio_version": "semver",
  "description": "string",
  "author": "string",
  "created": "ISO 8601 datetime",
  "exports": {
    "character": "character.json",
    "assembly": "assembly.json",
    "expressions": "expressions.json",
    "plays": "plays/"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | Display name |
| version | string | Yes | Semantic version |
| noodlestudio_version | string | Yes | Exporter version |
| description | string | No | Character description |
| author | string | No | Creator name |
| created | string | Yes | ISO 8601 timestamp |
| exports | object | Yes | File path mapping |

---

## character.json

Personality, motivation, and initial emotional state.

```json
{
  "id": "string",
  "name": "string",
  "full_name": "string",
  "role": "string",
  "initial_pad": {
    "pleasure": "float -1 to 1",
    "arousal": "float 0 to 1",
    "dominance": "float 0 to 1"
  },
  "motivation": "string",
  "personality_traits": ["string"],
  "voice": {
    "tone": "string",
    "vocalizations": ["string"]
  },
  "backstory": "string"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | Yes | Unique identifier |
| name | string | Yes | Display name |
| full_name | string | No | Formal name |
| role | string | No | Character role/type |
| initial_pad | object | Yes | Starting emotional state |
| motivation | string | No | Character motivation |
| personality_traits | array | No | Trait keywords |
| voice | object | No | Speech characteristics |
| backstory | string | No | Background story |

### initial_pad

The PAD (Pleasure-Arousal-Dominance) model:

| Field | Range | Description |
|-------|-------|-------------|
| pleasure | -1.0 to 1.0 | Positive/negative valence |
| arousal | 0.0 to 1.0 | Activation level |
| dominance | 0.0 to 1.0 | Control/submission |

---

## assembly.json

Facet assembly defining the character's cognition architecture.

```json
{
  "id": "string",
  "name": "string",
  "version": "semver",
  "facets": [
    {
      "id": "string",
      "name": "string",
      "type": "FACET_TYPE",
      "description": "string",
      "prompt_template": "string (LLM only)",
      "model": "string (LLM only)",
      "inputs": ["string"],
      "outputs": ["string"]
    }
  ],
  "connections": [
    {
      "from": "facet_id.pad_name",
      "to": "facet_id.pad_name"
    }
  ],
  "prompt_templates": {
    "facet_id.prompt": "string"
  }
}
```

### Facet Types

| Type | Description |
|------|-------------|
| INCOMING | Entry point for input |
| OUTGOING | Exit point for output |
| LLM | Language model processing |
| CHARM_NETWORK | Temporal affect processing |
| CONTEXT_INTELLIGENCE | Social/context analysis |
| CONVERGENCE | Merge multiple streams |

### Connection Format

Connections use the format `{facet_id}.{pad_name}`:

```json
{
  "from": "context_intel.output",
  "to": "embodiment.input"
}
```

---

## expressions.json

Complete PAD to facial expression mapping chain.

```json
{
  "mapping_version": "1.0.0",
  "avatar_type": "VRM | VRM1 | CUSTOM",

  "pad_to_emotion_weights": {
    "emotion_name": {
      "pleasure": "float",
      "arousal": "float",
      "dominance": "float"
    }
  },

  "emotion_to_aus": {
    "emotion_name": {
      "AU_NUMBER": "float 0 to 1"
    }
  },

  "au_to_vrm_blendshapes": {
    "AU_NUMBER": [
      {
        "blendshape": "string",
        "weight": "float 0 to 1"
      }
    ]
  },

  "transition_settings": {
    "blend_duration_ms": "integer",
    "idle_variation": "boolean",
    "blink_rate_per_minute": "integer"
  }
}
```

### Standard Emotions

| Emotion | PAD Profile |
|---------|-------------|
| joy | High pleasure, moderate arousal |
| sadness | Low pleasure, low arousal |
| anger | Low pleasure, high arousal, high dominance |
| fear | Low pleasure, high arousal, low dominance |
| surprise | Neutral pleasure, high arousal |
| disgust | Low pleasure, low-moderate arousal |
| contempt | Low pleasure, high dominance |
| concentration | Neutral pleasure, moderate arousal |

### FACS Action Units

Standard Ekman FACS action units:

| AU | Name | VRM Blendshape |
|----|------|----------------|
| AU1 | Inner Brow Raiser | Brow_InnerUp |
| AU2 | Outer Brow Raiser | Brow_OuterUp |
| AU4 | Brow Lowerer | Brow_Down |
| AU5 | Upper Lid Raiser | Eye_Wide |
| AU6 | Cheek Raiser | Cheek_Raise |
| AU7 | Lid Tightener | Eye_Squint |
| AU9 | Nose Wrinkler | Nose_Wrinkle |
| AU12 | Lip Corner Puller | Mouth_Smile |
| AU14 | Dimpler | Mouth_Dimple |
| AU15 | Lip Corner Depressor | Mouth_Frown |
| AU16 | Lower Lip Depressor | Mouth_LowerDown |
| AU20 | Lip Stretcher | Mouth_Stretch |
| AU23 | Lip Tightener | Mouth_Tight |
| AU26 | Jaw Drop | Jaw_Open |

---

## plays/{name}.play.json

Narrative beats for guided performances.

```json
{
  "name": "string",
  "version": "string",
  "characters": {
    "character_id": {
      "voice": "string",
      "initial_pad": {
        "pleasure": "float",
        "arousal": "float",
        "dominance": "float"
      }
    }
  },
  "beats": [
    {
      "id": "string",
      "character": "string",
      "speaks": "string",
      "pad_drift": {
        "pleasure": "float delta",
        "arousal": "float delta",
        "dominance": "float delta"
      },
      "computer_use": "object or null",
      "wait_after": "float seconds"
    }
  ]
}
```

---

## File Encoding

- All files: UTF-8
- JSON: Pretty-printed with 2-space indent
- Line endings: LF (Unix-style)

---

## Validation

Packages should validate:

1. All required files present
2. Valid JSON syntax
3. Required fields in each file
4. PAD values within range
5. Facet IDs in connections exist
6. AU references exist in mappings

---

## See Also

- [Unity Integration](unity.md)
- [Integration Overview](overview.md)
