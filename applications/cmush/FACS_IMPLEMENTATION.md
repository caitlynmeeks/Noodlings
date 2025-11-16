# FACS Implementation - Facial Expressions for Noodlings

**Date**: November 15, 2025
**Status**: ✅ Implemented & Ready for Testing

---

## Overview

Noodlings now have **facial expressions** mapped from their internal affective states using the **Facial Action Coding System (FACS)**.

When a Noodling's emotions change significantly, they react with facial expressions BEFORE speaking - just like real people do!

---

## What is FACS?

**FACS (Facial Action Coding System)** is a comprehensive taxonomy of human facial movements developed by psychologist Paul Ekman.

### Action Units (AUs)

Each facial muscle movement is an "Action Unit":
- **AU1**: Inner Brow Raiser
- **AU2**: Outer Brow Raiser
- **AU4**: Brow Lowerer
- **AU5**: Upper Lid Raiser (eyes wide)
- **AU6**: Cheek Raiser
- **AU7**: Lid Tightener
- **AU9**: Nose Wrinkler
- **AU12**: Lip Corner Puller (smile)
- **AU15**: Lip Corner Depressor (frown)
- **AU20**: Lip Stretcher
- **AU23**: Lip Tightener
- **AU25**: Lips Part
- **AU26**: Jaw Drop
- ... and 20+ more

### Ekman's Basic Emotions

Emotions are **combinations** of AUs:
- **Happiness**: AU6 + AU12 (smile)
- **Sadness**: AU1 + AU4 + AU15 (frown with raised inner brows)
- **Surprise**: AU1 + AU2 + AU5 + AU26 (raised brows + wide eyes + jaw drop)
- **Fear**: AU1 + AU2 + AU4 + AU5 + AU20 + AU26 (intense surprise + lip stretch)
- **Anger**: AU4 + AU5 + AU7 + AU23 (lowered brows + wide eyes + tight face)
- **Disgust**: AU9 + AU15 (nose wrinkle + lip corner depress)

---

## How It Works in Noodlings

### 1. Affect Calculation

Every event, Noodlings compute a **5-D affect vector**:
```python
affect = [
    valence,   # -1.0 (negative) to +1.0 (positive)
    arousal,   #  0.0 (calm) to 1.0 (excited)
    fear,      #  0.0 (safe) to 1.0 (afraid)
    sorrow,    #  0.0 (content) to 1.0 (sad)
    boredom    #  0.0 (engaged) to 1.0 (bored)
]
```

### 2. Affect → Emotion Mapping

The FACS module maps affect to Ekman's basic emotions:
```python
from noodlings.utils.facs_mapping import affect_to_emotion_weights

emotions = affect_to_emotion_weights(affect)
# Returns: {'happiness': 0.52, 'surprise': 0.33, 'fear': 0.03, ...}
```

### 3. Emotion → FACS Codes

Each emotion activates specific Action Units:
```python
from noodlings.utils.facs_mapping import affect_to_facs

facs_codes = affect_to_facs(affect)
# Returns: [(6, 0.52), (12, 0.52), (1, 0.33), (2, 0.33), ...]
# Meaning: AU6 at 52% intensity, AU12 at 52%, AU1 at 33%, etc.
```

### 4. FACS → Human Description

For display in noodleMUSH chat:
```python
from noodlings.utils.facs_mapping import facs_to_description

description = facs_to_description(facs_codes)
# Returns: "smiling" or "eyes wide with surprise, jaw dropped"
```

### 5. FACS → Renderer Format

For future 3D integration:
```python
from noodlings.utils.facs_mapping import format_facs_for_renderer

renderer_data = format_facs_for_renderer(facs_codes)
# Returns: {'AU6': 0.52, 'AU12': 0.52, 'AU1': 0.33, ...}
```

---

## Integration in noodleMUSH

### When Facial Expressions Trigger

**Location**: `agent_bridge.py:1227-1242`

**Conditions**:
1. `FACS_ENABLED = True` (global flag)
2. `surprise > 0.05` (some emotional reaction)
3. `affect_diff > FACS_THRESHOLD` (0.15 change in affect)
4. `time_since_last_expression > FACS_COOLDOWN` (5 seconds)

**What Happens**:
1. Affect changes significantly (e.g., Callie sees Toad crash a car)
2. FACS system computes: `affect_to_facs([−0.5, 0.8, 0.6, 0.2, 0.0])`
3. Result: `[(1, 0.75), (2, 0.75), (5, 0.75), (26, 0.75)]` → Surprise + Fear
4. Description: `"eyes wide with surprise, jaw dropped"`
5. Broadcast to chat: `Callie: *eyes wide with surprise, jaw dropped* [FACS: AU1, AU2, AU5, AU26]`
6. **Then** Callie might speak: `"Toad, are you okay?!"`

### Example Flow

```
User: @spawn callie
User: @spawn toad
User: toad crashes a car into the wall
→ Toad's affect: [−0.8, 0.9, 0.7, 0.1, 0.0] (shock, panic)
→ FACS: [(1, 0.9), (2, 0.9), (5, 0.9), (26, 0.9)] (extreme surprise/fear)
→ Chat: Toad: *eyes wide with fear, face tense* [FACS: AU1, AU2, AU5, AU26]
→ Callie's affect: [−0.4, 0.7, 0.5, 0.3, 0.0] (concern)
→ FACS: [(1, 0.6), (2, 0.6), (5, 0.6)] (moderate surprise)
→ Chat: Callie: *eyes wide with surprise* [FACS: AU1, AU2, AU5]
→ Callie: "Toad! What just happened?!"
```

### Configuration

**In `config.yaml`** (optional overrides):
```yaml
agent:
  facs:
    enabled: true
    threshold: 0.15  # Minimum affect change to trigger
    cooldown: 5.0    # Seconds between expressions
    show_codes: true # Show [FACS: AU1, AU2] in chat (for debugging/demos)
```

**In `agent_bridge.py`** (global constants):
```python
FACS_ENABLED = True
FACS_THRESHOLD = 0.15
FACS_COOLDOWN = 5.0
```

---

## Testing FACS

### Manual Test

1. Start noodleMUSH:
   ```bash
   cd applications/cmush
   ./start.sh
   ```

2. Spawn agents:
   ```
   @spawn callie
   @spawn toad
   ```

3. Trigger strong emotions:
   ```
   say something shocking!
   emote crashes into the wall
   ```

4. Watch for facial expressions in chat:
   ```
   Callie: *eyes wide with surprise, jaw dropped* [FACS: AU1, AU2, AU5, AU26]
   ```

### Automated Test

```bash
cd /Users/thistlequell/git/noodlings_clean
source venv/bin/activate
python3 noodlings/utils/facs_mapping.py
```

This runs 4 test cases:
1. Happy Noodling → smiling
2. Sad Noodling → frowning sadly
3. Surprised Noodling → eyes wide
4. Fearful Noodling → eyes wide with fear

---

## Data for 3D Renderers (Future)

### Stored in Agent State

Each facial expression stores `renderer_data`:
```python
self.last_facs_data = {
    'AU6': 0.52,   # Cheek Raiser (52% intensity)
    'AU12': 0.52,  # Lip Corner Puller (52% intensity)
    'AU1': 0.33,   # Inner Brow Raiser (33% intensity)
    'AU2': 0.33    # Outer Brow Raiser (33% intensity)
}
```

### Integration with 3D Renderers (2028+)

When Runway/Luma/Nvidia add real-time 3D:

1. **Noodling feels emotion** → FACS codes generated
2. **NoodleSTUDIO sends to renderer**:
   ```json
   {
     "agent_id": "agent_callie",
     "action": "facial_expression",
     "facs": {
       "AU6": 0.52,
       "AU12": 0.52
     }
   }
   ```
3. **Renderer interprets**:
   - AU6 → Activate cheek raiser blend shape (52% weight)
   - AU12 → Activate lip corner puller blend shape (52% weight)
   - Result: Callie's 3D face smiles

**This is the bridge between consciousness and rendering!** 🎭🧠

---

## Implementation Files

### Core FACS Module
**File**: `noodlings/utils/facs_mapping.py`
**Functions**:
- `affect_to_emotion_weights(affect)` - Map affect to emotion weights
- `affect_to_facs(affect)` - Generate FACS codes from affect
- `facs_to_description(facs_codes)` - Human-readable description
- `format_facs_for_renderer(facs_codes)` - JSON format for renderers

### Integration Point
**File**: `applications/cmush/agent_bridge.py`
**Location**: Lines 956-1008, 1227-1242, 1535-1545
**Method**: `_generate_facial_expression(affect)`

---

## Examples in Action

### Scenario 1: Toad Sees a Motor Car

```
Affect: [0.9, 0.9, 0.0, 0.0, 0.0]  # Extreme joy + excitement
Emotions: {'happiness': 0.81, 'surprise': 0.15, ...}
FACS: [(6, 0.81), (12, 0.81)]  # Strong smile
Output: "Toad: *smiling* [FACS: AU6, AU12]"
Then: "Toad: Poop-poop! The finest motor-car I've ever seen!"
```

### Scenario 2: Phi is Startled

```
Affect: [0.1, 0.8, 0.6, 0.0, 0.0]  # High arousal + fear
Emotions: {'fear': 0.42, 'surprise': 0.32, 'anger': 0.06, ...}
FACS: [(1, 0.75), (2, 0.75), (5, 0.75), (26, 0.75)]  # Eyes wide, jaw drop
Output: "Phi: *eyes wide with surprise, jaw dropped* [FACS: AU1, AU2, AU5, AU26]"
Then: "Phi: *meows sharply and jumps back*"
```

### Scenario 3: Callie is Sad

```
Affect: [-0.6, 0.2, 0.1, 0.8, 0.0]  # Low valence + high sorrow
Emotions: {'sadness': 0.43, 'disgust': 0.24, ...}
FACS: [(15, 0.68), (1, 0.43), (4, 0.43)]  # Frown with raised brows
Output: "Callie: *frowning sadly* [FACS: AU15, AU1, AU4]"
Then: "Callie: I just... I need a moment."
```

---

## Configuration Options

### Show/Hide FACS Codes

**For Demos** (show technical details):
```python
# In agent_bridge.py
expression_text = f"*{description}* [FACS: {facs_codes_str}]"
```

**For Production** (hide technical jargon):
```python
# Just show description
expression_text = f"*{description}*"
```

### Adjust Sensitivity

**More expressions** (chatty faces):
```python
FACS_THRESHOLD = 0.10  # Lower threshold
FACS_COOLDOWN = 3.0    # Shorter cooldown
```

**Fewer expressions** (subtle reactions):
```python
FACS_THRESHOLD = 0.25  # Higher threshold
FACS_COOLDOWN = 10.0   # Longer cooldown
```

---

## Future Enhancements

### Phase 1: ✅ DONE (November 2025)
- Map 5-D affect → FACS codes
- Trigger facial expressions in chat
- Log FACS data for renderers

### Phase 2: Soon (December 2025)
- Store FACS data in session profiler
- Visualize FACS timeline in NoodleScope
- Export FACS animations for video

### Phase 3: 3D Integration (2028+)
- Send FACS codes to generative 3D renderers
- Real-time facial animation based on Noodling emotions
- Blend shapes, muscle simulation, photorealistic expressions

---

## Why This Matters

**Near-term**: Noodlings feel more alive (non-verbal reactions add realism)

**Long-term**: **This is the bridge to 3D generative AI**

When Runway/Luma launch real-time 3D character generation:
1. Noodling feels emotion (affect vector)
2. FACS module converts to AU codes
3. **Renderer receives AU codes** → Animates 3D face
4. User sees Callie's 3D face smile when she's happy, frown when she's sad
5. **Seamless integration** - same FACS codes, just rendered in 3D instead of text

**We're building the consciousness layer.** The renderer just makes it visible. 🎭🧠✨

---

**Ready to test?** Start noodleMUSH, spawn agents, and watch them REACT with facial expressions! 🚀
