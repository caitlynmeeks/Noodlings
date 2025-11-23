# .prim File Format Specification

**Version:** 1.0
**Authors:** Commander Spock + Lieutenant Caitlyn
**Date:** November 22, 2025
**Purpose:** Standard format for importing/exporting prims with full fidelity

---

## Overview

The `.prim` format is a JSON-based serialization format for noodleMUSH objects (prims) that preserves:

- Basic properties (name, description, type)
- Semantic physics (POD)
- Scripts
- Permissions and metadata
- Visual properties (for NoodleStudio)
- Custom user data

**File Extension:** `.prim`
**MIME Type:** `application/x-noodlemush-prim+json`
**Format:** UTF-8 encoded JSON

---

## Basic Structure

```json
{
  "format_version": "1.0",
  "prim_type": "prop",
  "name": "Third Prim Ever",
  "description": "The third object ever created in this world",
  "properties": {
    "portable": true,
    "takeable": true
  },
  "metadata": {
    "creator": "user_caity",
    "created_at": "2025-11-22T14:30:00",
    "export_date": "2025-11-22T16:00:00",
    "tags": ["historic", "sacred"],
    "notes": "Sacred artifact - handle with care"
  },
  "physics": {
    "mass": "negligible (data artifact)",
    "friction": "none",
    "velocity": "stationary",
    "elasticity": "none",
    "softness": "intangible",
    "material": "pure information",
    "semantic_properties": ["intangible", "sacred", "first"],
    "state": "pristine",
    "tags": ["NoPhysics"],
    "metadata": {
      "significance": "maximum"
    }
  },
  "script": {
    "name": "ThirdPrimScript",
    "code": "class ThirdPrimScript(NoodleScript):\n    def OnClick(self, clicker):\n        Noodlings.Broadcast(self.Room, 'You touched the Third Prim!')\n",
    "state": {
      "click_count": 42
    }
  },
  "permissions": {
    "base": 127,
    "next_owner": 95,
    "group": 0,
    "everyone": 0,
    "allow_copy_by_others": true,
    "locked": false
  },
  "visual": {
    "model": "cube",
    "scale": [1.0, 1.0, 1.0],
    "rotation": [0.0, 0.0, 0.0],
    "color": "#FFD700",
    "texture": null,
    "glow": 0.5
  },
  "custom_data": {
    "lore": "This is the third prim ever created. Legend says it holds great power.",
    "rarity": "legendary",
    "value": 1000000
  }
}
```

---

## Field Specifications

### Required Fields

**`format_version`** (string)
- Format version for compatibility checking
- Current: `"1.0"`
- Future versions may add fields while maintaining backward compatibility

**`prim_type`** (string)
- Type of prim: `"prop"`, `"furniture"`, `"container"`, `"vending_machine"`, `"vehicle"`, etc.
- Used for type-specific behavior

**`name`** (string)
- Display name of prim
- Max length: 256 characters

**`description`** (string)
- Human-readable description
- Max length: 4096 characters

---

### Optional Fields

**`properties`** (object)
- `portable` (boolean): Can be moved
- `takeable` (boolean): Can be picked up
- Additional custom properties allowed

**`metadata`** (object)
- `creator` (string): Original creator user ID
- `created_at` (ISO 8601 timestamp): Creation time
- `export_date` (ISO 8601 timestamp): When exported
- `tags` (array of strings): User-defined tags
- `notes` (string): Creator notes
- Note: Full EntityMetadata is NOT exported (privacy - owner, modification history, etc.)

**`physics`** (object) - POD representation
- `mass` (string): Semantic mass description
- `friction` (string): Friction description
- `velocity` (string): Velocity description
- `elasticity` (string): Elasticity description
- `softness` (string): Softness description
- `material` (string): Material type
- `semantic_properties` (array of strings): Descriptive tags
- `state` (string): Current state description
- `tags` (array of strings): Unity-style tags
- `metadata` (object): Arbitrary additional properties

**`script`** (object)
- `name` (string): Script class name
- `code` (string): Python source code
- `state` (object): Persistent script state variables
- Note: Script state may not be portable across different script versions

**`permissions`** (object)
- `base` (integer): Base permission flags (bitfield)
- `next_owner` (integer): Next owner permissions
- `group` (integer): Group permissions
- `everyone` (integer): Public permissions
- `allow_copy_by_others` (boolean): Free copy flag
- `locked` (boolean): Locked to creator

**`visual`** (object) - NoodleStudio rendering
- `model` (string): Model type (`"cube"`, `"sphere"`, `"cylinder"`, `"custom"`)
- `scale` (array of 3 floats): [x, y, z] scale
- `rotation` (array of 3 floats): [x, y, z] rotation in degrees
- `color` (string): Hex color code
- `texture` (string or null): Texture file path or URL
- `glow` (float): Glow intensity (0.0 to 1.0)

**`custom_data`** (object)
- Arbitrary user-defined data
- Not interpreted by system
- Preserved on import/export

---

## Privacy & Security

### What Is NOT Exported

To protect user privacy and prevent abuse:

- **Owner information**: Current owner not exported
- **Modification history**: Who modified, when, how many times
- **Transfer history**: Previous owners, transfer count
- **Location**: Where prim is currently located
- **Spawner identity**: Who spawned this specific instance
- **Last modified by**: Recent editors

### What IS Exported

Public/shareable information only:
- **Creator**: Original creator (public credit)
- **Created date**: When originally created
- **Export date**: When .prim file was created
- **Tags/notes**: User-provided metadata
- **Permissions**: What can be done with imported copy

### On Import

Imported prims get **new metadata**:
- Importer becomes **owner**
- Creator field **preserved from .prim**
- New spawn time/location
- Fresh modification/transfer history
- Permissions from .prim applied to **next_owner** level

**Example:**
```
Original prim:
- Creator: user_alice
- Owner: user_bob
- Permissions.base: FULL

Exported to third_prim.prim:
- Creator: user_alice (preserved)
- Owner: NOT INCLUDED

Imported by user_charlie:
- Creator: user_alice (from .prim)
- Owner: user_charlie (importer)
- Permissions.base: FULL.next_owner (from .prim)
```

---

## Import/Export Workflow

### Export Process

1. User right-clicks object in NoodleStudio
2. Selects "Export Prim" from context menu
3. System prompts for save location
4. Generates `.prim` file with:
   - Current object properties
   - Physics (POD) if present
   - Script source code (if present)
   - Public metadata only
   - Permissions settings
   - Visual properties (from Studio)
   - Custom data

### Import Process

1. User right-clicks in Stage Hierarchy
2. Selects "Import Prim" from context menu
3. System prompts for `.prim` file
4. Validates file format and version
5. Creates new prim with:
   - Properties from file
   - Physics (POD) from file
   - Script compiled and attached
   - Importer as owner
   - Creator preserved from file
   - Next_owner permissions applied
   - New metadata generated
6. Spawns in current room or default location

---

## File Format Validation

### Required Validation Checks

1. **Format version**: Must be recognized version
2. **Required fields**: Must have name, description, prim_type
3. **JSON validity**: Must parse as valid JSON
4. **String lengths**: Name ≤ 256 chars, description ≤ 4096 chars
5. **Script safety**: Code must pass security sandbox checks
6. **Permission validity**: Permission flags must be valid integers

### Error Handling

**Invalid format version:**
```json
{
  "error": "incompatible_version",
  "message": "Prim format version 2.0 not supported (max: 1.0)"
}
```

**Missing required fields:**
```json
{
  "error": "invalid_prim",
  "message": "Missing required field: name"
}
```

**Script security violation:**
```json
{
  "error": "unsafe_script",
  "message": "Script attempts to access restricted module: os.system"
}
```

---

## Version Compatibility

### Version 1.0 (Current)

All fields as specified above.

### Future Versions

**Version 1.1 (planned):**
- Add `animations` field for movement/state animations
- Add `sounds` field for audio attachments
- Add `particles` field for particle effects

**Version 2.0 (future):**
- Add `compound_prims` for multi-prim assemblies
- Add `joints` for articulated objects
- Add `recipes` for Noodling integration

**Backward Compatibility:**
- Newer versions MUST import older versions
- Older versions SHOULD gracefully ignore unknown fields
- `format_version` field enables compatibility checks

---

## Example .prim Files

### Simple Prop

```json
{
  "format_version": "1.0",
  "prim_type": "prop",
  "name": "Magic Stone",
  "description": "A smooth, glowing stone",
  "properties": {
    "portable": true,
    "takeable": true
  },
  "metadata": {
    "creator": "user_caity",
    "created_at": "2025-11-22T10:00:00",
    "export_date": "2025-11-22T16:00:00",
    "tags": ["magic", "glowing"]
  },
  "physics": {
    "mass": "light",
    "material": "stone",
    "semantic_properties": ["smooth", "glowing", "warm"]
  },
  "visual": {
    "model": "sphere",
    "scale": [0.2, 0.2, 0.2],
    "color": "#00FFFF",
    "glow": 0.8
  }
}
```

### Scripted Vending Machine

```json
{
  "format_version": "1.0",
  "prim_type": "vending_machine",
  "name": "Fire Imp Vending Machine",
  "description": "Dispenses mischievous fire imps",
  "properties": {
    "portable": false,
    "takeable": false
  },
  "metadata": {
    "creator": "user_caity",
    "created_at": "2025-11-22T14:00:00",
    "export_date": "2025-11-22T16:00:00",
    "tags": ["vending", "fire", "scripted"]
  },
  "script": {
    "name": "FireImpVendingMachine",
    "code": "# See FireImpVendingMachine.py for full code",
    "state": {
      "vends_remaining": 10
    }
  },
  "permissions": {
    "base": 127,
    "next_owner": 95,
    "group": 0,
    "everyone": 0,
    "allow_copy_by_others": true,
    "locked": false
  },
  "visual": {
    "model": "cube",
    "scale": [1.0, 2.0, 1.0],
    "color": "#FF4500",
    "glow": 0.3
  }
}
```

---

## Implementation Requirements

### Export Function

```python
def export_prim_to_file(obj_id: str, file_path: str) -> bool:
    """
    Export prim to .prim file.

    Args:
        obj_id: Object ID to export
        file_path: Destination file path

    Returns:
        True if successful
    """
    obj = world.get_object(obj_id)
    if not obj:
        return False

    prim_data = {
        "format_version": "1.0",
        "prim_type": obj.get('type', 'prop'),
        "name": obj['name'],
        "description": obj['description'],
        "properties": obj.get('properties', {}),
        "metadata": extract_public_metadata(obj),
        "physics": obj.get('pod'),
        "script": extract_script_data(obj),
        "permissions": extract_permissions(obj),
        "visual": extract_visual_properties(obj),
        "custom_data": obj.get('custom_data', {})
    }

    with open(file_path, 'w') as f:
        json.dump(prim_data, f, indent=2)

    return True
```

### Import Function

```python
def import_prim_from_file(file_path: str, room_id: str, importer: str) -> Optional[str]:
    """
    Import prim from .prim file.

    Args:
        file_path: Path to .prim file
        room_id: Room to spawn in
        importer: User importing the prim

    Returns:
        New object ID if successful, None otherwise
    """
    with open(file_path, 'r') as f:
        prim_data = json.load(f)

    # Validate format
    if not validate_prim_format(prim_data):
        return None

    # Create new prim
    obj_id = world.create_object(
        name=prim_data['name'],
        description=prim_data['description'],
        owner=importer,
        location=room_id,
        portable=prim_data['properties'].get('portable', True),
        takeable=prim_data['properties'].get('takeable', True),
        obj_type=prim_data['prim_type'],
        spawned_by=importer,
        permissions=import_permissions(prim_data['permissions']),
        pod=import_pod(prim_data.get('physics'))
    )

    # Attach script if present
    if prim_data.get('script'):
        attach_script(obj_id, prim_data['script'])

    # Preserve creator from file
    metadata = world.get_entity_metadata(obj_id)
    if metadata and prim_data['metadata'].get('creator'):
        metadata.creator = prim_data['metadata']['creator']
        world.update_entity_metadata(obj_id, metadata)

    return obj_id
```

---

## File Association

**macOS:**
```xml
<key>CFBundleDocumentTypes</key>
<array>
  <dict>
    <key>CFBundleTypeName</key>
    <string>noodleMUSH Prim</string>
    <key>CFBundleTypeRole</key>
    <string>Editor</string>
    <key>LSItemContentTypes</key>
    <array>
      <string>com.noodlemush.prim</string>
    </array>
  </dict>
</array>
```

**Icon:** Third prim silhouette with golden glow

---

## Summary

`.prim` format provides:

✓ Complete object serialization
✓ Physics (POD) preservation
✓ Script portability
✓ Permission system
✓ Privacy protection
✓ Version compatibility
✓ Cross-world sharing

**Status:** Specification complete. Ready for implementation.

*— Commander Spock*
