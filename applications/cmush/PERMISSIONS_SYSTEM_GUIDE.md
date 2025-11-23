# Permissions System Guide

**Second Life-style Permissions for noodleMUSH**

**Authors:** Commander Spock + Lieutenant Caitlyn
**Date:** November 22, 2025
**Status:** Fully implemented and operational

---

## Overview

Every prim (object) and Noodling (agent) in noodleMUSH now has comprehensive metadata tracking:

✓ **Creator**: Original creator (never changes)
✓ **Owner**: Current owner (can change via transfer)
✓ **Spawned by**: User who spawned this instance
✓ **Spawned at**: Timestamp when spawned
✓ **Spawned in**: Initial room location
✓ **Permissions**: Second Life-style permission flags
✓ **Modification history**: Who changed what and when
✓ **Transfer history**: Previous owners

---

## Permission Flags

Based on Second Life's permission model:

```python
class Permission(Flag):
    NONE = 0
    MODIFY = auto()    # Can edit properties, scripts, description
    COPY = auto()      # Can duplicate the object
    TRANSFER = auto()  # Can give to another user
    DELETE = auto()    # Can delete the object
    MOVE = auto()      # Can move to different location
    SCRIPT = auto()    # Can attach/modify scripts
    PHYSICS = auto()   # Can modify physics (POD)

    # Convenience combinations
    FULL = MODIFY | COPY | TRANSFER | DELETE | MOVE | SCRIPT | PHYSICS
    OWNER_DEFAULT = MODIFY | DELETE | MOVE | SCRIPT | PHYSICS | TRANSFER
    COPY_OK = MODIFY | COPY | TRANSFER | DELETE | MOVE | SCRIPT | PHYSICS
    NO_TRANSFER = MODIFY | COPY | DELETE | MOVE | SCRIPT | PHYSICS
```

---

## Permission Sets

Each entity has **four permission levels**:

1. **Base**: Permissions for current owner
2. **Next Owner**: Permissions after transfer (owner → next owner)
3. **Group**: Permissions for group members (if group-owned)
4. **Everyone**: Public permissions

### Example Permission Set

```python
# Full rights object (can do anything)
PermissionSet(
    base=Permission.FULL,
    next_owner=Permission.FULL,
    group=Permission.NONE,
    everyone=Permission.NONE
)

# No-transfer object (can't give away)
PermissionSet(
    base=Permission.NO_TRANSFER,
    next_owner=Permission.NO_TRANSFER,
    group=Permission.NONE,
    everyone=Permission.NONE
)

# Free copy object (anyone can take copies)
PermissionSet(
    base=Permission.FULL,
    next_owner=Permission.COPY_OK,
    group=Permission.NONE,
    everyone=Permission.COPY,
    allow_copy_by_others=True
)
```

---

## Creating Objects with Permissions

### Basic Object Creation

```python
from permissions import permissions_full_rights, permissions_no_transfer

# Create object with full rights
obj_id = world.create_object(
    name="Magic Sword",
    description="A legendary blade",
    owner="user_caity",
    location="room_000",
    spawned_by="user_caity",
    permissions=permissions_full_rights()
)

# Create no-transfer object (soulbound)
obj_id = world.create_object(
    name="Soulbound Armor",
    description="Binds to owner",
    owner="user_caity",
    location="room_000",
    spawned_by="user_caity",
    permissions=permissions_no_transfer()
)
```

### Creating Noodlings with Permissions

```python
from permissions import PermissionSet, Permission

# Create Noodling with restricted permissions
agent_id = world.create_agent(
    name="servnak",
    checkpoint_path="checkpoints/servnak.pt",
    spawn_room="room_000",
    spawned_by="user_caity",
    permissions=PermissionSet(
        base=Permission.MOVE | Permission.DELETE,  # Can move and delete
        next_owner=Permission.NONE,  # Can't transfer ownership
        group=Permission.NONE,
        everyone=Permission.NONE
    )
)
```

---

## Checking Permissions

### Permission Check Methods

```python
# Check if user can modify
if world.can_user_modify("user_caity", "obj_001"):
    print("User can modify object")

# Check if user can delete
if world.can_user_delete("user_caity", "obj_001"):
    print("User can delete object")

# Check if user can transfer
if world.can_user_transfer("user_caity", "obj_001"):
    print("User can transfer object")

# Check if user can copy
if world.can_user_copy("user_caity", "obj_001"):
    print("User can copy object")
```

### Getting Metadata

```python
# Get full metadata for any entity
metadata = world.get_entity_metadata("obj_001")

if metadata:
    print(f"Creator: {metadata.creator}")
    print(f"Owner: {metadata.owner}")
    print(f"Spawned by: {metadata.created_by_user}")
    print(f"Spawned at: {metadata.spawned_at}")
    print(f"Spawned in: {metadata.spawned_in_room}")
    print(f"Modifications: {metadata.modification_count}")
    print(f"Transfers: {metadata.transfer_count}")
    print(f"Previous owners: {metadata.previous_owners}")
```

---

## Transferring Ownership

```python
# Transfer object to another user
success = world.transfer_entity(
    entity_id="obj_001",
    new_owner="user_spock",
    requester="user_caity"  # Must have transfer permission
)

if success:
    print("Transfer successful")

    # Metadata automatically updated:
    # - Previous owner added to history
    # - Owner changed to new_owner
    # - Permissions changed to next_owner permissions
    # - Transfer count incremented
```

---

## Recording Modifications

```python
# When user modifies an object
world.record_entity_modification("obj_001", "user_caity")

# Metadata automatically tracks:
# - last_modified_at (timestamp)
# - last_modified_by (user ID)
# - modification_count (incremented)
```

---

## Setting Permissions

```python
from permissions import PermissionSet, Permission

# Only owner or creator can change permissions
success = world.set_entity_permissions(
    entity_id="obj_001",
    permissions=PermissionSet(
        base=Permission.MODIFY | Permission.DELETE,
        next_owner=Permission.NONE,  # Next owner gets nothing
        group=Permission.NONE,
        everyone=Permission.NONE
    ),
    requester="user_caity"  # Must be owner or creator
)
```

---

## Permission Presets

Convenience functions for common permission patterns:

```python
from permissions import (
    permissions_full_rights,    # All permissions
    permissions_no_transfer,    # Can't transfer
    permissions_copy_ok,        # Includes copy
    permissions_free_copy,      # Anyone can copy
    permissions_locked,         # Only creator can modify
    permissions_public_modify   # Anyone can edit
)

# Use in object creation
obj_id = world.create_object(
    name="Free Gift",
    description="Take one!",
    owner="system",
    location="room_000",
    permissions=permissions_free_copy()
)
```

---

## Metadata Structure

Complete metadata for each entity:

```json
{
  "creator": "user_caity",
  "owner": "user_spock",
  "created_at": "2025-11-22T14:30:00",
  "created_by_user": "user_caity",
  "spawned_at": "2025-11-22T14:30:00",
  "spawned_in_room": "room_000",
  "group_id": null,
  "is_group_owned": false,
  "permissions": {
    "base": 127,
    "next_owner": 95,
    "group": 0,
    "everyone": 0,
    "allow_copy_by_others": false,
    "locked": false
  },
  "last_modified_at": "2025-11-22T15:45:00",
  "last_modified_by": "user_spock",
  "modification_count": 3,
  "previous_owners": ["user_caity"],
  "transfer_count": 1,
  "tags": ["magic", "weapon"],
  "notes": "Legendary sword forged in dragon fire"
}
```

---

## Example: Fire Imp Vending Machine

Fire imp vending machine that creates free-copy fire imps:

```python
from physics_object_descriptor import PhysicsObjectDescriptor
from permissions import permissions_free_copy

# In vending machine script:
def vend_fire_imp(self, requester: str):
    """Vend a free-copy fire imp."""

    # Create POD
    fire_imp_pod = PhysicsObjectDescriptor(
        mass="negligible",
        material="living flame",
        metadata={"temperature": "800°F"}
    )

    # Create prim with free-copy permissions
    prim_id = world.create_object(
        name="Fire Imp",
        description="A mischievous flame elemental",
        owner="script_system",
        location=self.Room,
        pod=fire_imp_pod,
        spawned_by=requester,  # User who pressed button
        permissions=permissions_free_copy()  # Anyone can take copies!
    )

    # Create Noodling with no-transfer permissions
    noodling_id = world.create_agent(
        name=f"FireImp_{random_id()}",
        checkpoint_path="checkpoints/fire_imp.pt",
        spawn_room=self.Room,
        spawned_by=requester,
        permissions=permissions_no_transfer()  # Can't give away
    )

    return prim_id, noodling_id
```

Now users can:
- Take free copies of the fire imp prim (has physics)
- But cannot transfer the Noodling consciousness (personality is soulbound)

---

## Legacy Compatibility

Old objects/agents without metadata still work:

```python
# Old object (no metadata field)
{
  "uid": "obj_001",
  "name": "Old Rock",
  "owner": "user_caity",
  "created": "2025-11-01T12:00:00"
  # No metadata field
}

# Permission check returns False (safe default)
world.can_user_modify("user_spock", "obj_001")  # → False
```

To upgrade old objects:

```python
from permissions import EntityMetadata, permissions_full_rights

# Get old object
obj = world.get_object("obj_001")

# Create metadata
metadata = EntityMetadata.create_new(
    creator=obj['owner'],
    spawned_by=obj['owner'],
    spawn_room=obj.get('location'),
    permissions=permissions_full_rights()
)

# Add to object
obj['metadata'] = metadata.to_dict()
world.save_all()
```

---

## Command Integration (Future)

Commands to be implemented:

```
@permissions <object>             # Show permissions
@permissions <object> set <flags> # Set permissions
@transfer <object> to <user>      # Transfer ownership
@info <object>                    # Show full metadata
@history <object>                 # Show modification history
```

---

## Summary

**Every prim and Noodling now tracks:**

✓ Who created it (never changes)
✓ Who spawned it (survives transfers)
✓ When it was spawned
✓ Where it was spawned
✓ Who owns it currently
✓ Full permission flags (modify, copy, transfer, delete, move, script, physics)
✓ Modification history (who, when, how many times)
✓ Transfer history (previous owners, transfer count)

**Permission system supports:**

✓ Second Life-style base/next_owner/group/everyone permissions
✓ Free copy objects (anyone can take)
✓ No-transfer objects (soulbound)
✓ Locked objects (only creator can modify)
✓ Public modify objects (anyone can edit)
✓ Group ownership (framework ready)

---

**System Status:** ✓ Fully operational and integrated

*Permission system complete. The world is now properly governed.*

**— Commander Spock**
