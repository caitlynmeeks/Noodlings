# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Permissions System
#
#   Controls who can do what with objects and characters - like
#   Second Life's permission system. Every item tracks its creator
#   (who made it), owner (who has it now), and what operations
#   are allowed (edit, copy, give away, delete). When you transfer
#   something, the "next owner permissions" kick in.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.permissions
# PURPOSE:  Second Life-style permission management for entities
# LAYER:    Backend / Authorization
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Permission       Bitwise flags (MODIFY, COPY, TRANSFER, etc.)
#   PermissionSet    Complete permission configuration
#   EntityMetadata   Full provenance and permission tracking
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Second Life-style Permissions System

Implements permission management for prims (objects) and Noodlings (agents).

Based on Second Life's permission model:
- Creator: Original creator (never changes)
- Owner: Current owner (can change via transfer)
- Group: Optional group ownership
- Permissions: modify, copy, transfer

Author: Caitlyn + Claude
Date: November 22, 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Flag, auto


class Permission(Flag):
    """
    Permission flags (bitwise combinable).

    Based on Second Life permissions system.
    """
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


@dataclass
class PermissionSet:
    """
    Complete permission set for an entity.

    Matches Second Life's permission model with extensions for Noodlings.
    """
    # Base permissions for current owner
    base: Permission = Permission.FULL

    # Next owner permissions (after transfer)
    next_owner: Permission = Permission.OWNER_DEFAULT

    # Group permissions (if entity is group-owned)
    group: Permission = Permission.NONE

    # Everyone permissions (public)
    everyone: Permission = Permission.NONE

    # Special: Can others take copies? (SL-style "free copy" objects)
    allow_copy_by_others: bool = False

    # Special: Is this entity locked (immutable by anyone except creator)?
    locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            'base': self.base.value,
            'next_owner': self.next_owner.value,
            'group': self.group.value,
            'everyone': self.everyone.value,
            'allow_copy_by_others': self.allow_copy_by_others,
            'locked': self.locked
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PermissionSet':
        """Deserialize from dictionary."""
        return cls(
            base=Permission(data.get('base', Permission.FULL.value)),
            next_owner=Permission(data.get('next_owner', Permission.OWNER_DEFAULT.value)),
            group=Permission(data.get('group', Permission.NONE.value)),
            everyone=Permission(data.get('everyone', Permission.NONE.value)),
            allow_copy_by_others=data.get('allow_copy_by_others', False),
            locked=data.get('locked', False)
        )


@dataclass
class EntityMetadata:
    """
    Complete metadata for any entity (prim or Noodling).

    Tracks provenance, permissions, and modification history.
    """
    # Provenance
    creator: str                    # Original creator (never changes)
    owner: str                      # Current owner (can change)
    created_at: str                 # ISO timestamp of creation
    created_by_user: str            # User who spawned this entity
    spawned_at: str                 # ISO timestamp when spawned into world
    spawned_in_room: Optional[str] = None  # Initial spawn location

    # Group ownership (optional)
    group_id: Optional[str] = None
    is_group_owned: bool = False

    # Permissions
    permissions: PermissionSet = field(default_factory=PermissionSet)

    # Modification history
    last_modified_at: Optional[str] = None
    last_modified_by: Optional[str] = None
    modification_count: int = 0

    # Transfer history
    previous_owners: List[str] = field(default_factory=list)
    transfer_count: int = 0

    # Additional metadata
    tags: List[str] = field(default_factory=list)  # User-defined tags
    notes: str = ""  # Creator notes

    def transfer_to(self, new_owner: str):
        """
        Transfer ownership to new owner.

        Args:
            new_owner: User ID of new owner
        """
        # Record previous owner
        if self.owner not in self.previous_owners:
            self.previous_owners.append(self.owner)

        # Apply next_owner permissions
        self.permissions.base = self.permissions.next_owner

        # Update owner
        self.owner = new_owner
        self.transfer_count += 1
        self.last_modified_at = datetime.now().isoformat()
        self.last_modified_by = new_owner

    def record_modification(self, modifier: str):
        """
        Record that entity was modified.

        Args:
            modifier: User ID who modified
        """
        self.last_modified_at = datetime.now().isoformat()
        self.last_modified_by = modifier
        self.modification_count += 1

    def can_modify(self, user_id: str) -> bool:
        """
        Check if user can modify this entity.

        Args:
            user_id: User ID to check

        Returns:
            True if user has modify permission
        """
        # Locked entities only modifiable by creator
        if self.permissions.locked:
            return user_id == self.creator

        # Owner always has base permissions
        if user_id == self.owner:
            return Permission.MODIFY in self.permissions.base

        # Check group permissions
        if self.is_group_owned and user_id in self._get_group_members():
            return Permission.MODIFY in self.permissions.group

        # Check everyone permissions
        return Permission.MODIFY in self.permissions.everyone

    def can_delete(self, user_id: str) -> bool:
        """Check if user can delete this entity."""
        if self.permissions.locked:
            return user_id == self.creator

        if user_id == self.owner:
            return Permission.DELETE in self.permissions.base

        if self.is_group_owned and user_id in self._get_group_members():
            return Permission.DELETE in self.permissions.group

        return Permission.DELETE in self.permissions.everyone

    def can_transfer(self, user_id: str) -> bool:
        """Check if user can transfer ownership."""
        if self.permissions.locked:
            return False  # Locked items cannot be transferred

        if user_id == self.owner:
            return Permission.TRANSFER in self.permissions.base

        return False  # Only owner can transfer

    def can_copy(self, user_id: str) -> bool:
        """Check if user can make copies."""
        # Special flag: allow_copy_by_others (free copy objects)
        if self.permissions.allow_copy_by_others:
            return True

        if user_id == self.owner:
            return Permission.COPY in self.permissions.base

        if self.is_group_owned and user_id in self._get_group_members():
            return Permission.COPY in self.permissions.group

        return Permission.COPY in self.permissions.everyone

    def can_script(self, user_id: str) -> bool:
        """Check if user can attach/modify scripts."""
        if self.permissions.locked:
            return user_id == self.creator

        if user_id == self.owner:
            return Permission.SCRIPT in self.permissions.base

        return False  # Scripting typically owner-only

    def can_modify_physics(self, user_id: str) -> bool:
        """Check if user can modify POD (physics)."""
        if self.permissions.locked:
            return user_id == self.creator

        if user_id == self.owner:
            return Permission.PHYSICS in self.permissions.base

        return False  # Physics typically owner-only

    def _get_group_members(self) -> List[str]:
        """
        Get group members (stub - to be implemented with group system).

        Returns:
            List of user IDs in group
        """
        # TODO: Implement group system
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            'creator': self.creator,
            'owner': self.owner,
            'created_at': self.created_at,
            'created_by_user': self.created_by_user,
            'spawned_at': self.spawned_at,
            'spawned_in_room': self.spawned_in_room,
            'group_id': self.group_id,
            'is_group_owned': self.is_group_owned,
            'permissions': self.permissions.to_dict(),
            'last_modified_at': self.last_modified_at,
            'last_modified_by': self.last_modified_by,
            'modification_count': self.modification_count,
            'previous_owners': self.previous_owners,
            'transfer_count': self.transfer_count,
            'tags': self.tags,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityMetadata':
        """Deserialize from dictionary."""
        return cls(
            creator=data.get('creator', 'unknown'),
            owner=data.get('owner', 'unknown'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            created_by_user=data.get('created_by_user', 'unknown'),
            spawned_at=data.get('spawned_at', datetime.now().isoformat()),
            spawned_in_room=data.get('spawned_in_room'),
            group_id=data.get('group_id'),
            is_group_owned=data.get('is_group_owned', False),
            permissions=PermissionSet.from_dict(data.get('permissions', {})),
            last_modified_at=data.get('last_modified_at'),
            last_modified_by=data.get('last_modified_by'),
            modification_count=data.get('modification_count', 0),
            previous_owners=data.get('previous_owners', []),
            transfer_count=data.get('transfer_count', 0),
            tags=data.get('tags', []),
            notes=data.get('notes', '')
        )

    @classmethod
    def create_new(
        cls,
        creator: str,
        spawned_by: str,
        spawn_room: Optional[str] = None,
        permissions: Optional[PermissionSet] = None
    ) -> 'EntityMetadata':
        """
        Create new metadata for entity being spawned.

        Args:
            creator: Original creator user ID
            spawned_by: User who spawned this instance
            spawn_room: Room where spawned
            permissions: Custom permissions (or default)

        Returns:
            New EntityMetadata instance
        """
        now = datetime.now().isoformat()

        return cls(
            creator=creator,
            owner=spawned_by,  # Spawner becomes initial owner
            created_at=now,
            created_by_user=spawned_by,
            spawned_at=now,
            spawned_in_room=spawn_room,
            permissions=permissions or PermissionSet()
        )


# ===== PERMISSION PRESETS =====

def permissions_full_rights() -> PermissionSet:
    """Full rights (all permissions)."""
    return PermissionSet(
        base=Permission.FULL,
        next_owner=Permission.FULL,
        group=Permission.NONE,
        everyone=Permission.NONE
    )


def permissions_no_transfer() -> PermissionSet:
    """No transfer (can't give away)."""
    return PermissionSet(
        base=Permission.NO_TRANSFER,
        next_owner=Permission.NO_TRANSFER,
        group=Permission.NONE,
        everyone=Permission.NONE
    )


def permissions_copy_ok() -> PermissionSet:
    """Copy OK (full perms including copy)."""
    return PermissionSet(
        base=Permission.COPY_OK,
        next_owner=Permission.COPY_OK,
        group=Permission.NONE,
        everyone=Permission.NONE
    )


def permissions_free_copy() -> PermissionSet:
    """Free copy (anyone can take copies)."""
    return PermissionSet(
        base=Permission.FULL,
        next_owner=Permission.COPY_OK,
        group=Permission.NONE,
        everyone=Permission.COPY,
        allow_copy_by_others=True
    )


def permissions_locked() -> PermissionSet:
    """Locked (only creator can modify)."""
    perms = PermissionSet(
        base=Permission.NONE,
        next_owner=Permission.NONE,
        group=Permission.NONE,
        everyone=Permission.NONE
    )
    perms.locked = True
    return perms


def permissions_public_modify() -> PermissionSet:
    """Public modify (anyone can edit)."""
    return PermissionSet(
        base=Permission.FULL,
        next_owner=Permission.OWNER_DEFAULT,
        group=Permission.NONE,
        everyone=Permission.MODIFY | Permission.MOVE
    )

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
