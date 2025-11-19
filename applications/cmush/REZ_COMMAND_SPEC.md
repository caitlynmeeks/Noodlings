# @rez Command Redesign - Unified Rezzing

**Status**: Specification (implementation next session)

---

## Unified @rez Command

One command to rez everything: Noodlings, prims, exits, ensembles.

### Syntax

```
@rez -n [-f] <name> [description]      # Rez Noodling
@rez -p <name> <description>           # Rez prim/object
@rez -d <direction> <description>      # Rez exit/direction
@rez -e <ensemble_name>                # Rez ensemble
```

---

## Type Flags

### `-n` Noodling (default if no flag)

```
@rez -n phi
@rez -n -f callie servnak phi phido   # Fresh memory for all
@rez -n "Backwards Dweller"           # Quoted names
@rez phi                               # -n is default
```

**Behavior:**
- Loads recipe from `recipes/{name}.yaml`
- Creates kindled Noodling with personality
- `-f` flag: fresh memory (ignore saved state)
- Can rez multiple at once

### `-p` Prim/Object

```
@rez -p "Old Radio" "An old-time radio with glowing tubes"
@rez -p "Wooden Sword" "Caity's practice sword"
@rez -p "Tensor Taffy" "Glowing candy that tastes like mathematics"
```

**Behavior:**
- Creates simple prim (not kindled)
- Has name and description
- Can be @take, @drop, @look
- Stored in world state

### `-d` Direction/Exit

```
@rez -d "north" "Way to the Monkey Park"
@rez -d "through the portal" "A shimmering gateway"
```

**Behavior:**
- Creates exit from current room
- First arg: direction keyword
- Second arg: description of destination
- Creates or links to destination room

### `-e` Ensemble

```
@rez -e "bridge crew"
@rez -e "commedia_dellarte"
@rez -e space_trek
```

**Behavior:**
- Loads .ensemble file
- Rezzes all Noodlings in ensemble
- Creates ensemble parent in hierarchy
- Noodlings are children of ensemble

---

## Hierarchy View

### Ensemble Structure

```
┬ Scene: The Nexus
├─┬ Ensembles
│ └─┬ Bridge Crew (ensemble)
│   ├─ Captain Sterling
│   ├─ Commander Velar
│   ├─ Chief MacReady
│   └─ Dr. Patel
├─┬ Noodlings (standalone)
│ ├─ Phi
│ └─ Servnak
├─┬ Prims
│ ├─ Old Radio
│ └─ Wooden Sword
```

**Noodlings in ensembles:**
- Listed as children of ensemble parent
- Can still be selected/inspected independently
- Ensemble tracks relationship dynamics

---

## USD Mapping

### Ensemble in USD

```usd
def Ensemble "BridgeCrew" {
    prepend rel ensemble:characters = [
        </Stage/Noodlings/Captain>,
        </Stage/Noodlings/Engineer>,
        </Stage/Noodlings/Doctor>
    ]

    string ensemble:relationshipDynamics = "..."
}

def Character "Captain" {
    rel character:ensemble = </Stage/Ensembles/BridgeCrew>
}
```

All @rez types map to USD prims with proper schemas.

---

## Implementation Plan

### Phase 1: Parse Flags
- Detect -n, -p, -d, -e flags
- Route to appropriate handler
- Maintain backwards compatibility (no flag = -n)

### Phase 2: Implement Handlers
- `rez_noodling()` - Current implementation
- `rez_prim()` - New: create simple object
- `rez_direction()` - New: create exit
- `rez_ensemble()` - New: load .ensemble, rez all members

### Phase 3: Hierarchy Integration
- Add "Ensembles" folder to Scene Hierarchy
- Show ensemble as parent with Noodling children
- Track ensemble membership

### Phase 4: USD Integration
- Ensemble schema export
- Character → ensemble relationships
- Proper prim types for exits

---

## Examples

### Rez Bridge Crew Ensemble

```
@rez -e bridge_crew
```

**Result:**
- Loads `bridge_crew.ensemble`
- Rezzes Captain, Engineer, Doctor, Logician
- Creates ensemble parent in hierarchy
- Sets up relationships

### Rez Props for Scene

```
@rez -p "Control Console" "Starship command console"
@rez -p "Captain's Chair" "Imposing seat of authority"
@rez -p "Viewscreen" "Shows stars and threats"
```

**Result:**
- 3 prims in current room
- Can be examined with @look
- Appear in Prims folder in hierarchy

### Rez Monkey Park Exit

```
@rez -d "north" "Way to the Monkey Park"
```

**Result:**
- Creates "north" exit
- Links to (or creates) Monkey Park room
- Shows in Exits folder in hierarchy

---

## Backwards Compatibility

**Old command:**
```
@spawn phi
@spawn -f callie
@spawn -e servnak
```

**Still works** (@spawn is alias for @rez -n)

---

Ready to implement when you are. This is a session's worth of work.
