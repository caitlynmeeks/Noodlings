# "Rez" not "Spawn" - Second Life Heritage! 🎯

**Decision**: November 18, 2025

---

## Why "Rez" Instead of "Spawn"

### "Spawn" is Icky
- Biological (spawn = eggs, offspring)
- Gaming cliché (every game uses "spawn")
- Implies creation from nothing
- Feels mechanical/impersonal

### "Rez" is Rad! 🔥
- Second Life heritage (Philip Rosedale!)
- **Rez** = materialize, bring into being
- Short, punchy, memorable
- Already familiar to virtual world builders
- Implies intention and craft

---

## Terminology

| Old | New | Usage |
|-----|-----|-------|
| Spawn | **Rez** | Noodlings.Rez("phi.noodling") |
| Spawner | **Rezzer** | class EnsembleRezzer |
| spawn_count | **rez_count** | self.rez_count += 1 |
| spawned_ids | **rezzed_ids** | List of rezzed prims |
| Spawn time | **Reztime** | When the Noodling was rezzed |

---

## The Lineage

**Second Life prims (2003)** → Philip Rosedale coins "rez"
↓
**USD prims (2015)** → Pixar standardizes scene description
↓
**Noodling prims (2025)** → We honor both traditions!

**"Rez" connects us to Second Life's legacy!** 🎯

---

## API Changes

### Before:
```python
Noodlings.Spawn("anklebiter.noodling")
spawned = Noodlings.SpawnPrim("prop", "Box")
```

### After:
```python
Noodlings.Rez("anklebiter.noodling")
rezzed = Noodlings.RezPrim("prop", "Box")
```

---

## UI Changes

✅ **Menu Items:**
- ~~Spawn Noodling~~ → **Rez Noodling**
- ~~Spawn Ensemble~~ → **Rez Ensemble**

✅ **Scripting API:**
- `Noodlings.Rez()` - Rez a Noodling
- `Noodlings.RezPrim()` - Rez a prim
- `EnsembleRezzer` class

✅ **Example Scripts:**
- ClickableBox uses `self.rez_count`
- QuestGiver rezzes rewards
- VendingMachine rezzes items

✅ **Dialogs:**
- "Rezzing not yet implemented"
- "Rezzed ensemble: Space Trek Crew"

---

**Philip Rosedale would approve!** 🎯✨
