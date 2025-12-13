# FacetAssemblyProxy

class in Noodlings.Scripting

## Description

Proxy object for a facet assembly (cognitive topology). Provides methods to query, modify, and save facet assemblies.

## Methods

| Method | Description |
|--------|-------------|
| [get_facet()](#get_facet) | Get facet by ID |
| [get_facet_by_name()](#get_facet_by_name) | Get facet by display name |
| [list_facets()](#list_facets) | List all facets in assembly |
| [add_facet()](#add_facet) | Add new facet to assembly |
| [remove_facet()](#remove_facet) | Remove facet from assembly |
| [connect()](#connect) | Connect two facets via data pads |
| [disconnect()](#disconnect) | Disconnect two facets |
| [save()](#save) | Save modified assembly to YAML file |

---

## get_facet()

Get facet by ID.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| facet_id | string | Facet UUID or ID from YAML |

**Returns:** FacetProxy instance or null

**Example:**
```javascript
var charm_facet = assembly.get_facet("CHARM_NET");
if (charm_facet) {
    context.log("Facet type: " + charm_facet.get_type());
}
```

---

## get_facet_by_name()

Get facet by display name.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| name | string | Facet name (e.g., "Red's Mind") |

**Returns:** FacetProxy instance or null

**Example:**
```javascript
var mind = assembly.get_facet_by_name("Red's Mind");
if (mind) {
    var model = mind.get_property("model");
    context.log("Red's Mind uses: " + model);
}
```

---

## list_facets()

List all facets in the assembly.

**Parameters:** None

**Returns:** Array of {id, name, type} objects

**Example:**
```javascript
var facets = assembly.list_facets();
facets.forEach(function(f) {
    context.log(f.id + ": " + f.name + " (" + f.type + ")");
});
```

---

## add_facet()

Add a new facet to the assembly.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| facet_type | string | Facet type (e.g., "LLMFacet", "ScriptedFacet") |
| name | string | Display name for the facet |
| properties | object | Initial properties (optional) |

**Returns:** Facet ID (string) or null on failure

**Example:**
```javascript
var facet_id = assembly.add_facet("LLMFacet", "Custom Reasoner", {
    model: "LARGE",
    temperature: 0.8
});
```

---

## remove_facet()

Remove a facet from the assembly.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| facet_id | string | Facet ID |

**Returns:** true if removed successfully, false on error

**Example:**
```javascript
var removed = assembly.remove_facet("OLD_FACET");
```

---

## connect()

Connect two facets via their data pads.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| from_facet | string | Source facet ID |
| from_pad | string | Source pad name |
| to_facet | string | Target facet ID |
| to_pad | string | Target pad name |

**Returns:** true if connected successfully, false on error

**Example:**
```javascript
assembly.connect("CHARM_NET", "affect_valence", "RED_MIND", "affect");
```

---

## disconnect()

Disconnect two facets.

**Parameters:** Same as connect()

**Returns:** true if disconnected successfully, false on error

**Example:**
```javascript
assembly.disconnect("CHARM_NET", "affect_valence", "RED_MIND", "affect");
```

---

## save()

Save modified assembly to YAML file.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| filepath | string | Path to save file |

**Returns:** true if saved successfully, false on error

**Example:**
```javascript
assembly.save("facet_assemblies/red_modified.yaml");
```
