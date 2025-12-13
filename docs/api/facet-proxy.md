# FacetProxy

class in Noodlings.Scripting

## Description

Proxy object for a single facet. Provides access to facet properties and metadata.

## Methods

| Method | Description |
|--------|-------------|
| [get_property()](#get_property) | Get facet property value |
| [set_property()](#set_property) | Set facet property value |
| [get_all_properties()](#get_all_properties) | Get all facet properties |
| [get_type()](#get_type) | Get facet type |
| [get_id()](#get_id) | Get facet ID |
| [get_name()](#get_name) | Get facet display name |

---

## get_property()

Get a facet property value.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| name | string | Property name (e.g., "model", "temperature") |

**Returns:** Property value (any type) or null

**Example:**
```javascript
var model = facet.get_property("model");
var temp = facet.get_property("temperature");
```

---

## set_property()

Set a facet property value.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| name | string | Property name |
| value | any | New value |

**Returns:** true if set successfully, false on error

**Example:**
```javascript
facet.set_property("temperature", 0.95);
facet.set_property("model", "LARGE");
```

---

## get_all_properties()

Get all facet properties as an object.

**Parameters:** None

**Returns:** Object with all properties

**Example:**
```javascript
var props = facet.get_all_properties();
for (var key in props) {
    context.log(key + ": " + props[key]);
}
```

---

## get_type()

Get facet type.

**Parameters:** None

**Returns:** Type string (e.g., "LLMFacet", "ScriptedFacet")

**Example:**
```javascript
var type = facet.get_type();
```

---

## get_id()

Get facet ID.

**Parameters:** None

**Returns:** ID string

**Example:**
```javascript
var id = facet.get_id();
```

---

## get_name()

Get facet display name.

**Parameters:** None

**Returns:** Name string

**Example:**
```javascript
var name = facet.get_name();
// "Red's Mind"
```
