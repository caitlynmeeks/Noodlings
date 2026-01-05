# UI Components - Full Reference

## Panel

Container component for grouping other components.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| background | color | #2d2d2d | Background color |
| padding | number | 8 | Inner padding in pixels |
| border_radius | number | 4 | Corner rounding |
| border_color | color | transparent | Border color |
| border_width | number | 0 | Border thickness |

**Events:** None (container only)

**Usage:** Drag other components into Panel to group them.

---

## Button

Clickable button component.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| label | string | "Button" | Button text |
| enabled | bool | true | Whether clickable |
| background | color | #3e3e3e | Button color |
| text_color | color | #ffffff | Label color |
| hover_color | color | #4e4e4e | Color on hover |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onClick | {} | Fired when clicked |
| onHover | {hovering: bool} | Fired on mouse enter/leave |

**Common wiring:**
- onClick -> Run Assembly
- onClick -> Set Value (toggle something)
- onClick -> Call Script

---

## Label

Static text display.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| text | string | "Label" | Display text |
| font_size | number | 14 | Text size |
| text_color | color | #d2d2d2 | Text color |
| alignment | enum | left | left, center, right |
| bold | bool | false | Bold text |

**Events:** None

**Binding:** Can bind `text` to assembly output: `{assembly.output_name}`

---

## TextField

Editable text input or output display.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| text | string | "" | Current text value |
| placeholder | string | "" | Placeholder when empty |
| editable | bool | true | Whether user can edit |
| multiline | bool | false | Allow multiple lines |
| font_size | number | 14 | Text size |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onChange | {value: string} | Text changed |
| onSubmit | {value: string} | Enter pressed |

**Common wiring:**
- Bind `text` to assembly output for display
- onChange -> trigger assembly for live processing
- onSubmit -> trigger assembly for form submission

---

## ImageDisplay

Displays images, accepts drag-drop.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| image | image | null | Current image |
| fit | enum | contain | contain, cover, fill |
| background | color | #1a1a1a | Background when no image |
| accept_drop | bool | true | Accept dragged images |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onDrop | {image: ImageData} | Image dropped |
| onClick | {} | Clicked |

**Common wiring:**
- onDrop -> Run Assembly (process image with VisionFacet)

---

## ChatHistory

Scrolling message history for chat interfaces.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| messages | array | [] | Message objects |
| max_messages | number | 100 | Scroll buffer |
| show_timestamps | bool | false | Show message times |
| user_color | color | #4a9eff | User message color |
| assistant_color | color | #10a37f | Assistant message color |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onMessageClick | {message: Message} | Message clicked |

**API:**
- `addMessage(role, content)` - Add message programmatically
- `clear()` - Clear history

---

## ChatInput

Text input with send button for chat interfaces.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| placeholder | string | "Type a message..." | Placeholder text |
| send_label | string | "Send" | Send button text |
| enabled | bool | true | Whether input active |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onSend | {text: string} | Send button clicked or Enter pressed |

**Common wiring:**
- onSend -> Run Assembly (LLM processing)
- Assembly output -> ChatHistory.addMessage

---

## Checkbox

Boolean toggle with label.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| checked | bool | false | Current state |
| label | string | "" | Label text |
| enabled | bool | true | Whether clickable |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onChange | {checked: bool} | State changed |

**API:**
- `toggle()` - Flip the checked state

---

## Dropdown

ComboBox/select component.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| value | string | "" | Selected value |
| options | array | [] | Available options |
| placeholder | string | "Select..." | Placeholder when none selected |
| enabled | bool | true | Whether interactive |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onChange | {value: string} | Selection changed |

**API:**
- `addOption(value, label)` - Add option dynamically
- `removeOption(value)` - Remove option
- `setOptions(options)` - Replace all options

---

## Slider

Numeric range slider.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| value | number | 0 | Current value |
| min | number | 0 | Minimum value |
| max | number | 100 | Maximum value |
| step | number | 1 | Step increment |
| show_value | bool | true | Display current value |
| format | string | "{value}" | Value display format |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onChange | {value: number} | Value changed |
| onSlideStart | {} | User started dragging |
| onSlideEnd | {value: number} | User stopped dragging |

---

## RadioGroup

Mutually exclusive selection group.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| value | string | "" | Selected option value |
| options | array | [] | Array of {value, label} objects |
| orientation | enum | vertical | vertical, horizontal |
| enabled | bool | true | Whether interactive |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onChange | {value: string} | Selection changed |

---

## RadianceViewport

3D Gaussian splatting viewport.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| radiance_path | string | "" | Path to .radiance file |
| camera_position | point3 | {0,0,5} | Camera position |
| camera_target | point3 | {0,0,0} | Look-at target |
| auto_rotate | bool | false | Spin scene continuously |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onLoad | {} | Radiance loaded |
| onCameraMove | {position, target} | Camera changed |
| onGaussianClick | {position, entity, semantics} | Gaussian clicked |
| onGaussianHover | {position, entity, semantics} | Gaussian hovered |

---

## WebView

Embedded web browser component.

**Properties:**
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| url | string | "" | URL to load |
| html | string | "" | HTML content (alternative to url) |
| javascript_enabled | bool | true | Allow JS execution |
| zoom | number | 1.0 | Zoom level |

**Events:**
| Event | Payload | Description |
|-------|---------|-------------|
| onLoad | {url: string} | Page loaded |
| onNavigate | {url: string} | Navigation requested |

**Note:** Requires PyQt6-WebEngine package.
