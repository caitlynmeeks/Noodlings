# Facet Types - Full Reference

## LLMFacet

Calls a language model.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| prompt | string | User prompt / input text |
| context | string | Optional additional context |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| response | string | Model response |
| error | string | Error message if failed |

**Properties:**
| Property | Default | Description |
|----------|---------|-------------|
| system_prompt | "" | System instructions |
| model | "auto" | Model selection |
| temperature | 0.7 | Response randomness |
| max_tokens | 1024 | Response length limit |

---

## VisionFacet

Analyzes images using vision model.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| image | ImageData | Image to analyze |
| prompt | string | What to look for / describe |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| description | string | Image description |
| error | string | Error if failed |

**Properties:**
| Property | Default | Description |
|----------|---------|-------------|
| detail_level | "auto" | low, high, auto |
| model | "auto" | Vision model to use |

---

## ScriptedFacet

Runs JavaScript code.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| input | any | Data to process |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| output | any | Processed result |

**Properties:**
| Property | Description |
|----------|-------------|
| script | JavaScript code (has access to `input` variable, return value becomes `output`) |

**Example script:**
```javascript
// Reverse a string
return input.split('').reverse().join('');
```

---

## BranchFacet

Conditional routing based on condition.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| input | any | Data to route |
| condition | bool | Which branch to take |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| true_out | any | Output if condition true |
| false_out | any | Output if condition false |

---

## MergeFacet

Combines multiple inputs.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| input_1 | any | First input |
| input_2 | any | Second input |
| input_n | any | Additional inputs (configurable) |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| merged | object | Combined inputs as object |

**Properties:**
| Property | Default | Description |
|----------|---------|-------------|
| merge_mode | "object" | object, array, concat |

---

## ContextIntelligenceFacet

Gathers context for LLM processing.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| trigger | any | Trigger to gather context |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| context | string | Assembled context |

**Properties:**
| Property | Description |
|----------|-------------|
| include_history | Include conversation history |
| include_perception | Include what Thing perceives |
| max_tokens | Context size limit |

---

## CharmNetFacet

Neural network integration for charm networks.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| input | tensor | Input data |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| output | tensor | Network output |
| activations | object | Layer activations (for visualization) |

**Properties:**
| Property | Description |
|----------|-------------|
| network_type | LSTM, GRU, or Transformer |
| hidden_size | Network hidden dimension |
| layers | Number of layers |

---

## TickerFacet

Emits signals at regular intervals.

**Inputs:** None (self-triggering)

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| tick | signal | Emitted each interval |
| count | number | Number of ticks so far |

**Properties:**
| Property | Default | Description |
|----------|---------|-------------|
| interval | 1.0 | Seconds between ticks |
| auto_start | true | Start ticking immediately |

---

## RateLimiterFacet

Limits how often data passes through.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| input | any | Data to rate-limit |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| output | any | Data (when allowed) |
| dropped | any | Data (when rate-limited) |

**Properties:**
| Property | Default | Description |
|----------|---------|-------------|
| max_per_second | 1.0 | Maximum throughput |

---

## CacheFacet

Caches data with optional expiry.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| input | any | Data to cache |
| key | string | Cache key |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| output | any | Cached value |
| hit | bool | Whether cache was used |

**Properties:**
| Property | Default | Description |
|----------|---------|-------------|
| ttl_seconds | 300 | Time to live |
| max_entries | 100 | Maximum cached items |

---

## AccumulatorFacet

Collects data over time.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| input | any | Item to add |
| flush | signal | Trigger to emit accumulated |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| accumulated | array | All collected items |
| count | number | Current count |

**Properties:**
| Property | Default | Description |
|----------|---------|-------------|
| max_items | 100 | Maximum buffer size |
| auto_flush_at | 0 | Auto-flush at count (0=never) |

---

## MCPFacet

Calls Model Context Protocol tools.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| input | any | Tool input |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| output | any | Tool result |
| error | string | Error if failed |

**Properties:**
| Property | Description |
|----------|-------------|
| server | MCP server name |
| tool | Tool name to call |

---

## Special Nodes

### INCOMING
Entry point for assembly. Receives trigger data.
- Always present in assembly
- Output connects to first processing facet

### OUTGOING
Exit point for assembly. Emits final result.
- Always present in assembly
- Input receives final processed data
- Triggers OnComplete event

### CONVERGENCE
Multi-input synchronization gate.
- Waits for ALL inputs before passing through
- Use for parallel processing merge points

---

## Assembly Properties

When a Facet Assembly is attached to a Thing, these properties are available:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| run_in_cognition_loop | bool | false | If true, runs continuously at tick_rate |
| tick_rate | float | 0.1 | Seconds between executions (if continuous) |
| auto_run_on_attach | bool | false | Run once when component attached |

**Events:**
| Event | Description |
|-------|-------------|
| OnComplete | Fired after one-shot execution |
| OnStateChange | Fired when continuous state changes |
| OnError | Fired on execution error |
