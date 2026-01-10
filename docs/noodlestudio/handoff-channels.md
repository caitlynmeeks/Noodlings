# Handoff: Channel Architecture Implementation

**From**: Architecture Claude
**To**: Coding Claude
**Date**: 2026-01-08
**Priority**: High (unblocks stage direction, ensemble dynamics)

---

## Context

We've designed a channel architecture for inter-noodling communication. The full spec is at:

**`/docs/noodlestudio/channels.md`** ← READ THIS FIRST

Channels are named message buses that enable pub/sub patterns beyond direct speech. This powers:
- Stage direction (Brenda → Guide)
- Environmental context (#world.weather)
- Group communication (#bridge.comms)
- Private messaging (#dm.a→b)

---

## Implementation Order

### 1. Assembly Schema Update

Add `channels` field to the assembly YAML schema:

```yaml
# In assembly.yaml
channels:
  subscribe:
    - "#directors.cues"
    - "#world.context"
  publish:
    - "#directors.feedback"
```

Location: Look for where assembly validation happens. Add channels as optional field with subscribe/publish arrays.

### 2. ChannelBus Class

Create a pub/sub message bus in the runtime:

```python
# runtime/channels.py (new file)
class ChannelBus:
    """Named message bus for inter-noodling communication."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._history: dict[str, list[ChannelMessage]] = {}

    def subscribe(self, channel: str, callback: Callable):
        """Subscribe to a channel."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)

    def unsubscribe(self, channel: str, callback: Callable):
        """Unsubscribe from a channel."""
        if channel in self._subscribers:
            self._subscribers[channel].remove(callback)

    def publish(self, channel: str, message: ChannelMessage):
        """Publish message to all subscribers."""
        # Store in history
        if channel not in self._history:
            self._history[channel] = []
        self._history[channel].append(message)

        # Notify subscribers
        for callback in self._subscribers.get(channel, []):
            callback(message)

    def get_latest(self, channel: str) -> Optional[ChannelMessage]:
        """Get most recent message on a channel."""
        history = self._history.get(channel, [])
        return history[-1] if history else None


@dataclass
class ChannelMessage:
    channel: str
    from_noodling: str  # sender ID or "system"
    timestamp: float
    payload: dict
```

### 3. Wire ChannelBus to Stage

The Stage (or StageRuntime) should own the ChannelBus instance. All noodlings on a stage share the same bus.

```python
class Stage:
    def __init__(self, ...):
        self.channel_bus = ChannelBus()
```

### 4. Noodling Channel Integration

When a noodling loads, read its assembly channels config and:
- Subscribe to listed channels
- Store publish permissions

```python
class Noodling:
    def __init__(self, assembly, stage):
        self.stage = stage
        self._subscribed_channels = []
        self._publish_channels = set()

        # Wire up channels from assembly
        channels_config = assembly.get('channels', {})
        for channel in channels_config.get('subscribe', []):
            stage.channel_bus.subscribe(channel, self._on_channel_message)
            self._subscribed_channels.append(channel)

        self._publish_channels = set(channels_config.get('publish', []))

    def _on_channel_message(self, message: ChannelMessage):
        """Handle incoming channel message - queue for facet processing."""
        self._pending_channel_messages[message.channel] = message
```

### 5. Facet Integration

Facets need to read channel input and write channel output.

**Reading** - In facet incoming data resolution:
```python
# When resolving incoming for a facet
if incoming_name.startswith('channel:'):
    channel = incoming_name[8:]  # strip 'channel:'
    message = noodling._pending_channel_messages.get(channel)
    if message:
        data[incoming_name] = message.payload
```

**Writing** - In facet output handling:
```python
# When processing facet outputs
if output_name.startswith('channel:'):
    channel = output_name[8:]
    if channel in noodling._publish_channels:
        noodling.stage.channel_bus.publish(channel, ChannelMessage(
            channel=channel,
            from_noodling=noodling.id,
            timestamp=time.time(),
            payload=output_value
        ))
```

### 6. Facets Editor UI (Lower Priority)

In the facets editor, subscribed/published channels should appear as extra pads on INCOMING/OUTGOING nodes. See the ASCII diagrams in channels.md for the visual design.

This can come later - runtime first.

---

## Testing Strategy

1. **Unit test ChannelBus**: pub/sub, history, get_latest
2. **Integration test**: Two noodlings, one publishes, one receives
3. **Let's Consciousness test**: Brenda sends cue, Guide receives it

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `runtime/channels.py` | CREATE - ChannelBus, ChannelMessage |
| `runtime/stage.py` or equivalent | MODIFY - Add channel_bus instance |
| `runtime/noodling.py` or equivalent | MODIFY - Channel subscription, message handling |
| `runtime/facets/executor.py` or equivalent | MODIFY - Channel input/output resolution |
| Assembly schema/validation | MODIFY - Add channels field |

---

## After Implementation

Once channels work, we can build:
1. **Brenda** - Stage director noodling (invisible, sends cues)
2. **World channels** - #world.time, #world.weather for environmental context
3. **Guide upgrade** - Subscribe to #directors.cues, report to #directors.feedback

The goal: Let's Consciousness demo uses channels for all orchestration, no hardcoded logic.

---

## Questions?

Check `/docs/noodlestudio/channels.md` for the full spec including message format, naming conventions, and example YAML configurations.

*"The stage is set. The channels are open. Brenda is waiting in the wings."*
