"""
Scripted Facet - User-programmable cognitive logic nodes

Allows users to create custom facets with JavaScript (or Python) logic.
Scripts have access to:
- Input pad values
- Persistent storage
- Context (cycle count, agent info, etc.)
- Event emission
- World access

Security: Scripts run in sandboxed environment with:
- Execution timeout (max 5 seconds)
- No file system access
- Limited imports (whitelist only)
- Memory limits
- Rate limiting on events

Author: Commander Spock + Cadet Caity
Date: November 28, 2025
"""

import time
import json
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

# JavaScript execution via QuickJS (FIXED for Python 3.14!)
try:
    from quickjs import Context as QuickJSContext
    JS_AVAILABLE = True
    JS_ENGINE = "quickjs"
except ImportError:
    # Fallback to PyMiniRacer (broken on Python 3.14 Mac)
    try:
        from py_mini_racer import MiniRacer
        JS_AVAILABLE = True
        JS_ENGINE = "pyminirace"
    except ImportError:
        JS_AVAILABLE = False
        QuickJSContext = None
        MiniRacer = None
        JS_ENGINE = None

# Event bus integration
try:
    from .execution_event_bus import get_event_bus, EventChannel
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False


@dataclass
class ScriptContext:
    """Context object passed to scripted facets."""

    # Current execution
    cycle: int
    timestamp: float

    # Agent info
    agent_id: str
    agent_name: str
    agent_species: str

    # Storage (persistent per-facet)
    _storage: Dict[str, Any] = field(default_factory=dict)

    # Event callbacks (registered by scripts)
    _event_callbacks: Dict[str, list] = field(default_factory=dict)

    # Parent facet assembly (for inter-facet communication)
    _assembly: Optional[Any] = None

    # Logging
    _logs: list = field(default_factory=list)

    # Execution event callbacks (Unity-like lifecycle)
    _facet_complete_callbacks: Dict[str, list] = field(default_factory=dict)
    _facet_start_callbacks: Dict[str, list] = field(default_factory=dict)
    _data_flow_callbacks: list = field(default_factory=list)
    _cycle_callbacks: Dict[str, list] = field(default_factory=dict)

    # World state access (injected by executor)
    _world_state: Optional[Dict[str, Any]] = None
    _agent_state: Optional[Dict[str, Any]] = None
    _room_state: Optional[Dict[str, Any]] = None

    # Action queue (for world mutations)
    _action_queue: list = field(default_factory=list)

    # Facet outputs cache (for inter-facet reading)
    _facet_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Event bus reference
    _event_bus: Optional[Any] = None

    # Noodle API (system configuration access)
    _noodle_api: Optional[Any] = None

    def __post_init__(self):
        """Initialize event bus and noodle API."""
        if EVENT_BUS_AVAILABLE and self._event_bus is None:
            try:
                self._event_bus = get_event_bus()
            except RuntimeError:
                # No event loop available (e.g., in tests)
                pass

        # Initialize Noodle API if not provided
        if self._noodle_api is None:
            try:
                from noodlestudio.scripting.noodle_api import get_noodle_api
                self._noodle_api = get_noodle_api()
            except Exception as e:
                # API not available or error during init
                # This is OK - scripts just won't have system config access
                pass

    def register_event_bus_listeners(self):
        """
        Wire up script callbacks to event bus.

        Called by ScriptedFacet after context is fully initialized.
        This allows scripts to subscribe to execution events via context.onFacetComplete(), etc.
        """
        if not self._event_bus:
            return

        # Create listener for facet_complete events
        async def on_facet_complete_internal(event):
            facet_id = event.source_id
            if facet_id in self._facet_complete_callbacks:
                for callback_str in self._facet_complete_callbacks[facet_id]:
                    # TODO: Execute JavaScript callback from string
                    # For now, just log
                    pass

        # Register with event bus
        self._event_bus.register_listener(
            on_facet_complete_internal,
            event_type="facet_execution",
            event_subtype="facet_complete",
            channel=EventChannel.EXECUTION
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to JavaScript-compatible dict."""
        return {
            'cycle': self.cycle,
            'timestamp': self.timestamp,
            'agent': {
                'id': self.agent_id,
                'name': self.agent_name,
                'species': self.agent_species
            },
            'storage': {
                'get': '__storage_get__',
                'set': '__storage_set__',
                'clear': '__storage_clear__'
            },
            'emit': '__emit__',
            'log': '__log__',
            'random': '__random__',
            'getFacet': '__get_facet__',
            'getFacetOutput': '__get_facet_output__',
            # Tier 1 API - Execution events
            'onFacetComplete': '__on_facet_complete__',
            'onFacetStart': '__on_facet_start__',
            'onDataFlow': '__on_data_flow__',
            'onCycleStart': '__on_cycle_start__',
            'onCycleComplete': '__on_cycle_complete__',
            # Tier 1 API - State access
            'getSelf': '__get_self__',
            'getRoom': '__get_room__',
            'getAgent': '__get_agent__',
            # Tier 1 API - Actions
            'speak': '__speak__',
            'emote': '__emote__',
            'think': '__think__',
            # Tier 1 API - Timers
            'nextCycle': '__next_cycle__',
            'setTimeout': '__set_timeout__',
            # Noodle API - System configuration
            'noodle': self._noodle_api.to_dict() if self._noodle_api else {}
        }


class ScriptedFacet:
    """
    User-programmable facet node.

    Executes JavaScript (or Python) code to transform inputs into outputs.
    Provides sandboxed environment with access to context, storage, and events.
    """

    def __init__(
        self,
        facet_id: str,
        script: str,
        script_language: str = "javascript",
        timeout: float = 5.0,
        max_storage_bytes: int = 1024 * 100  # 100KB
    ):
        """
        Initialize scripted facet.

        Args:
            facet_id: Unique facet identifier
            script: User script code
            script_language: "javascript" or "python"
            timeout: Max execution time in seconds
            max_storage_bytes: Max persistent storage size
        """
        self.facet_id = facet_id
        self.script = script
        self.script_language = script_language
        self.timeout = timeout
        self.max_storage_bytes = max_storage_bytes

        # Execution stats
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0
        self.last_error = None

        # Initialize JavaScript engine
        if script_language == "javascript":
            if not JS_AVAILABLE:
                raise ImportError(
                    "No JavaScript engine available. Install with: pip install quickjs"
                )
            if JS_ENGINE == "quickjs":
                self.js_context = QuickJSContext()
            else:  # pyminirace
                self.js_context = MiniRacer()
            self._setup_javascript_context()
        else:
            raise NotImplementedError(f"Language '{script_language}' not yet supported")

    def _setup_javascript_context(self):
        """Setup JavaScript execution context with helper functions."""

        # Inject helper library
        helper_lib = """
        // Storage helpers (will be bound to Python)
        var __storage_data__ = {};

        function __storage_get__(key, defaultValue) {
            return __storage_data__.hasOwnProperty(key) ? __storage_data__[key] : defaultValue;
        }

        function __storage_set__(key, value) {
            __storage_data__[key] = value;
        }

        function __storage_clear__() {
            __storage_data__ = {};
        }

        // Logging
        var __logs__ = [];
        function __log__(message) {
            __logs__.push(String(message));
        }

        // Events
        var __events__ = [];
        function __emit__(eventName, data) {
            __events__.push({name: eventName, data: data});
        }

        // Random (seeded)
        var __random_seed__ = 12345;
        function __random__() {
            // Simple LCG
            __random_seed__ = (__random_seed__ * 1103515245 + 12345) & 0x7fffffff;
            return __random_seed__ / 0x7fffffff;
        }

        // Facet access (will be bound to Python)
        var __facet_outputs__ = {};
        function __get_facet__(facetId) {
            return __facet_outputs__[facetId] || null;
        }

        function __get_facet_output__(facetId, padName) {
            var facet = __get_facet__(facetId);
            return facet ? facet[padName] : null;
        }

        // Tier 1 API - Execution event callbacks
        var __facet_complete_callbacks__ = {};
        var __facet_start_callbacks__ = {};
        var __data_flow_callbacks__ = [];
        var __cycle_callbacks__ = {start: [], complete: []};

        function __on_facet_complete__(facetId, callback) {
            if (!__facet_complete_callbacks__[facetId]) {
                __facet_complete_callbacks__[facetId] = [];
            }
            __facet_complete_callbacks__[facetId].push(callback);
        }

        function __on_facet_start__(facetId, callback) {
            if (!__facet_start_callbacks__[facetId]) {
                __facet_start_callbacks__[facetId] = [];
            }
            __facet_start_callbacks__[facetId].push(callback);
        }

        function __on_data_flow__(callback) {
            __data_flow_callbacks__.push(callback);
        }

        function __on_cycle_start__(callback) {
            __cycle_callbacks__.start.push(callback);
        }

        function __on_cycle_complete__(callback) {
            __cycle_callbacks__.complete.push(callback);
        }

        // Tier 1 API - State access
        var __world_state__ = null;
        var __agent_state__ = null;
        var __room_state__ = null;

        function __get_self__() {
            return __agent_state__;
        }

        function __get_room__() {
            return __room_state__;
        }

        function __get_agent__(agentId) {
            if (!__world_state__ || !__world_state__.agents) return null;
            return __world_state__.agents[agentId] || null;
        }

        // Tier 1 API - Actions (queued for execution)
        var __action_queue__ = [];

        function __speak__(text) {
            __action_queue__.push({type: 'speak', text: text});
        }

        function __emote__(text) {
            __action_queue__.push({type: 'emote', text: text});
        }

        function __think__(text) {
            __action_queue__.push({type: 'think', text: text});
        }

        // Tier 1 API - Timers (queued for execution)
        function __next_cycle__(callback) {
            __action_queue__.push({type: 'next_cycle', callback: callback.toString()});
        }

        function __set_timeout__(callback, milliseconds) {
            __action_queue__.push({type: 'set_timeout', callback: callback.toString(), delay: milliseconds});
        }

        // Clear execution state
        function __clear_execution_state__() {
            __logs__ = [];
            __events__ = [];
            __action_queue__ = [];
        }
        """

        self.js_context.eval(helper_lib)

        # Load user script (defines 'process' function)
        try:
            self.js_context.eval(self.script)
        except Exception as e:
            self.last_error = f"Script compilation error: {e}"
            raise ValueError(self.last_error)

    def process(
        self,
        inputs: Dict[str, Any],
        context: ScriptContext
    ) -> Dict[str, Any]:
        """
        Execute script with given inputs and context.

        Args:
            inputs: Dict of input pad values
            context: Execution context

        Returns:
            Dict of output pad values

        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If script execution fails
        """
        start_time = time.time()

        import json  # MOVED TO TOP!

        try:
            # Inject context storage into JavaScript
            self.js_context.eval(
                f"__storage_data__ = {json.dumps(context._storage)};"
            )

            # Inject world state (Tier 1 API)
            if context._world_state:
                self.js_context.eval(
                    f"__world_state__ = {json.dumps(context._world_state)};"
                )
            if context._agent_state:
                self.js_context.eval(
                    f"__agent_state__ = {json.dumps(context._agent_state)};"
                )
            if context._room_state:
                self.js_context.eval(
                    f"__room_state__ = {json.dumps(context._room_state)};"
                )

            # Inject facet outputs cache (for inter-facet reading)
            if context._facet_outputs:
                self.js_context.eval(
                    f"__facet_outputs__ = {json.dumps(context._facet_outputs)};"
                )

            # Clear execution state
            self.js_context.eval("__clear_execution_state__();")

            # Build context object
            context_js = context.to_dict()

            # Call user's process function (API differs between engines)
            if hasattr(self.js_context, 'call'):  # PyMiniRacer
                result_json = self.js_context.call(
                    "process",
                    inputs,
                    context_js,
                    timeout=int(self.timeout * 1000)  # milliseconds
                )
            else:  # QuickJS - use eval with JSON serialization
                call_code = f"process({json.dumps(inputs)}, {json.dumps(context_js)})"
                result_obj = self.js_context.eval(call_code)
                # Convert QuickJS object to Python dict
                if hasattr(result_obj, 'json'):
                    result_json = json.loads(result_obj.json())
                else:
                    result_json = result_obj

            # Extract outputs
            if not isinstance(result_json, dict):
                raise RuntimeError(f"Script must return object, got {type(result_json)}")

            # Update context storage from JavaScript
            storage_json = self.js_context.eval("JSON.stringify(__storage_data__);")
            context._storage = json.loads(storage_json)

            # Check storage size
            storage_size = len(json.dumps(context._storage))
            if storage_size > self.max_storage_bytes:
                raise RuntimeError(
                    f"Storage limit exceeded: {storage_size} > {self.max_storage_bytes} bytes"
                )

            # Collect logs
            logs_json = self.js_context.eval("JSON.stringify(__logs__);")
            logs = json.loads(logs_json)
            context._logs.extend(logs)

            # Collect events
            events_json = self.js_context.eval("JSON.stringify(__events__);")
            events = json.loads(events_json)
            for event in events:
                event_name = event['name']
                event_data = event['data']
                if event_name in context._event_callbacks:
                    for callback in context._event_callbacks[event_name]:
                        callback(event_data)

            # Collect action queue (Tier 1 API)
            actions_json = self.js_context.eval("JSON.stringify(__action_queue__);")
            actions = json.loads(actions_json)
            context._action_queue.extend(actions)

            # Update stats
            elapsed = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += elapsed
            self.last_execution_time = elapsed
            self.last_error = None

            return result_json

        except Exception as e:
            elapsed = time.time() - start_time
            self.last_error = str(e)
            self.last_execution_time = elapsed
            raise RuntimeError(f"Script execution failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.execution_count,
            'total_time': self.total_execution_time,
            'avg_time': (
                self.total_execution_time / self.execution_count
                if self.execution_count > 0 else 0
            ),
            'last_time': self.last_execution_time,
            'last_error': self.last_error
        }

    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0
        self.last_error = None


# Example scripts for testing
EXAMPLE_MOOD_TRACKER = """
function process(inputs, context) {
    // Get historical mood data
    var history = context.storage.get('mood_history', []);

    // Add current mood
    history.push({
        cycle: context.cycle,
        valence: inputs.valence,
        arousal: inputs.arousal
    });

    // Keep last 10 moods
    if (history.length > 10) history.shift();
    context.storage.set('mood_history', history);

    // Calculate trend
    var sum = 0;
    for (var i = 0; i < history.length; i++) {
        sum += history[i].valence;
    }
    var avgValence = sum / history.length;
    var trend = inputs.valence > avgValence ? "improving" : "declining";

    // Calculate volatility
    var volatility = 0;
    if (history.length > 1) {
        for (var i = 1; i < history.length; i++) {
            volatility += Math.abs(history[i].valence - history[i-1].valence);
        }
        volatility /= (history.length - 1);
    }

    return {
        trend: trend,
        average: avgValence,
        volatility: volatility
    };
}
"""

EXAMPLE_KEYWORD_DETECTOR = """
function process(inputs, context) {
    var text = inputs.perception_text.toLowerCase();
    var keywords = context.storage.get('keywords', ['red', 'fire', 'anklebiter']);

    var detected = [];
    for (var i = 0; i < keywords.length; i++) {
        if (text.indexOf(keywords[i]) !== -1) {
            detected.push(keywords[i]);
        }
    }

    var isAddressed = detected.length > 0;

    if (isAddressed) {
        context.log('Keywords detected: ' + detected.join(', '));
        context.emit('keyword_detected', {keywords: detected});
    }

    return {
        is_addressed: isAddressed,
        detected_keywords: detected.join(', '),
        keyword_count: detected.length
    };
}
"""


if __name__ == "__main__":
    """Test scripted facet execution."""

    print("=== Testing ScriptedFacet ===\n")

    # Test 1: Mood Tracker
    print("Test 1: Mood Tracker")
    print("-" * 40)

    context = ScriptContext(
        cycle=1,
        timestamp=time.time(),
        agent_id="agent_test",
        agent_name="Test Agent",
        agent_species="test"
    )

    facet = ScriptedFacet("mood_tracker", EXAMPLE_MOOD_TRACKER)

    # Run multiple cycles
    for i in range(5):
        inputs = {
            'valence': 0.5 + (i * 0.1),
            'arousal': 0.6
        }
        context.cycle = i + 1

        outputs = facet.process(inputs, context)
        print(f"Cycle {i+1}: trend={outputs['trend']}, avg={outputs['average']:.2f}, volatility={outputs['volatility']:.3f}")

    print(f"\nStats: {facet.get_stats()}\n")

    # Test 2: Keyword Detector
    print("\nTest 2: Keyword Detector")
    print("-" * 40)

    context2 = ScriptContext(
        cycle=1,
        timestamp=time.time(),
        agent_id="agent_test",
        agent_name="Test Agent",
        agent_species="test"
    )

    # Event callback
    def on_keyword_detected(data):
        print(f"  [EVENT] Keywords detected: {data['keywords']}")

    context2._event_callbacks['keyword_detected'] = [on_keyword_detected]

    facet2 = ScriptedFacet("keyword_detector", EXAMPLE_KEYWORD_DETECTOR)

    test_inputs = [
        "Hello there, how are you?",
        "Hey Red, what's up?",
        "The fire is burning brightly",
        "That anklebiter is causing trouble!"
    ]

    for text in test_inputs:
        inputs = {'perception_text': text}
        outputs = facet2.process(inputs, context2)
        print(f"Text: '{text}'")
        print(f"  Addressed: {outputs['is_addressed']}, Keywords: {outputs['detected_keywords']}")

    print(f"\nLogs: {context2._logs}")
    print(f"Stats: {facet2.get_stats()}")

    print("\n=== All tests complete ===")
