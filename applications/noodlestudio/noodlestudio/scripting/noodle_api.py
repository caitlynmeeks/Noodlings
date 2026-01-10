# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Noodle API - Unified scriptable interface to Noodlings systems.
#
#   Main entry point for ScriptedFacet access to: - Models/pr...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.noodle_api
# PURPOSE:  Noodle Api
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NoodleAPI, get_noodle_api()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional
from .models_api import ModelsAPI
from .neural_api import NeuralAPI
from .agents_api import AgentsAPI
from .quantum_api import QuantumAPI
from .audio_api import AudioAPI, get_audio_api
from .vision_api import VisionAPI, get_vision_api
from .cloud_api import CloudAPI, CloudAPIJS, get_cloud_api
from .world_api import WorldAPI, get_world_api
from .affect_api import AffectAPI, get_affect_api
from .pose_api import PoseAPI, get_pose_api
from .radiance_api import RadianceAPI, get_radiance_api


class NoodleAPI:
    """
    Unified Noodlings scripting API.

    Provides JavaScript-accessible interface to all system components.
    Available in ScriptedFacets via context.noodle

    Example (JavaScript in ScriptedFacet):
        function process(inputs, context) {
            // Change model for LARGE label
            context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");

            // Modify neural topology
            var network = context.noodle.neural.get_network("default");
            var lstm = network.create_node("LSTM", {hidden_dim: 64});

            // Reconfigure facet assembly
            var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
            var facet = assembly.get_facet("CHARM_NET");
            facet.set_property("model", "LARGE");

            return {modified: true};
        }
    """

    def __init__(
        self,
        model_label_manager=None,
        provider_manager=None
    ):
        """
        Initialize Noodle API.

        Args:
            model_label_manager: ModelLabelManager instance (optional, lazy init)
            provider_manager: ProviderManager instance (optional, lazy init)
        """
        # Sub-API instances
        self._models_api = None
        self._neural_api = NeuralAPI()
        self._agents_api = AgentsAPI()
        self._quantum_api = QuantumAPI()
        self._audio_api = None  # Lazy init
        self._vision_api = None  # Lazy init
        self._cloud_api = None  # Lazy init
        self._world_api = None  # Lazy init - per-noodling, set via set_world_api()
        self._affect_api = None  # Lazy init - affect animation tracks
        self._pose_api = None    # Lazy init - pose/body animation tracks
        self._radiance_api = None  # Lazy init - Gaussian splat visuals

        # Manager references (lazy initialization)
        self._model_label_manager = model_label_manager
        self._provider_manager = provider_manager

        # UUID registry (future enhancement)
        self._uuid_registry: Dict[str, Any] = {}

    @property
    def models(self) -> ModelsAPI:
        """
        Access Models API.

        Returns:
            ModelsAPI instance

        Example (JavaScript):
            var assignment = context.noodle.models.get_label("SMALL");
        """
        if self._models_api is None:
            # Lazy initialization
            if self._model_label_manager is None:
                from noodlestudio.core.model_label_manager import get_model_label_manager
                self._model_label_manager = get_model_label_manager()

            if self._provider_manager is None:
                from noodlestudio.core.provider_manager import ProviderManager
                self._provider_manager = ProviderManager()

            self._models_api = ModelsAPI(
                self._model_label_manager,
                self._provider_manager
            )

        return self._models_api

    @property
    def neural(self) -> NeuralAPI:
        """
        Access Neural Canvas API.

        Returns:
            NeuralAPI instance

        Example (JavaScript):
            var network = context.noodle.neural.create_network("MyNet");
        """
        return self._neural_api

    @property
    def agents(self) -> AgentsAPI:
        """
        Access Agents API.

        Returns:
            AgentsAPI instance

        Example (JavaScript):
            var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
        """
        return self._agents_api

    @property
    def quantum(self) -> QuantumAPI:
        """
        Access Quantum API.

        Provides quantum computation simulation for ScriptedFacets:
        - Qubit measurement with simulated quantum randomness
        - Schrodinger's Cat experiment helper
        - Neural Canvas quantum node execution
        - Entanglement simulation

        Returns:
            QuantumAPI instance

        Example (JavaScript):
            // Measure a qubit
            var q = context.noodle.quantum.measure_qubit();

            // Schrodinger's Cat experiment
            var cat = context.noodle.quantum.schrodingers_cat();
            if (cat.is_alive) {
                console.log("The cat lives!");
            }

            // Execute a quantum canvas
            var result = context.noodle.quantum.execute_canvas(
                "tutorials/08_schrodingers_cat.nncanvas"
            );
        """
        return self._quantum_api

    @property
    def audio(self) -> AudioAPI:
        """
        Access Audio API.

        Provides real-time audio I/O for ScriptedFacets:
        - Speech-to-text transcription (Whisper)
        - Text-to-speech synthesis (ElevenLabs, etc.)
        - Voice activity detection
        - Interrupt handling

        Returns:
            AudioAPI instance

        Example (JavaScript):
            // Speak text
            context.noodle.audio.speak("Hello there!");

            // Check state
            if (context.noodle.audio.isSpeaking) {
                context.log("Currently speaking...");
            }

            // Get transcription
            var text = context.noodle.audio.lastTranscription;

            // Start listening
            context.noodle.audio.listen();

            // Event subscription (in salience script)
            context.noodle.audio.onTranscriptionReady((text) => {
                context.log("User said: " + text);
            });
        """
        if self._audio_api is None:
            self._audio_api = get_audio_api()
        return self._audio_api

    @property
    def vision(self) -> VisionAPI:
        """
        Access Vision API.

        Provides image understanding and generation for ScriptedFacets:
        - Image analysis (Claude Vision, GPT-4V, LLaVA)
        - Screenshot capture
        - Image generation (DALL-E, Flux, Stable Diffusion)
        - Image memory with hybrid storage
        - Semantic image search

        Returns:
            VisionAPI instance

        Example (JavaScript):
            // Analyze an image
            var result = context.noodle.vision.analyze("/path/to/image.png");
            context.log("I see: " + result.description);

            // Generate an image
            context.noodle.vision.generate("a sunset over mountains", "artistic");

            // Search image memory
            var cats = context.noodle.vision.searchImages("cat");

            // Capture screenshot
            var screen = context.noodle.vision.screenshot();
        """
        if self._vision_api is None:
            self._vision_api = get_vision_api()
        return self._vision_api

    @property
    def cloud(self) -> CloudAPIJS:
        """
        Access Cloud API.

        Provides cloud account and storage integration for ScriptedFacets:
        - User authentication state
        - Credit balance checking
        - Noodling cloud sync (save/load)
        - Routed LLM generation (uses credits)
        - Asset Store browsing

        Note: Some operations (like generate) are async and use events.

        Returns:
            CloudAPIJS instance (JavaScript-friendly wrapper)

        Example (JavaScript):
            // Check if logged in
            if (context.noodle.cloud.isAuthenticated()) {
                var user = context.noodle.cloud.getUser();
                context.log("Logged in as: " + user.email);
                context.log("Credits: " + user.creditsBalance);
            }

            // Get credit balance
            var balance = context.noodle.cloud.getCredits();

            // Estimate LLM cost
            var estimate = context.noodle.cloud.estimateCost({
                model: "anthropic/claude-3-sonnet",
                messages: [{role: "user", content: "Hello!"}]
            });
            context.log("Estimated cost: " + estimate.estimated_credits);

            // List available models
            var models = context.noodle.cloud.listModels();

            // Save noodling to cloud
            context.noodle.cloud.saveNoodling({
                name: "my-noodling",
                displayName: "My Noodling",
                recipe: recipeYaml,
                facetAssembly: assemblyYaml
            });

            // Browse public noodlings
            var store = context.noodle.cloud.browseStore({tag: "fantasy"});
        """
        if self._cloud_api is None:
            self._cloud_api = CloudAPIJS(get_cloud_api())
        return self._cloud_api

    @property
    def affect(self) -> AffectAPI:
        """
        Access Affect Animation Track API.

        Provides keyframeable emotional curves for ScriptedFacets:
        - Load and play affect tracks (.affecttrack files)
        - Control playback (play, pause, seek, speed)
        - Sample affect values at any time
        - Blend authored tracks with live CharmNetwork affect
        - Emotional momentum handoff (Donald Duck problem)
        - Arbitrary affect dimensions (PAD+BS default, extensible)

        Returns:
            AffectAPI instance

        Example (JavaScript):
            // Load and play a track
            var track = context.noodle.affect.loadTrack("grief_reaction.affecttrack");
            track.play();

            // Control playback
            track.pause();
            track.seek(3.5);
            track.speed = 0.5;

            // Sample current affect
            var state = context.noodle.affect.getState();
            context.log("Valence: " + state.valence);

            // Blend track with live CharmNetwork
            context.noodle.affect.setBlendMode("weighted", {track: 0.7, live: 0.3});

            // Listen for markers
            track.onMarker("tears_start", function() {
                context.noodle.events.emit("start_tears", {intensity: 0.6});
            });

            // Inject affect directly (momentum)
            context.noodle.affect.inject({
                valence: -0.5,
                arousal: 0.8
            }, "natural");

            // Play with momentum handoff when track ends
            track.play({
                onComplete: "momentum",
                transferScale: 0.9
            });
        """
        if self._affect_api is None:
            self._affect_api = get_affect_api()
        return self._affect_api

    @property
    def pose(self) -> PoseAPI:
        """
        Access Pose Animation Track API.

        Provides rig-agnostic body animation for ScriptedFacets:
        - Load and play pose tracks (.posetrack files)
        - Control playback (play, pause, seek, speed)
        - Sample muscle values at any time
        - Direct muscle control for procedural animation
        - Mecanim-style muscle space (~47 standard muscles)
        - Retargeting to any humanoid avatar

        Returns:
            PoseAPI instance

        Example (JavaScript):
            // Load and play a pose track
            var wave = context.noodle.pose.loadTrack("wave.posetrack");
            wave.play();

            // Sample muscles at current time
            var muscles = wave.getMuscles();
            context.log("Arm: " + muscles["RightArm.DownUp"]);

            // Direct muscle control (procedural)
            context.noodle.pose.setMuscle("Head.NodDownUp", 0.5);

            // Get bone rotations after retargeting
            var bones = context.noodle.pose.getBoneRotations();

            // Momentum handoff when track ends
            wave.onComplete(function(finalPose) {
                context.noodle.pose.setMomentum(finalPose.muscles, {
                    decay: "spring",
                    stiffness: 0.5
                });
            });
        """
        if self._pose_api is None:
            self._pose_api = get_pose_api()
        return self._pose_api

    @property
    def radiance(self) -> RadianceAPI:
        """
        Access Radiance API.

        Provides Gaussian splat visual component control for ScriptedFacets:
        - Load and manage RadianceComponents
        - Entity-level material overrides (tint, emission, alpha)
        - Region-level overrides (by body part)
        - Per-Gaussian overrides (for FX like dissolve, damage)
        - Spatial queries (raycast, radius search)
        - Scene composition and lighting

        Returns:
            RadianceAPI instance

        Example (JavaScript):
            // Get entity's radiance component
            var red = context.noodle.radiance.get("red_fire_anklebiter");

            // Tint based on affect
            var arousal = context.noodle.affect.getArousal();
            red.set_tint(1.0, 0.5 + arousal * 0.5, 0.5);

            // Make left arm glow when excited
            if (arousal > 0.7) {
                red.set_region_override("left_arm", {
                    emission: {r: 0.3, g: 0, b: 0}
                });
            }

            // Scene-wide raycast
            var hit = context.noodle.radiance.scene.raycast(0, 1, 0, 0, 0, -1);
            if (hit.hit) {
                context.log("Looking at: " + hit.body_part + " of " + hit.entity_id);
            }

            // Per-Gaussian FX (damage decal)
            var nearby = red.query_radius(impactX, impactY, impactZ, 0.1);
            for (var i = 0; i < nearby.length; i++) {
                red.set_gaussian_override(nearby[i], {
                    tint: {r: 0.2, g: 0.2, b: 0.2}  // Burn mark
                });
            }
        """
        if self._radiance_api is None:
            self._radiance_api = get_radiance_api()
        return self._radiance_api

    @property
    def world(self) -> WorldAPI:
        """
        Access World API.

        Provides perception-filtered access to scene state for ScriptedFacets.
        Each noodling sees only what they can perceive (information asymmetry).

        Properties (read-only):
        - perceivedEntities: List of entities I can see
        - perceivedEvents: List of events I witnessed
        - myPosition, myFacing, myZone, myZoneName
        - myPosture, myAction, myExpression, myGaze
        - affect: {valence, arousal, dominance, boredom, sorrow}
        - zoneExits, ambientSounds, ambientMood
        - conversationPartner, lastInput

        Query methods:
        - canSee(entityId): Check if I can see someone
        - canHear(entityId): Check if I heard someone recently
        - getEntity(entityId): Get observable info about an entity
        - getDistanceTo(entityId): Distance in meters
        - getDirectionTo(entityId): "in_front", "left", "behind", etc.
        - isLookingAtMe(entityId): Is entity looking at me?
        - getRecentSpeech(limit): Recent dialogue I heard

        Command methods:
        - setExpression(expr): Change my facial expression
        - setPosture(posture): Change my posture
        - setGaze(target): Look at something/someone
        - speak(text, tone): Say something
        - emote(text): Do an action
        - moveTo(x, y, z): Move to position

        Camera methods (if enabled):
        - focusCamera(entity, framing): Focus on entity
        - twoShot(a, b, framing): Frame two entities
        - povCamera(entity): Switch to POV
        - establishShot(zone): Wide zone shot

        Returns:
            WorldAPI instance for this noodling

        Example (JavaScript):
            // Who can I see?
            var entities = context.noodle.world.perceivedEntities;
            for (var i = 0; i < entities.length; i++) {
                context.log("I see: " + entities[i].displayName);
            }

            // Can I see Yuki?
            if (context.noodle.world.canSee("yuki")) {
                var yuki = context.noodle.world.getEntity("yuki");
                context.log("Yuki is " + yuki.direction);
            }

            // React to someone looking at me
            if (context.noodle.world.isLookingAtMe("caity")) {
                context.noodle.world.setGaze("caity");
                context.noodle.world.setExpression("curious");
            }

            // Say something
            context.noodle.world.speak("Hello!", "friendly");
        """
        if self._world_api is None:
            # Default instance - should be set per-noodling via set_world_api()
            self._world_api = get_world_api("default")
        return self._world_api

    def set_world_api(self, world_api: WorldAPI):
        """
        Set the WorldAPI instance for this noodling.

        Called by FacetExecutor to provide noodling-specific
        perception-filtered world access.

        Args:
            world_api: WorldAPI instance configured for this noodling
        """
        self._world_api = world_api

    def get_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get any entity by UUID (future enhancement).

        Args:
            uuid: Universal unique identifier

        Returns:
            Entity dict with {type, properties, methods} or None

        Example (JavaScript):
            var entity = context.noodle.get_by_uuid("550e8400-...");
            console.log(entity.type);  // "LLMFacet", "LSTMNode", etc.
        """
        # TODO: Implement UUID registry
        return self._uuid_registry.get(uuid)

    def register_entity(self, uuid: str, entity: Any):
        """
        Register entity in UUID registry (internal use).

        Args:
            uuid: Entity UUID
            entity: Entity object
        """
        self._uuid_registry[uuid] = entity

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JavaScript-compatible dict for context injection.

        Returns:
            Dict with sub-API method names as placeholders

        This is used by ScriptContext to inject the API into JavaScript.
        """
        return {
            'models': self.models.to_dict(),
            'neural': self.neural.to_dict(),
            'agents': self.agents.to_dict(),
            'quantum': self.quantum.to_dict(),
            'audio': self.audio.to_dict(),
            'vision': self.vision.to_dict(),
            'affect': self.affect.to_dict(),
            'pose': self.pose.to_dict(),
            'world': self.world.to_dict(),
            'radiance': '__noodle_radiance__',  # Special handling for RadianceAPI
            'get_by_uuid': '__noodle_get_by_uuid__'
        }


# Global singleton instance
_noodle_api_instance = None


def get_noodle_api(
    model_label_manager=None,
    provider_manager=None
) -> NoodleAPI:
    """
    Get global NoodleAPI singleton.

    Args:
        model_label_manager: ModelLabelManager (optional)
        provider_manager: ProviderManager (optional)

    Returns:
        NoodleAPI instance
    """
    global _noodle_api_instance

    if _noodle_api_instance is None:
        _noodle_api_instance = NoodleAPI(
            model_label_manager=model_label_manager,
            provider_manager=provider_manager
        )

    return _noodle_api_instance

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
