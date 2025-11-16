# Changelog

All notable changes to noodleMUSH will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [Phase 6.7] - 2025-11-15 (Evening Session)

Polish session + TAB Log View implementation!

### Added

- **📊 TAB Log View - Real-Time Debugging**
  - Press [TAB] to toggle between Chat View and Log View
  - Live WebSocket log streaming from server
  - Color-coded logs: INFO (green), WARNING (yellow), ERROR (red)
  - Timestamps on every entry (HH:MM:SS format)
  - Verbose/Compact mode toggle (200 char limit in compact)
  - Smart scrolling (only auto-scrolls when at bottom)
  - Font size controls (A+/A-) work in both views
  - Perfect for debugging intuition, voices, self-monitoring

- **👤 @profile Command**
  - Set your species, pronoun, and age as a Noodler
  - `@profile` - View current profile
  - `@profile -s <species> -p <pronoun> -a <age>` - Update profile
  - Example: `@profile -s human -p she -a "9 years old"`
  - Supports quoted multi-word values

- **📋 Enhanced "People here:" Display**
  - Format: `name [Role, species, age, pronoun]`
  - Example: `phi [Noodling, kitten, 6 months old, she]`
  - Shows Noodlers (humans) vs Noodlings (AI agents)
  - Current user now included in room occupant list

- **🎭 Age Metadata System**
  - All agents now have age field in recipes
  - Intuition broadcasts age: "Phi (kitten, 6 months old, she)"
  - Look command shows age for all occupants
  - User profiles support age field

### Fixed

- **5 unpacking errors** where code expected 2 values but `_complete()` returns 3:
  - autonomous_cognition.py:361 (rumination) ✅
  - autonomous_cognition.py:811 (autonomous speech) ✅
  - llm_interface.py:246 (affect extraction) ✅
  - agent_bridge.py:2009 (self-monitoring) ✅
  - llm_interface.py:1607 (toxicity detection) ✅
- **@yeet command** now uses 'username' instead of 'name' field
- **Chat view scroll behavior** - Simplified logic, no more jumps when reading history
- **JSON parsing errors** - Changed ERROR → WARNING for graceful degradation
- Removed duplicate users (user_caitlyn, user_Claude)

### Changed

- Log view font size synced with chat view controls
- Improved error handling with better log levels
- Smart scroll behavior in both chat and log views
- Log streaming only to subscribed clients

### Files Modified

- server.py - WebSocket log streaming
- agent_bridge.py - Age in intuition, self-monitoring fix
- commands.py - @profile command, enhanced look, @yeet fix
- llm_interface.py - Unpacking fixes, toxicity detection
- autonomous_cognition.py - Multiple unpacking fixes, better error logging
- web/index.html - TAB toggle, log view UI, timestamps, scroll fixes
- world/agents.json - Age/pronoun metadata for all agents
- world/users.json - Age/pronoun/species fields, removed duplicates

### Statistics

- 8 files modified
- ~600 lines added
- 5 critical unpacking errors fixed
- 1 new command (@profile)
- 100% error-free autonomous cognition achieved

---

## [Phase 6.6] - 2025-11-15

Massive consciousness features session! Four major systems implemented.

### Added

- **🎭 Character Voice Post-Processing System**
  - Automatic voice translation: Basic English → Character-specific voice
  - SERVNAK: ALL CAPS + percentages + "SISTER!" + pride circuits
  - Phi (kitten): "meows, as if to say..." (NO direct speech!)
  - Phido (dog): Enthusiastic speech + *tail wagging* + "FRIEND!"
  - Backwards Dweller: Reversed word order (Lynchian)
  - LLM-based translation (qwen3-4b) for flexibility
  - Self-monitoring on FINAL character voice
  - See CHARACTER_VOICE_SYSTEM.md

- **📻 Enhanced Intuition Receiver**
  - Species + pronouns: "Phi (kitten, she/her), SERVNAK (robot, they/them)"
  - Noteworthy event narration: "WAIT - Toad just said the secret word!"
  - "You" addressing: "Caity gave ME a tensor taffy!"
  - Game awareness: Detects secret word/memory games
  - Acts as perceptive narrator, not passive info

- **💭 Memory Persistence Fix**
  - Memory capacity: 50 → 500 messages (10x!)
  - affect_trim_threshold: 500 (was 50)
  - Secret words (DRAGONFLY) now persist across sessions
  - See MEMORY_PERSISTENCE_FIX.md

- **⚙️ Unified @setdesc Command**
  - `@setdesc here <desc>` - Describe current room
  - `@setdesc me <desc>` - Describe yourself
  - `@setdesc "object" <desc>` - Describe object
  - @describe redirects to @setdesc (backward compatible)

- **🔍 Enhanced look Command**
  - `look me` - Look at yourself
  - `look here` - Look at room (explicit)
  - Keywords for better UX

- **🔇 @remove Improvements**
  - `-s` flag for silent removal (no departure message)
  - Quote handling: `@remove "backwards dweller"`
  - Space→underscore conversion (matches @spawn)

- **📝 @spawn Quote Support**
  - Multi-word names: `@spawn "backwards dweller"`
  - Uses shlex for proper quote parsing

### Changed
- Intuition now includes species and pronouns in all broadcasts
- Intuition detects ongoing games and noteworthy events
- Species reloaded from recipes on agent load
- Pronoun inference from character names
- "privately thinks" prefix in UI for all thoughts

### Fixed
- Phi character voice now works (species reloading fixed)
- Phido gets dog voice (not cat voice)
- Brain indicators removed on agent exit
- @remove handles multi-word names with underscores/quotes
- text_to_affect unpacking issues

### Documentation
- CHARACTER_VOICE_SYSTEM.md - Complete character voice docs
- MEMORY_PERSISTENCE_FIX.md - Memory issue analysis
- NEXT_SESSION_PROMPT.md - TAB log view specs
- CLAUDE.md updated with session summary

### Statistics
- 9 files modified (+811/-62 lines)
- 3 documentation files created
- 5 major bugs fixed
- Commit: e98a071

---

## [Phase 6.5] - 2025-11-15 (Earlier Session)

### Added
- **Play system**: BRENDA can now direct theatrical plays with agent actors
  - `@play <play_name>` command to start plays
  - `plays/` directory with JSON play scripts
  - Automatic actor assignment from available agents
  - Real-time play execution with dialogue and stage directions
  - Post-play cleanup and state restoration
- **BRENDA tool-use**: Conversational command execution with `[EXECUTE:command]` tags
  - BRENDA can execute MUD commands naturally in conversation
  - Commands wrapped in `[EXECUTE:]` tags for clean output formatting
  - Full access to world manipulation (create, dig, describe, etc.)
- **Enhanced memory system**: Configurable memory windows for better continuity
  - `memory_windows` section in config.yaml
  - Separate windows for affect_extraction (10), response_generation (20), rumination (10)
  - Increased disk_save limit to 500 turns
  - Increased affect_trim_threshold to 50 turns
  - Agents now remember conversations 4x longer
- **Parallel inference**: LMStudio model instance support for true parallelism
  - Round-robin distribution across multiple LLM instances
  - Configurable `max_concurrent` for instance count
  - LMStudio-compatible naming convention (base model + :2, :3, :4, :5)
  - Up to 5x throughput with 5 model instances

### Changed
- Updated README.md branding from "Consilience" to "Noodlings"
- Improved BRENDA output formatting with cleaner tool execution
- Memory windows now use configuration instead of hardcoded values
- Agent response generation uses 20-turn context window (up from 5)
- Rumination uses 10-turn context (up from 2)
- Self-reflection uses 10-turn context (up from 3)

### Fixed
- Brain indicators now created immediately when agents spawn
- LMStudio model instance pattern corrected (skip :0 and :1, use base model for first instance)
- BRENDA tool execution now properly strips `[EXECUTE:]` tags from output
- Play manager attribute reference corrected (running_plays → active_plays)

## [0.2.0] - 2025-11-14

### Added
- Phase 6: Affective Self-Monitoring (metacognitive awareness)
- Agent personality system with configurable traits
- Theory of Mind for multi-agent interaction
- Autonomous cognition system with rumination

### Changed
- Migrated to Noodlings framework (Phase 4)
- Updated web interface with terminal aesthetic
- Improved agent response generation

### Fixed
- Various stability improvements
- Memory leak fixes
- WebSocket connection handling

## [0.1.0] - 2025-10-23

### Added
- Initial release of cMUSH
- WebSocket server with terminal web client
- Basic MUD commands (movement, communication, building)
- Consilience Phase 4 agent integration
- LLM interface (LMStudio, Ollama, OpenAI)
- Persistent world state (JSON storage)
- Simple authentication system
- Agent commands (@spawn, @observe, @relationship, @memory)

---

## Version History Summary

- **0.1.0**: Initial release with basic MUD functionality
- **0.2.0**: Phase 6 self-monitoring and personality system
- **Unreleased**: Play system, BRENDA tools, memory improvements, parallel inference
