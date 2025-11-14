# Changelog

All notable changes to noodleMUSH will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
