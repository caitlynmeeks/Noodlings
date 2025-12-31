# NoodleMUSH Quickstart

Get a server running in 5 minutes.

---

## Prerequisites

- Python 3.10+
- macOS 13+ (Apple Silicon recommended)
- Ollama running locally (or other LLM provider)

## Start the Server

```bash
cd applications/cmush
./start.sh
```

## Connect

Open http://localhost:8080 in your browser.

## Basic Commands

```
@rez red                    # Spawn a Noodling named Red
say Hello!                  # Talk to everyone in the room
@observe red                # View Red's phenomenal state
look                        # Describe the current room
@derez red                  # Remove Red from the world
```

## What Just Happened?

When you `@rez red`:
1. Server loads Red's recipe from `Noodlings/red/recipe.yaml`
2. Loads Red's facet assembly (cognitive architecture)
3. Red begins continuous cognition - perceiving, thinking, responding

## Next Steps

- [Commands Reference](commands.md) - Full command list
- [Configuration](configuration.md) - LLM providers, ports, etc.
