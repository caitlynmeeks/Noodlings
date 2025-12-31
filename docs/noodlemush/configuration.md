# NoodleMUSH Configuration

Server settings and LLM provider setup.

---

## Config File

`applications/cmush/config.yaml`

```yaml
server:
  http_port: 8080
  ws_port: 8765

llm:
  default_provider: ollama

ollama:
  host: http://localhost:11434
  model: llama3.2

# Additional providers...
```

## LLM Providers

NoodleMUSH supports 8 LLM providers:

| Provider | Local/Cloud | Notes |
|----------|-------------|-------|
| Ollama | Local | Recommended for development |
| LMStudio | Local | Good for larger models |
| OpenAI | Cloud | GPT-4, GPT-3.5 |
| Anthropic | Cloud | Claude models |
| Google | Cloud | Gemini |
| Groq | Cloud | Fast inference |
| OpenRouter | Cloud | Multi-model gateway |
| Together | Cloud | Open models |

## Model Labels

Assign semantic labels to models for different cognitive tasks:

```yaml
model_labels:
  thinking: ollama/llama3.2        # General cognition
  speaking: ollama/llama3.2        # Dialogue generation
  perception: ollama/llama3.2      # Scene understanding
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI access |
| `ANTHROPIC_API_KEY` | Anthropic access |
| `GOOGLE_API_KEY` | Google AI access |

## Startup Options

```bash
./start.sh                    # Normal start
./start.sh --debug           # Verbose logging
./start.sh --port 9000       # Custom port
```
