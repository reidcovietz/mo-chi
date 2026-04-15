# mo-chi — Local / Open Source / TUI Plan

## The goal
Make mo-chi open source so anyone can plug in their own models.
Alongside the browser sphere UI, offer a CLI and TUI for local/terminal users.

---

## Multi-provider model slots

Each agent is already a config dict. Expose these as user-configurable slots
in `mo-chi.config.yaml`. Add `ollama` as a valid provider alongside existing
cloud providers — it speaks the same OpenAI-compatible API, zero code changes needed.

```python
CLIENTS = {
    "groq":       AsyncOpenAI(base_url="https://api.groq.com/openai/v1", ...),
    "anthropic":  AsyncOpenAI(base_url="https://api.anthropic.com/v1",   ...),
    "openrouter": AsyncOpenAI(base_url="https://openrouter.ai/api/v1",   ...),
    "ollama":     AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
}
```

### Example user config

```yaml
# mo-chi.config.yaml
agents:
  analytical:
    provider: groq
    model: llama-3.1-8b
    bias_weight: 1.2        # how heavily aggregator weights this agent's output
  creative:
    provider: ollama
    model: mistral
    bias_weight: 0.8
  critic:
    provider: anthropic
    model: claude-haiku-4-5-20251001
    bias_weight: 1.0
  aggregator:
    provider: ollama
    model: llama3.1:70b     # bigger model recommended for synthesis quality
```

### Bias weights
Users tune how much each agent influences the final synthesis.
Heavy analytical weight → more logical outputs.
Heavy creative weight → more lateral thinking.
The aggregator multiplies each agent's output by its weight before synthesizing.

---

## Cloud + local hybrid

Both modes work simultaneously. Users can mix per agent slot:

| Use case | Setup |
|---|---|
| Full local, offline, private | All agents → Ollama |
| Full cloud, no local setup | All agents → Groq / Anthropic / OpenRouter |
| Privacy split | Sensitive agents → Ollama, fast agents → cloud |
| Cost optimization | Proposers → cheap cloud, aggregator → local 70B |
| Offline fallback | Cloud primary, auto-failover to Ollama if rate-limited |

### Constraint to document
Ollama must be reachable from wherever the server runs.
If mo-chi is deployed on Railway → cloud models only.
If mo-chi runs locally → full hybrid available.
Surface this at startup: detect Ollama at `localhost:11434`, grey out local
option in config if unreachable.

---

## Repo structure (open source)

```
mo-chi/
├── mo-chi.config.yaml      ← user fills this in
├── .env.example            ← API keys for cloud providers
├── server.py               ← reads config, builds agent slots dynamically
├── agent-sphere.html       ← browser sphere UI (no changes needed)
├── cli.py                  ← CLI client
├── tui.py                  ← Textual TUI client
├── setup.py                ← detects Ollama, walks through slot assignment
└── docs/
    ├── ollama-setup.md
    ├── cloud-setup.md
    └── hybrid-setup.md
```

---

## CLI (cli.py)

Simple, unix-composable. Streams tokens to stdout.

```bash
mochi "why is the sky blue"                  # single prompt, streams to stdout
mochi --agent analytical "explain entropy"   # query one specific agent only
mochi --local                                # force all agents to Ollama
mochi --cloud                                # force all agents to cloud
mochi recall                                 # what have you learned lately (discoveries.md)
mochi discoveries                            # dump discoveries.md
mochi soul                                   # print current soul.md
mochi embarrassments                         # print recent embarrassments
cat prompt.txt | mochi                       # pipe input
```

Composes with normal shell tools. Tokens stream as they arrive.

---

## TUI (tui.py)

Built with Textual (Python). Connects to the same WebSocket server,
receives the same event stream, renders into terminal panels.

```
┌─ mo-chi v1.x.x ───────────────────────────────────────┐
│ analytical  ████████░░  The core mechanism here is...  │
│ creative    ██████████  Consider the angle of...       │
│ critic      ████░░░░░░  thinking...                    │
│ visionary   ██████░░░░  thinking...                    │
│ contrarian  ░░░░░░░░░░  waiting                        │
│ reasoning   ████████░░  Breaking this into components  │
│ pragmatist  ██████████  Practically speaking...        │
├────────────────────────────────────────────────────────┤
│ synthesis                                              │
│ The data from three agents converges on one point...   │
│                                                        │
├────────────────────────────────────────────────────────┤
│ > _                                                    │
└────────────────────────────────────────────────────────┘
```

Real-time streaming, progress bars per agent, synthesis panel updates live.
Watching 7 agents stream in parallel in split-pane terminal is the differentiator
for the open source release — most local LLM users live in the terminal.

---

## Shared architecture

All three clients speak the same WebSocket event protocol.
Nothing in server.py changes when you add a new client.

```
server.py
    ↑
    ├── agent-sphere.html   ← browser (existing)
    ├── cli.py              ← CLI (new)
    └── tui.py              ← Textual TUI (new)
```

Add more later (VS Code extension, Raycast plugin, etc.) — server never changes.

---

## What needs to change in server.py

1. Read `mo-chi.config.yaml` at startup instead of hardcoded `LAYER1_AGENTS`
2. Build `CLIENTS` dict dynamically from config (add ollama entry if detected)
3. Pass `bias_weight` from config into `run_aggregator` — weight each agent's
   output before the aggregator sees it instead of equal weighting
4. Startup check: ping `localhost:11434` to detect Ollama, log result

---

## Honest caveats to document

- Aggregator needs a capable model (30B+ recommended) for quality synthesis.
  Small 3B Ollama models doing self-reflection / embarrassment loops produce shallow results.
- Curiosity/research loops, soul reflection, and embarrassment evolution all work
  best with a reasoning-capable model. Worth calling out in docs per feature.
- The sphere UI still needs a browser — TUI is the terminal-native alternative,
  not a replacement for the 3D visualization.
