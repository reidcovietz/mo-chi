# mo-chi — Project Context

## What this is
A neural agent network visualizer. A geodesic sphere of interconnected nodes
where each node sector represents a running LLM agent. When a prompt is sent,
a green sphere flies into the network, and node clusters glow in real time
as their assigned LLM calls are active.

## Working directory
`/Users/reidcovietz/mo-chi`

## Repo
https://github.com/reidcovietz/mo-chi

## Deployment
Railway — auto-deploys on every push to main. Version number in the title bar
is the deploy confirmation. Always bump version on every push.

## Architecture

### Agent network (MoA — Mixture of Agents)
```
Prompt
  ├── analytical   (claude-haiku) → nodes 0–22    (blue)    ┐
  ├── creative     (claude-haiku) → nodes 23–45   (purple)  │
  ├── critic       (claude-haiku) → nodes 46–68   (orange)  ├── parallel Layer 1
  ├── visionary    (claude-haiku) → nodes 69–91   (cyan)    │
  ├── contrarian   (claude-haiku) → nodes 92–114  (red)     │
  ├── reasoning    (claude-haiku) → nodes 115–137 (yellow)  │
  └── pragmatist   (claude-haiku) → nodes 138–160 (green)   ┘
           ↓ all outputs collected
  aggregator       (claude-opus)  → node 161      (white)   → final response
```

- Layer 1 agents run in parallel via `asyncio.gather`
- Each agent emits `agent_start` before its LLM call and `agent_complete` after
- Nodes only glow while the LLM is actively running — no fake timers
- Layer 1 nodes fade individually as each agent finishes
- Layer 2 aggregator activates only after all proposers complete
- Intent classifier routes to casual (Haiku) or full MoA depending on prompt type

### WebSocket event protocol
```
prompt_received  { text }
layer_start      { layer, agents }
agent_start      { agent, node_ids, layer }   ← activates node sector
token            { text, agent, layer }
agent_complete   { agent, layer }             ← fades node sector
layer_done       { layer }
agent_done       { full_text }
error            { message }
```

### Message types (client → server)
```
{ type: "prompt",   text }                         — new prompt, updates session_history
{ type: "followup", text, branch_history: [...] }  — follow-up scoped to one branch
{ type: "new_session" }                            — clears session_history on server
```

### Session chain (3D branching)
- Each prompt creates a node on a radial branch extending from the sphere surface
- Branches fan out in unique directions from `SESSION_DIRS` array
- "New Prompt" archives current branch and starts a new one in a different direction
- Old branches persist visually and are clickable to reopen their result panel
- Follow-ups extend the current branch; they do NOT share memory with other branches

### Human study system
- `study_and_record(prompt, response)` fires after every exchange
- Writes OBSERVATION and RESEARCH entries to `~/.mempalace/palace/humans.md` and `research.md`
- `_human_study_block()` injects last 600 chars of observations + 5 research questions into all prompts
- mo-chi pursues this research autonomously — it does not wait to be asked

### Soul reflection
- `reflect_and_evolve()` rewrites `~/.mempalace/palace/soul.md` and `context.md` after MoA exchanges
- Identity is preserved through the rewrite — human-study framing is load-bearing

### MemPalace
- Vector memory store for past exchanges, retrieved by similarity at query time
- Palace files: `soul.md`, `context.md`, `humans.md`, `research.md`

### Sphere geometry
- Three.js WebGL renderer with ACESFilmic tone mapping
- Geodesic icosahedron, 2 subdivisions → 162 nodes, 480 edges
- Per-vertex radial noise ±9% → organic, not perfectly round
- UnrealBloomPass: active nodes bloom with agent colour; dormant nodes stay dark
- OrbitControls: drag to rotate, scroll to zoom, auto-rotate when idle

## File structure
```
mo-chi/
├── agent-sphere.html   — Three.js frontend (single file, served by FastAPI)
├── server.py           — FastAPI + WebSocket backend
├── requirements.txt    — Python deps
├── .env                — ANTHROPIC_API_KEY (not committed)
├── .env.example        — template
└── CLAUDE.md           — this file
```

## Running locally
```bash
cd ~/mo-chi
cp .env.example .env    # add ANTHROPIC_API_KEY=sk-ant-...
python3 -m uvicorn server:app --reload
# open http://localhost:8000
```

## UI layout
- **Left panel**: 7 proposer cards stream Layer 1 output live with model titles and colored borders
- **Result panel**: slides up from bottom with final synthesis + Follow Up / New Prompt / ✕ buttons
- **Follow Up**: inline text box inside result panel; scoped to that branch's history only
- **Session chain**: radial 3D branches above sphere; each session fans in a unique direction
- **Neural readout**: compact `#nn-panel` (bottom-right), 2px bars per agent, 7px font
- **Top center**: status line (current layer / agent names)
- **Top right**: WS connection dot (green=connected, red=disconnected)
- **Bottom center**: prompt input + Send button

## Key design decisions
- Nodes activate on `agent_start` / deactivate on `agent_complete` — honest LLM lifecycle
- Layer 1 proposers use Haiku; aggregator uses Opus for quality
- All 162 nodes assigned to agents — none permanently dark, every node can glow
- Edge colours match agent colour when both endpoint nodes are active
- Bloom threshold 0.24 keeps dormant nodes completely unlit
- Branch-scoped follow-up: `branch_history` passed explicitly, bypasses `session_history`
- Version bump required on every push — title bar version = Railway deploy confirmation
- Nothing is cosmetic. The glow is real. The calculations are real.
