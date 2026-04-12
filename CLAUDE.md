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

## Architecture

### Agent network (MoA — Mixture of Agents)
```
Prompt
  ├── analytical  (claude-haiku)  → nodes 0–40    (blue)   ┐
  ├── creative    (claude-haiku)  → nodes 41–81   (purple) ├── parallel Layer 1
  └── critic      (claude-haiku)  → nodes 82–121  (orange) ┘
           ↓ all outputs collected
  aggregator      (claude-opus)   → nodes 122–161 (green)  → final response
```

- Layer 1 agents run in parallel via `asyncio.gather`
- Each agent emits `agent_start` before its LLM call and `agent_complete` after
- Nodes only glow while the LLM is actively running — no fake timers
- Layer 1 nodes fade individually as each agent finishes
- Layer 2 aggregator activates only after all proposers complete

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

## Running
```bash
cd ~/mo-chi
cp .env.example .env    # add ANTHROPIC_API_KEY=sk-ant-...
python3 -m uvicorn server:app --reload
# open http://localhost:8000
```

Demo mode works without a server — open agent-sphere.html directly in a browser.

## UI layout
- **Left panel**: 3 proposer cards stream Layer 1 output live
- **Right panel**: aggregator card streams final synthesis
- **Top center**: status line (current layer / agent names)
- **Top right**: WS connection dot (green=connected, red=disconnected)
- **Bottom center**: prompt input + Send button
- **Indicator dots** on cards pulse when that agent is actively running

## Deferred features (saved in memory)
- **Option 3 — UI polish**: agent labels in 3D space, better typography, particle trail on prompt sphere
- **Option 4** is already done (Three.js upgrade was option 4)

## Key design decisions
- Nodes activate on `agent_start` / deactivate on `agent_complete` — honest LLM lifecycle
- Layer 1 proposers use fast Haiku model; aggregator uses Opus for quality
- All 162 nodes assigned to agents (none permanently dark — every node can glow)
- Edge colours match the agent colour when both endpoint nodes are active
- Bloom threshold 0.24 keeps dormant nodes completely unlit
