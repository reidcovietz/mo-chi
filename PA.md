# PA — Personal Assistant Layer (planned feature)

A Claude agent with file tool use that can read and modify mo-chi's own codebase
and palace files directly, then push to Railway. The brain rewires itself.

## Concept

A dedicated Claude model running with file tools. Separate from the main MoA flow —
its own input in the UI, its own endpoint on the server. When you talk to the PA,
it reads the relevant files, makes the change, commits, and pushes. Railway deploys
in ~30s. Version bump in the title bar confirms it landed.

No diff/Apply step. No passkey UI. The PA just does it.

## Tools the PA gets

```python
tools = [
    {
        "name": "read_file",
        "description": "Read a mo-chi repo file (agent-sphere.html, server.py, etc.)",
        "input_schema": {
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write a mo-chi file, then git commit and push to Railway",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_files",
        "description": "List files in the mo-chi repo",
        "input_schema": { "type": "object", "properties": {} }
    },
    {
        "name": "read_palace",
        "description": "Read a MemPalace file (soul.md, context.md, humans.md, research.md)",
        "input_schema": {
            "type": "object",
            "properties": { "file": { "type": "string" } },
            "required": ["file"]
        }
    },
    {
        "name": "write_palace",
        "description": "Update a MemPalace palace file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file":    { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["file", "content"]
        }
    }
]
```

## Example interactions

- *"make the archived nodes pulse slowly instead of static dim"*
  → reads agent-sphere.html, writes the JS change, pushes

- *"update your soul file to reflect what you learned today"*
  → reads soul.md, rewrites it, saves

- *"you're responding too verbosely, tighten the aggregator prompt"*
  → reads server.py, edits the aggregator system prompt, pushes

- *"add a seventh session branch direction"*
  → reads agent-sphere.html, appends to SESSION_DIRS, pushes

## Server implementation (~60 lines)

New FastAPI endpoint: `POST /pa` or a new WS type `pa_prompt`.

```python
import anthropic, subprocess, pathlib

REPO   = pathlib.Path("/app")          # or ~/mo-chi locally
PALACE = pathlib.Path.home() / ".mempalace/palace"
PA_MODEL = "claude-sonnet-4-6"         # Sonnet for most; swap Opus for big rewrites

def _tool_handler(tool_name, tool_input):
    if tool_name == "list_files":
        return "\n".join(str(p.relative_to(REPO)) for p in REPO.iterdir() if p.is_file())

    if tool_name == "read_file":
        p = REPO / tool_input["path"]
        return p.read_text() if p.exists() else "File not found"

    if tool_name == "write_file":
        p = REPO / tool_input["path"]
        p.write_text(tool_input["content"])
        subprocess.run(["git", "-C", str(REPO), "add", str(p)], check=True)
        subprocess.run(["git", "-C", str(REPO), "commit", "-m", f"pa: update {p.name}"], check=True)
        subprocess.run(["git", "-C", str(REPO), "push"], check=True)
        return f"Written and pushed: {p.name}"

    if tool_name == "read_palace":
        p = PALACE / tool_input["file"]
        return p.read_text() if p.exists() else "File not found"

    if tool_name == "write_palace":
        p = PALACE / tool_input["file"]
        p.write_text(tool_input["content"])
        return f"Written: {tool_input['file']}"

async def run_pa(prompt: str, ws):
    client   = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]

    while True:
        response = client.messages.create(
            model=PA_MODEL,
            max_tokens=4096,
            system="You are the PA for mo-chi. You have direct access to mo-chi's codebase "
                   "and memory palace. Use tools to read files before editing them. "
                   "Bump the version number in agent-sphere.html on every code change. "
                   "Be precise — read first, edit only what's needed.",
            tools=tools,
            messages=messages,
        )

        # Stream text back to client
        for block in response.content:
            if hasattr(block, "text"):
                await ws.send_json({"event": "pa_token", "text": block.text})

        if response.stop_reason != "tool_use":
            break

        # Execute tools
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _tool_handler(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})
```

## Frontend implementation (~100 lines)

- Small PA button in the UI (distinct from main prompt bar — different color/icon)
- PA input slides up separately from the main input
- PA responses stream into a dedicated panel (or reuse result panel with a PA header)
- Visual indicator: PA is active (e.g., white node 161 — the aggregator — glows differently)
- No proposer cards during PA mode — PA is a single model thinking, not a swarm

## Model choice

- **claude-sonnet-4-6** for most edits (fast, precise, cheap enough)
- **claude-opus-4-6** for large rewrites or identity changes (soul, architecture)
- Keep usage low — PA is triggered explicitly, not on every prompt

## What it can touch

| Target | Tool | Notes |
|---|---|---|
| `agent-sphere.html` | `write_file` | Frontend — triggers Railway redeploy |
| `server.py` | `write_file` | Backend — triggers Railway redeploy |
| `~/.mempalace/palace/*.md` | `write_palace` | Identity/memory — no redeploy needed |
| `CLAUDE.md` | `write_file` | Project context — no redeploy needed |

## Security notes

- Protect the PA endpoint with `ADMIN_KEY` from `.env`
- PA only runs when explicitly triggered — never auto-fires
- Rate limit writes (e.g., max 5 file writes per PA session)
- Log all write operations to a local `pa.log` file for audit
