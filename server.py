"""
mo-chi · FastAPI WebSocket backend
Multi-agent MoA architecture:
  Layer 1 — 3 parallel proposers (analytical, creative, critic)
  Layer 2 — 1 aggregator synthesizes all outputs

Node activation model: emit agent_start/agent_complete so the frontend
only lights nodes while that specific LLM call is actually running.

Events:
  prompt_received  { text }
  layer_start      { layer, agents }
  agent_start      { agent, node_ids, layer }
  token            { text, agent, layer }
  agent_complete   { agent, layer }
  layer_done       { layer }
  agent_done       { full_text }
  error            { message }
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import anthropic

load_dotenv()

app = FastAPI(title="mo-chi")
aclient = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Agent definitions ─────────────────────────────────────────────────────────
# 162 total nodes on geodesic detail=2 sphere, split into 4 sectors.
LAYER1_AGENTS = [
    {
        "name": "analytical",
        "nodes": list(range(0, 41)),          # 41 nodes
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 256,
        "system": (
            "You are an analytical reasoning agent. Given a prompt, provide a "
            "precise, logical analysis focusing on structure and evidence. "
            "Be concise — 2 to 3 sentences."
        ),
    },
    {
        "name": "creative",
        "nodes": list(range(41, 82)),          # 41 nodes
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 256,
        "system": (
            "You are a creative thinking agent. Given a prompt, offer an "
            "imaginative, lateral, or unexpected perspective. "
            "Be concise — 2 to 3 sentences."
        ),
    },
    {
        "name": "critic",
        "nodes": list(range(82, 122)),         # 40 nodes
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 256,
        "system": (
            "You are a critical evaluation agent. Given a prompt, identify "
            "key challenges, risks, or counterpoints worth considering. "
            "Be concise — 2 to 3 sentences."
        ),
    },
]

LAYER2_AGENT = {
    "name": "aggregator",
    "nodes": list(range(122, 162)),            # 40 nodes
    "model": "claude-opus-4-6",
    "max_tokens": 1024,
    "system": (
        "You are a synthesis agent. You receive perspectives from three specialist "
        "agents — analytical, creative, and critic — and weave them into a single "
        "well-rounded, coherent response. Write naturally, without labeling sections."
    ),
}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    with open("agent-sphere.html", "r") as f:
        return HTMLResponse(f.read())


async def emit(ws: WebSocket, event: str, **kwargs):
    await ws.send_text(json.dumps({"event": event, **kwargs}))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "prompt":
                prompt = msg.get("text", "").strip()
                if prompt:
                    await run_moa(ws, prompt)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await emit(ws, "error", message=str(e))
        except Exception:
            pass


# ── Agent runners ─────────────────────────────────────────────────────────────
async def run_proposer(ws: WebSocket, agent: dict, prompt: str) -> str:
    # Nodes light up exactly when this LLM call is live
    await emit(ws, "agent_start",
               agent=agent["name"],
               node_ids=agent["nodes"],
               layer=1)

    full_text = []
    async with aclient.messages.stream(
        model=agent["model"],
        max_tokens=agent["max_tokens"],
        system=agent["system"],
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            full_text.append(text)
            await emit(ws, "token", text=text, agent=agent["name"], layer=1)
            await asyncio.sleep(0)

    await emit(ws, "agent_complete", agent=agent["name"], layer=1)
    return "".join(full_text)


async def run_aggregator(ws: WebSocket, layer1_outputs: dict) -> str:
    combined = "\n\n".join(
        f"[{name}]: {text}" for name, text in layer1_outputs.items()
    )
    agg_prompt = (
        f"Here are three specialist perspectives:\n\n{combined}\n\n"
        f"Synthesize these into a single clear, complete response."
    )

    await emit(ws, "agent_start",
               agent="aggregator",
               node_ids=LAYER2_AGENT["nodes"],
               layer=2)

    full_text = []
    async with aclient.messages.stream(
        model=LAYER2_AGENT["model"],
        max_tokens=LAYER2_AGENT["max_tokens"],
        system=LAYER2_AGENT["system"],
        messages=[{"role": "user", "content": agg_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            full_text.append(text)
            await emit(ws, "token", text=text, agent="aggregator", layer=2)
            await asyncio.sleep(0)

    await emit(ws, "agent_complete", agent="aggregator", layer=2)
    return "".join(full_text)


# ── MoA orchestrator ──────────────────────────────────────────────────────────
async def run_moa(ws: WebSocket, prompt: str):
    await emit(ws, "prompt_received", text=prompt)

    # Layer 1: 3 proposers in parallel
    await emit(ws, "layer_start", layer=1,
               agents=[a["name"] for a in LAYER1_AGENTS])

    results = await asyncio.gather(
        *[run_proposer(ws, agent, prompt) for agent in LAYER1_AGENTS],
        return_exceptions=True,
    )

    for agent, result in zip(LAYER1_AGENTS, results):
        if isinstance(result, Exception):
            await emit(ws, "error", message=f"{agent['name']}: {result}")
            return

    await emit(ws, "layer_done", layer=1)

    layer1_outputs = {
        agent["name"]: result
        for agent, result in zip(LAYER1_AGENTS, results)
    }

    # Layer 2: aggregator
    await emit(ws, "layer_start", layer=2, agents=["aggregator"])

    try:
        final_text = await run_aggregator(ws, layer1_outputs)
    except Exception as e:
        await emit(ws, "error", message=str(e))
        return

    await emit(ws, "layer_done", layer=2)
    await emit(ws, "agent_done", full_text=final_text)
