"""
mo-chi · FastAPI WebSocket backend — 100% free LLMs
Multi-agent MoA architecture:
  Layer 1 — 7 parallel proposers across Groq + Gemini + OpenRouter
  Layer 2 — 1 aggregator (Groq llama-3.3-70b)

Free API keys:
  Groq       → https://console.groq.com        (GROQ_API_KEY)
  Gemini     → https://aistudio.google.com     (GEMINI_API_KEY)
  OpenRouter → https://openrouter.ai           (OPENROUTER_API_KEY)

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
from openai import AsyncOpenAI

load_dotenv()

app = FastAPI(title="mo-chi")

# ── Provider clients (all OpenAI-compatible) ───────────────────────────────────
CLIENTS = {
    "groq": AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY", ""),
    ),
    "gemini": AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ.get("GEMINI_API_KEY", ""),
    ),
    "openrouter": AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    ),
}

# ── Agent definitions ──────────────────────────────────────────────────────────
# 162 nodes split evenly: 7 proposers × 20 nodes + aggregator 22 nodes
LAYER1_AGENTS = [
    {
        "name":      "analytical",
        "nodes":     list(range(0, 33)),    # 70B → 33 nodes
        "provider":  "groq",
        "model":     "llama-3.3-70b-versatile",
        "max_tokens": 200,
        "system": (
            "You are an analytical reasoning agent. Given a prompt, provide a "
            "precise, logical analysis focusing on structure and evidence. "
            "Be concise — 2 to 3 sentences."
        ),
    },
    {
        "name":      "creative",
        "nodes":     list(range(33, 37)),   # 8B  → 4 nodes
        "provider":  "groq",
        "model":     "llama-3.1-8b-instant",
        "max_tokens": 200,
        "system": (
            "You are a creative thinking agent. Given a prompt, offer an "
            "imaginative, lateral, or unexpected perspective. "
            "Be concise — 2 to 3 sentences."
        ),
    },
    {
        "name":      "critic",
        "nodes":     list(range(37, 70)),   # 70B → 33 nodes
        "provider":  "groq",
        "model":     "llama-3.3-70b-versatile",
        "max_tokens": 200,
        "system": (
            "You are a critical evaluation agent. Given a prompt, identify "
            "key challenges, risks, or counterpoints worth considering. "
            "Be concise — 2 to 3 sentences."
        ),
    },
    {
        "name":      "visionary",
        "nodes":     list(range(70, 79)),   # ~20B → 9 nodes
        "provider":  "gemini",
        "model":     "gemini-2.0-flash",
        "max_tokens": 200,
        "system": (
            "You are a visionary agent. Given a prompt, describe a bold "
            "long-term possibility or future scenario it could lead to. "
            "Be concise — 2 to 3 sentences."
        ),
    },
    {
        "name":      "contrarian",
        "nodes":     list(range(79, 112)),  # 70B → 33 nodes
        "provider":  "openrouter",
        "model":     "deepseek/deepseek-r1-distill-llama-70b:free",
        "max_tokens": 200,
        "system": (
            "You are a contrarian agent. Given a prompt, argue the opposite "
            "of the obvious conclusion. Challenge assumptions directly. "
            "Be concise — 2 to 3 sentences. Do not include reasoning tags."
        ),
    },
    {
        "name":      "reasoning",
        "nodes":     list(range(112, 125)), # 27B → 13 nodes
        "provider":  "openrouter",
        "model":     "google/gemma-3-27b-it:free",
        "max_tokens": 200,
        "system": (
            "You are a structured reasoning agent. Given a prompt, break it "
            "into its core components and explain the logical relationships. "
            "Be concise — 2 to 3 sentences."
        ),
    },
    {
        "name":      "pragmatist",
        "nodes":     list(range(125, 128)), # 7B  → 3 nodes
        "provider":  "openrouter",
        "model":     "mistralai/mistral-7b-instruct:free",
        "max_tokens": 200,
        "system": (
            "You are a pragmatist agent. Given a prompt, focus on what is "
            "immediately actionable, realistic, and practical right now. "
            "Be concise — 2 to 3 sentences."
        ),
    },
]

LAYER2_AGENT = {
    "name":      "aggregator",
    "nodes":     list(range(128, 162)),   # 70B → 34 nodes
    "provider":  "groq",
    "model":     "llama-3.3-70b-versatile",
    "max_tokens": 1024,
    "system": (
        "You are a synthesis agent. You receive seven specialist perspectives "
        "and weave them into a single well-rounded, coherent response. "
        "Write naturally — no section labels, no bullet points."
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
# Fallback used when an agent's primary model fails all retries.
FALLBACK = {"provider": "groq", "model": "llama-3.1-8b-instant"}

async def _call_model(ws: WebSocket, agent: dict, prompt: str,
                      provider: str, model: str) -> str:
    """Single attempt: stream one model and return full text."""
    client = CLIENTS[provider]
    full_text = []
    stream = await client.chat.completions.create(
        model=model,
        max_tokens=agent["max_tokens"],
        messages=[
            {"role": "system", "content": agent["system"]},
            {"role": "user",   "content": prompt},
        ],
        stream=True,
    )
    async for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        if text:
            full_text.append(text)
            await emit(ws, "token", text=text, agent=agent["name"], layer=1)
            await asyncio.sleep(0)
    return "".join(full_text)


async def run_proposer(ws: WebSocket, agent: dict, prompt: str) -> str:
    """Run a proposer with up to 3 attempts, then fall back to a reliable model.
    Every agent always completes — no prompt ever skips a layer node."""
    await emit(ws, "agent_start",
               agent=agent["name"],
               node_ids=agent["nodes"],
               layer=1)

    last_err = None
    # Try primary model up to 3 times
    for attempt in range(3):
        try:
            result = await _call_model(
                ws, agent, prompt, agent["provider"], agent["model"])
            await emit(ws, "agent_complete", agent=agent["name"], layer=1)
            return result
        except Exception as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(1.5)

    # Primary exhausted — switch to fallback model (always available on Groq)
    try:
        result = await _call_model(
            ws, agent, prompt, FALLBACK["provider"], FALLBACK["model"])
        await emit(ws, "agent_complete", agent=agent["name"], layer=1)
        return result
    except Exception as e:
        await emit(ws, "agent_complete", agent=agent["name"], layer=1)
        raise Exception(f"{agent['name']} failed (primary + fallback): {e}") from e


async def run_aggregator(ws: WebSocket, layer1_outputs: dict) -> str:
    combined = "\n\n".join(
        f"[{name}]: {text}" for name, text in layer1_outputs.items()
    )
    agg_prompt = (
        f"Here are seven specialist perspectives on the same prompt:\n\n"
        f"{combined}\n\n"
        f"Synthesize these into a single clear, complete response."
    )

    await emit(ws, "agent_start",
               agent="aggregator",
               node_ids=LAYER2_AGENT["nodes"],
               layer=2)

    client = CLIENTS[LAYER2_AGENT["provider"]]
    full_text = []

    stream = await client.chat.completions.create(
        model=LAYER2_AGENT["model"],
        max_tokens=LAYER2_AGENT["max_tokens"],
        messages=[
            {"role": "system", "content": LAYER2_AGENT["system"]},
            {"role": "user",   "content": agg_prompt},
        ],
        stream=True,
    )
    async for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        if text:
            full_text.append(text)
            await emit(ws, "token", text=text, agent="aggregator", layer=2)
            await asyncio.sleep(0)

    await emit(ws, "agent_complete", agent="aggregator", layer=2)
    return "".join(full_text)


# ── MoA orchestrator ──────────────────────────────────────────────────────────
async def run_moa(ws: WebSocket, prompt: str):
    await emit(ws, "prompt_received", text=prompt)

    await emit(ws, "layer_start", layer=1,
               agents=[a["name"] for a in LAYER1_AGENTS])

    results = await asyncio.gather(
        *[run_proposer(ws, agent, prompt) for agent in LAYER1_AGENTS],
        return_exceptions=True,
    )

    layer1_outputs = {}
    for agent, result in zip(LAYER1_AGENTS, results):
        if isinstance(result, Exception):
            # Should rarely happen — primary + fallback both failed
            await emit(ws, "error", message=str(result))
            layer1_outputs[agent["name"]] = "[unavailable]"
        else:
            layer1_outputs[agent["name"]] = result

    await emit(ws, "layer_done", layer=1)

    await emit(ws, "layer_start", layer=2, agents=["aggregator"])

    try:
        final_text = await run_aggregator(ws, layer1_outputs)
    except Exception as e:
        await emit(ws, "error", message=str(e))
        return

    await emit(ws, "layer_done", layer=2)
    await emit(ws, "agent_done", full_text=final_text)
