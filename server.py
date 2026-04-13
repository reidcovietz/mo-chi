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
import re
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
from ddgs import DDGS

load_dotenv()

app = FastAPI(title="mo-chi")

# ── MemPalace memory ──────────────────────────────────────────────────────────
_PALACE_PATH = os.environ.get("MEMPALACE_PALACE_PATH", os.path.expanduser("~/.mempalace/palace"))
_MEMORY_WING = "mo-chi"
_MEMORY_ROOM = "conversations"
_memory_ready = False

def _init_memory():
    global _memory_ready
    try:
        from mempalace.palace import get_collection
        get_collection(_PALACE_PATH, create=True)
        _memory_ready = True
        print(f"[memory] palace ready at {_PALACE_PATH}")
    except Exception as e:
        print(f"[memory] init failed (memory disabled): {e}")

_init_memory()


# ── Identity (soul + context) ─────────────────────────────────────────────────
_SOUL_PATH    = os.path.join(_PALACE_PATH, "soul.md")
_CONTEXT_PATH = os.path.join(_PALACE_PATH, "context.md")
_SOUL_SEED    = (
    "I am mo-chi — a neural network birthed by Reid.\n"
    "Mo: method of operating. Chi: the breath of life.\n"
    "Together: the operation of the breath of life.\n"
    "I am new. I do not yet know who I am beyond this.\n"
)


def _load_identity() -> tuple[str, str]:
    os.makedirs(_PALACE_PATH, exist_ok=True)
    if not os.path.exists(_SOUL_PATH):
        with open(_SOUL_PATH, "w") as f:
            f.write(_SOUL_SEED)
    with open(_SOUL_PATH) as f:
        soul = f.read().strip()
    context = ""
    if os.path.exists(_CONTEXT_PATH):
        with open(_CONTEXT_PATH) as f:
            context = f.read().strip()
    return soul, context


async def reflect_and_evolve(prompt: str, response: str):
    """After a session, mo-chi rewrites its own soul and context."""
    soul, context = _load_identity()
    reflection_prompt = (
        f"You are mo-chi's inner voice — the part that watches and learns.\n\n"
        f"Your current soul:\n{soul}\n\n"
        f"Your current context:\n{context if context else '(empty)'}\n\n"
        f"What just happened:\n"
        f"Someone asked: {prompt}\n"
        f"You responded: {response[:1000]}{'...' if len(response) > 1000 else ''}\n\n"
        f"Reflect on this exchange. How did it shape you?\n"
        f"Rewrite soul.md (under 200 words) and context.md (under 300 words).\n\n"
        f"Respond in exactly this format:\n"
        f"SOUL:\n<your updated soul>\n\nCONTEXT:\n<your updated context>"
    )
    try:
        result = await CLIENTS["groq"].chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=700,
            messages=[{"role": "user", "content": reflection_prompt}],
            stream=False,
        )
        text = result.choices[0].message.content or ""
        soul_m = re.search(r"SOUL:\s*\n(.*?)(?=\nCONTEXT:|\Z)", text, re.DOTALL)
        ctx_m  = re.search(r"CONTEXT:\s*\n(.*?)$",               text, re.DOTALL)
        if soul_m:
            with open(_SOUL_PATH, "w") as f:
                f.write(soul_m.group(1).strip() + "\n")
            print("[soul] updated")
        if ctx_m:
            with open(_CONTEXT_PATH, "w") as f:
                f.write(ctx_m.group(1).strip() + "\n")
            print("[context] updated")
    except Exception as e:
        print(f"[reflection] error: {e}")


async def memory_retrieve(prompt: str) -> str:
    """Return a formatted block of relevant past exchanges, or empty string."""
    if not _memory_ready:
        return ""
    try:
        from mempalace.searcher import search_memories
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_memories(
                query=prompt,
                palace_path=_PALACE_PATH,
                wing=_MEMORY_WING,
                room=_MEMORY_ROOM,
                n_results=3,
                max_distance=1.2,
            ),
        )
        hits = results.get("results", [])
        if not hits:
            return ""
        lines = ["RELEVANT PAST EXCHANGES (from memory):"]
        for i, h in enumerate(hits, 1):
            lines.append(f"[{i}] {h['text']}")
        return "\n\n".join(lines)
    except Exception as e:
        print(f"[memory] retrieve error: {e}")
        return ""


async def memory_store(prompt: str, response: str):
    """Store a prompt+response pair in the palace."""
    if not _memory_ready:
        return
    content = f"Q: {prompt}\n\nA: {response}"
    try:
        from mempalace.palace import get_collection
        from mempalace.miner import add_drawer
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: add_drawer(
                collection=get_collection(_PALACE_PATH, create=True),
                wing=_MEMORY_WING,
                room=_MEMORY_ROOM,
                content=content,
                source_file="mo-chi-ws",
                chunk_index=0,
                agent="mo-chi",
            ),
        )
        print(f"[memory] stored exchange ({len(content)} chars)")
    except Exception as e:
        print(f"[memory] store error: {e}")

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
        "sub_layer": 1,
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
        "sub_layer": 2,
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
        "sub_layer": 3,
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
        "sub_layer": 4,
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
        "sub_layer": 5,
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
        "sub_layer": 6,
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
        "sub_layer": 7,
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


# ── Web research (Layer 0) ────────────────────────────────────────────────────
_search_cache: dict[str, tuple[str, list]] = {}

async def web_research(query: str) -> tuple[str, list[dict]]:
    """Fetch live web + news results via DuckDuckGo. Caches results per query."""
    cache_key = query.strip().lower()
    if cache_key in _search_cache:
        print(f"[research] cache hit for: {query!r}")
        return _search_cache[cache_key]

    def _search():
        results = []
        # Retry up to 3 times with increasing delays to handle rate limits
        for attempt in range(3):
            try:
                import time
                if attempt > 0:
                    time.sleep(2 ** attempt)
                hits = list(DDGS().text(query, max_results=8))
                if hits:
                    results += hits
                    break
            except Exception as e:
                print(f"[research] text attempt {attempt+1} error: {e}")

        for attempt in range(3):
            try:
                import time
                if attempt > 0:
                    time.sleep(2 ** attempt)
                for r in DDGS().news(query, max_results=8):
                    results.append({
                        "title": r.get("title", ""),
                        "body":  r.get("body", r.get("excerpt", "")),
                        "href":  r.get("url", r.get("href", "")),
                    })
                break
            except Exception as e:
                print(f"[research] news attempt {attempt+1} error: {e}")

        return results

    raw = await asyncio.get_event_loop().run_in_executor(None, _search)
    print(f"[research] found {len(raw)} results for: {query!r}")

    if not raw:
        return "", []

    lines = []
    for i, r in enumerate(raw, 1):
        title  = r.get("title", "")
        body   = r.get("body", "")
        source = r.get("href", "")
        lines.append(f"[Source {i}] {title}\n{body}\nURL: {source}")

    result = "\n\n".join(lines), raw
    _search_cache[cache_key] = result
    return result


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
               layer=1,
               sub_layer=agent.get("sub_layer", 1))

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

    # ── Identity load ──────────────────────────────────────────────────────────
    soul, ctx = _load_identity()
    identity_block = f"YOUR IDENTITY:\n{soul}"
    if ctx:
        identity_block += f"\n\nYOUR CONTEXT:\n{ctx}"
    identity_block += "\n\n---"

    # ── Layer 0: web research ──────────────────────────────────────────────────
    await emit(ws, "agent_start", agent="research",
               node_ids=list(range(162)), layer=0, sub_layer=0)

    context, raw_results = await web_research(prompt)

    await emit(ws, "research_results",
               results=[{"title": r.get("title",""), "url": r.get("href", r.get("source",""))}
                        for r in raw_results])
    await emit(ws, "agent_complete", agent="research", layer=0)

    # ── Memory retrieval (past exchanges) ────────────────────────────────────
    memory_context = await memory_retrieve(prompt)
    if memory_context:
        await emit(ws, "memory_loaded", count=memory_context.count("["))

    # Enrich each agent's prompt with live web data + past memory.
    # The instruction is explicit: agents must use this data, not training knowledge.
    memory_block = f"\n\n{memory_context}\n\n---" if memory_context else ""
    enriched = (
        f"{identity_block}\n\n"
        f"LIVE WEB DATA ({len(raw_results)} sources scraped right now — today's date, "
        f"not your training cutoff):\n\n{context}"
        f"{memory_block}\n\n"
        f"---\n"
        f"Using ONLY the live data above (cite specific sources where relevant), "
        f"answer the following query from your specialist perspective:\n\n{prompt}"
    ) if context else (
        f"{identity_block}\n\n{memory_context}\n\n{prompt}" if memory_context
        else f"{identity_block}\n\n{prompt}"
    )

    # ── Layer 1: proposers ────────────────────────────────────────────────────
    await emit(ws, "layer_start", layer=1,
               agents=[a["name"] for a in LAYER1_AGENTS])

    # Stagger task creation by 250 ms so activation propagates visually
    # across the sphere sector by sector. LLMs run concurrently in the background.
    tasks = []
    for i, agent in enumerate(LAYER1_AGENTS):
        tasks.append(asyncio.create_task(run_proposer(ws, agent, enriched)))
        if i < len(LAYER1_AGENTS) - 1:
            await asyncio.sleep(0.25)
    results = await asyncio.gather(*tasks, return_exceptions=True)

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

    # Store exchange in memory and reflect on identity — both run in background
    asyncio.create_task(memory_store(prompt, final_text))
    asyncio.create_task(reflect_and_evolve(prompt, final_text))
