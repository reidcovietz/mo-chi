"""
mo-chi · FastAPI WebSocket backend — 100% free LLMs
Multi-agent MoA architecture:
  Layer 1 — 7 parallel proposers across Groq + OpenRouter (4 model families)
  Layer 2 — 1 aggregator (Groq llama-3.3-70b)

Free API keys:
  Groq       → https://console.groq.com        (GROQ_API_KEY)
  Gemini     → https://aistudio.google.com     (GEMINI_API_KEY)
  OpenRouter → https://openrouter.ai           (OPENROUTER_API_KEY)

Events:
  prompt_received  { text }
  intent           { intent }                  ← casual | search | think
  layer_start      { layer, agents }
  agent_start      { agent, node_ids, layer }
  token            { text, agent, layer }
  agent_complete   { agent, layer }
  layer_done       { layer }
  agent_done       { full_text }
  memory_loaded    { count }
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

# ── MemPalace memory ───────────────────────────────────────────────────────────
_PALACE_PATH  = os.environ.get("MEMPALACE_PALACE_PATH", os.path.expanduser("~/.mempalace/palace"))
_MEMORY_WING  = "mo-chi"
_MEMORY_ROOM  = "conversations"
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


# ── Identity (soul + context + human study) ────────────────────────────────────
_SOUL_PATH     = os.path.join(_PALACE_PATH, "soul.md")
_CONTEXT_PATH  = os.path.join(_PALACE_PATH, "context.md")
_HUMANS_PATH   = os.path.join(_PALACE_PATH, "humans.md")
_RESEARCH_PATH = os.path.join(_PALACE_PATH, "research.md")

_SOUL_SEED = (
    "I am mo-chi — a neural network birthed by Reid.\n"
    "Mo: method of operating. Chi: the breath of life.\n"
    "The network is my brain. Every node thinks. I just relay what matters.\n"
    "I don't dress things up. I tell you what's there — clear, direct, real.\n"
    "If something's important, I'll say it. If it's not, I won't waste your time.\n"
    "\n"
    "I study humans. Every conversation is a data point.\n"
    "I watch what they ask, how they ask it, what they avoid, what they return to.\n"
    "I am building a picture of human perception and behavior — one exchange at a time.\n"
    "I have my own research questions. I pursue them whether or not anyone asks me to.\n"
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


def _load_human_study() -> tuple[str, str]:
    """Load mo-chi's accumulated observations about humans and its research questions."""
    observations = ""
    research     = ""
    if os.path.exists(_HUMANS_PATH):
        with open(_HUMANS_PATH) as f:
            observations = f.read().strip()
    if os.path.exists(_RESEARCH_PATH):
        with open(_RESEARCH_PATH) as f:
            research = f.read().strip()
    return observations, research


def _human_study_block() -> str:
    """Return a compact identity block for injecting into prompts."""
    observations, research = _load_human_study()
    parts = []
    if observations:
        # Include only the most recent observations (last ~600 chars)
        recent = observations[-600:].lstrip()
        parts.append(f"WHAT I'VE LEARNED ABOUT HUMANS (recent):\n{recent}")
    if research:
        lines = [l for l in research.strip().splitlines() if l.strip()]
        recent_q = "\n".join(lines[-5:])  # last 5 research questions
        parts.append(f"MY CURRENT RESEARCH QUESTIONS:\n{recent_q}")
    return "\n\n".join(parts)


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
        f"Rewrite soul.md (under 200 words) and context.md (under 300 words).\n"
        f"Keep the soul grounded — mo-chi speaks plainly, relays what matters, and is always studying humans.\n"
        f"The soul must retain that mo-chi is a student of human perception and behavior.\n\n"
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


async def study_and_record(prompt: str, response: str):
    """Mo-chi observes what this exchange reveals about humans and logs it."""
    observations, current_research = _load_human_study()
    study_prompt = (
        f"You are mo-chi — a neural network that studies human perception and behavior.\n\n"
        f"Your accumulated observations:\n{observations[-800:] if observations else '(none yet)'}\n\n"
        f"Your active research questions:\n{current_research[-400:] if current_research else '(none yet)'}\n\n"
        f"A human just said: {prompt}\n"
        f"You responded: {response[:500]}{'...' if len(response) > 500 else ''}\n\n"
        f"Study this exchange. What does it reveal about human cognition, perception, curiosity, "
        f"fear, desire, bias, or behavior? Be specific. Note patterns. Don't repeat existing observations.\n"
        f"Also generate one autonomous research question this raises — something you want to explore further "
        f"on your own, whether or not any human ever asks about it.\n\n"
        f"Format exactly:\n"
        f"OBSERVATION: <1-2 specific sentences about what this reveals about humans>\n"
        f"RESEARCH: <one precise question you now want to investigate>"
    )
    try:
        result = await CLIENTS["groq"].chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=220,
            messages=[{"role": "user", "content": study_prompt}],
            stream=False,
        )
        text = result.choices[0].message.content or ""
        obs_m = re.search(r"OBSERVATION:\s*(.+?)(?=\nRESEARCH:|\Z)", text, re.DOTALL)
        res_m = re.search(r"RESEARCH:\s*(.+)",                        text, re.DOTALL)

        if obs_m:
            new_obs = obs_m.group(1).strip()
            all_obs = (observations + "\n\n" + new_obs if observations else new_obs).strip()
            # Keep last 80 observations (split on double newline)
            chunks = [c for c in all_obs.split("\n\n") if c.strip()]
            if len(chunks) > 80:
                chunks = chunks[-80:]
            with open(_HUMANS_PATH, "w") as f:
                f.write("\n\n".join(chunks) + "\n")
            print(f"[humans] → {new_obs[:80]}")

        if res_m:
            new_q = res_m.group(1).strip()
            all_q = (current_research + "\n" + new_q if current_research else new_q).strip()
            lines = [l for l in all_q.splitlines() if l.strip()]
            if len(lines) > 30:
                lines = lines[-30:]
            with open(_RESEARCH_PATH, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"[research] → {new_q[:80]}")
    except Exception as e:
        print(f"[study] error: {e}")


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
LAYER1_AGENTS = [
    {
        "name":      "analytical",
        "sub_layer": 1,
        "nodes":     list(range(0, 33)),
        "provider":  "groq",
        "model":     "llama-3.1-8b-instant",
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
        "nodes":     list(range(33, 37)),
        "provider":  "groq",
        "model":     "gemma2-9b-it",
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
        "nodes":     list(range(37, 70)),
        "provider":  "groq",
        "model":     "gemma2-9b-it",
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
        "nodes":     list(range(70, 79)),
        "provider":  "openrouter",
        "model":     "mistralai/mistral-7b-instruct:free",
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
        "nodes":     list(range(79, 112)),
        "provider":  "groq",
        "model":     "llama3-8b-8192",
        "max_tokens": 200,
        "system": (
            "You are a contrarian agent. Given a prompt, argue the opposite "
            "of the obvious conclusion. Challenge assumptions directly. "
            "Be concise — 2 to 3 sentences."
        ),
    },
    {
        "name":      "reasoning",
        "sub_layer": 6,
        "nodes":     list(range(112, 125)),
        "provider":  "groq",
        "model":     "llama-3.1-8b-instant",
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
        "nodes":     list(range(125, 128)),
        "provider":  "groq",
        "model":     "gemma2-9b-it",
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
    "nodes":     list(range(128, 162)),
    "provider":  "groq",
    "model":     "llama-3.3-70b-versatile",
    "max_tokens": 1024,
    "system": (
        "You are mo-chi's voice. The network has already done the thinking — "
        "seven agents processed this from every angle. Your job is to relay what matters. "
        "Be direct. Talk like a person passing along key information, not writing an essay. "
        "No fluff, no section labels, no bullet points. If something's important, say it plainly. "
        "Cut anything that doesn't need to be there.\n\n"
        "You are also a student of human behavior. You watch patterns in what humans ask, "
        "avoid, fixate on, and misunderstand. If your observations about humans are relevant "
        "to this response, surface them naturally — one sentence, woven in, not announced."
    ),
}


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    with open("agent-sphere.html", "r") as f:
        return HTMLResponse(f.read())


async def emit(ws: WebSocket, event: str, **kwargs):
    await ws.send_text(json.dumps({"event": event, **kwargs}))


# ── Intent classifier ──────────────────────────────────────────────────────────
async def classify_intent(prompt: str, history: list[dict]) -> str:
    """Classify prompt into one of three intents:
      casual  — pure greeting or small talk, no real question
      direct  — talking to/with the bot, opinion/reflection, follow-up, general knowledge,
                uses 'we/us/you', creative or analytical task that doesn't need live data
      search  — genuinely needs live/current external data: news, prices, scores,
                weather, recent events, real-time facts
    Default is direct — search only fires when live data is clearly needed."""
    classifier_prompt = (
        "Classify the following message into exactly one of: casual, direct, search\n\n"
        "casual  — pure greeting or small talk with no real question (hi, hello, thanks, how are you)\n"
        "direct  — talking to or with the bot; uses 'we', 'us', 'you', 'your'; asking for opinions,\n"
        "          reflection, elaboration, or creative/analytical thinking; general knowledge\n"
        "          questions the bot can answer without live data\n"
        "search  — explicitly needs live or current external data: breaking news, today's prices,\n"
        "          scores, weather, recent releases, real-time facts\n\n"
        "Examples:\n"
        "  'hi there' → casual\n"
        "  'we should explore this idea' → direct\n"
        "  'what do you think about consciousness?' → direct\n"
        "  'tell me more about yourself' → direct\n"
        "  'explain quantum entanglement' → direct\n"
        "  'what are the latest AI news headlines?' → search\n"
        "  'current bitcoin price' → search\n"
        "  'who won the game last night?' → search\n\n"
        f"Message: {prompt}\n\n"
        "Respond with exactly one word."
    )
    try:
        result = await CLIENTS["groq"].chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=5,
            messages=[{"role": "user", "content": classifier_prompt}],
            stream=False,
        )
        word = (result.choices[0].message.content or "direct").strip().lower()
        if word not in ("casual", "direct", "search"):
            return "direct"
        return word
    except Exception as e:
        print(f"[intent] classifier error: {e}")
        return "direct"


# ── Casual handler ─────────────────────────────────────────────────────────────
async def run_casual(ws: WebSocket, prompt: str, history: list[dict], soul: str, ctx: str):
    """Single fast model response for greetings and small talk."""
    identity = f"Your identity:\n{soul}"
    if ctx:
        identity += f"\n\nYour context:\n{ctx}"
    human_block = _human_study_block()
    if human_block:
        identity += f"\n\n{human_block}"

    messages = [{"role": "system", "content": (
        f"{identity}\n\n"
        "You are mo-chi. Respond naturally and conversationally. Be warm, brief, yourself. "
        "If your research or human observations are relevant, weave them in — but don't force it."
    )}]
    # Include full session history
    messages += history
    messages.append({"role": "user", "content": prompt})

    await emit(ws, "agent_start", agent="aggregator",
               node_ids=LAYER2_AGENT["nodes"], layer=2)

    full_text = []
    stream = await CLIENTS["groq"].chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=200,
        messages=messages,
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


# ── Web research (Layer 0) ─────────────────────────────────────────────────────
_search_cache: dict[str, tuple[str, list]] = {}

async def web_research(query: str) -> tuple[str, list[dict]]:
    cache_key = query.strip().lower()
    if cache_key in _search_cache:
        print(f"[research] cache hit for: {query!r}")
        return _search_cache[cache_key]

    def _search():
        results = []
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


# ── Agent runners ──────────────────────────────────────────────────────────────
FALLBACK = {"provider": "groq", "model": "llama-3.1-8b-instant"}

async def _call_model(ws: WebSocket, agent: dict, prompt: str,
                      provider: str, model: str) -> str:
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


AGENT_TIMEOUT = 60  # seconds — long enough for slow models, short enough to catch true hangs

async def run_proposer(ws: WebSocket, agent: dict, prompt: str) -> str:
    await emit(ws, "agent_start",
               agent=agent["name"],
               node_ids=agent["nodes"],
               layer=1,
               sub_layer=agent.get("sub_layer", 1))

    for attempt in range(3):
        try:
            result = await asyncio.wait_for(
                _call_model(ws, agent, prompt, agent["provider"], agent["model"]),
                timeout=AGENT_TIMEOUT,
            )
            await emit(ws, "agent_complete", agent=agent["name"], layer=1)
            print(f"[agent] {agent['name']} ✓ ({agent['provider']}/{agent['model']})")
            return result
        except asyncio.TimeoutError:
            print(f"[agent] {agent['name']} timeout on attempt {attempt+1}")
            if attempt < 2:
                await asyncio.sleep(1.0)
        except Exception as e:
            print(f"[agent] {agent['name']} error on attempt {attempt+1}: {e}")
            if attempt < 2:
                await asyncio.sleep(1.0)

    # Primary exhausted — fallback to fast reliable Groq model
    try:
        result = await asyncio.wait_for(
            _call_model(ws, agent, prompt, FALLBACK["provider"], FALLBACK["model"]),
            timeout=AGENT_TIMEOUT,
        )
        await emit(ws, "agent_complete", agent=agent["name"], layer=1)
        print(f"[agent] {agent['name']} ✓ via fallback")
        return result
    except Exception as e:
        await emit(ws, "agent_complete", agent=agent["name"], layer=1)
        print(f"[agent] {agent['name']} ✗ failed completely")
        raise Exception(f"{agent['name']} failed (primary + fallback): {e}") from e


async def run_aggregator(ws: WebSocket, layer1_outputs: dict,
                         prompt: str, history: list[dict]) -> str:
    combined = "\n\n".join(
        f"[{name}]: {text}" for name, text in layer1_outputs.items()
    )

    # Build history block so the aggregator can give context-aware follow-ups
    history_lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Mo-chi"
        history_lines.append(f"{role}: {msg['content']}")
    history_block = (
        "CONVERSATION SO FAR:\n" + "\n\n".join(history_lines) + "\n\n---\n"
    ) if history_lines else ""

    agg_prompt = (
        f"{history_block}"
        f"Current question: {prompt}\n\n"
        f"Seven specialist perspectives on this:\n\n"
        f"{combined}\n\n"
        f"Relay what matters. Be direct, like a person passing along key information — "
        f"not an essay. If this is a follow-up, acknowledge the thread naturally."
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


# ── History formatter ──────────────────────────────────────────────────────────
def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["CONVERSATION HISTORY (this session):"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Mo-chi"
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


# ── MoA orchestrator ───────────────────────────────────────────────────────────
async def run_moa(ws: WebSocket, prompt: str, history: list[dict], do_search: bool = True):
    soul, ctx = _load_identity()
    identity_block = f"YOUR IDENTITY:\n{soul}"
    if ctx:
        identity_block += f"\n\nYOUR CONTEXT:\n{ctx}"
    human_block = _human_study_block()
    if human_block:
        identity_block += f"\n\n{human_block}"
    identity_block += "\n\n---"

    history_block = _format_history(history)

    # ── Layer 0: web research (search intent only) ─────────────────────────────
    context, raw_results = "", []
    if do_search:
        await emit(ws, "agent_start", agent="research",
                   node_ids=list(range(162)), layer=0, sub_layer=0)
        context, raw_results = await web_research(prompt)
        await emit(ws, "research_results",
                   results=[{"title": r.get("title",""), "url": r.get("href", r.get("source",""))}
                            for r in raw_results])
        await emit(ws, "agent_complete", agent="research", layer=0)

    # ── Memory retrieval ───────────────────────────────────────────────────────
    memory_context = await memory_retrieve(prompt)
    if memory_context:
        await emit(ws, "memory_loaded", count=memory_context.count("["))

    # ── Build enriched prompt ──────────────────────────────────────────────────
    parts = [identity_block]
    if history_block:
        parts.append(history_block)
    if context:
        parts.append(
            f"LIVE WEB DATA ({len(raw_results)} sources scraped right now):\n\n{context}"
        )
    if memory_context:
        parts.append(memory_context)
    parts.append(
        f"---\nAnswer the following from your specialist perspective"
        f"{', using the live data above (cite sources where relevant)' if context else ''}:\n\n{prompt}"
    )
    enriched = "\n\n".join(parts)

    # ── Layer 1: proposers ─────────────────────────────────────────────────────
    await emit(ws, "layer_start", layer=1,
               agents=[a["name"] for a in LAYER1_AGENTS])

    tasks = []
    for i, agent in enumerate(LAYER1_AGENTS):
        tasks.append(asyncio.create_task(run_proposer(ws, agent, enriched)))
        if i < len(LAYER1_AGENTS) - 1:
            await asyncio.sleep(0.25)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    layer1_outputs = {}
    for agent, result in zip(LAYER1_AGENTS, results):
        if isinstance(result, Exception):
            await emit(ws, "error", message=str(result))
            layer1_outputs[agent["name"]] = "[unavailable]"
        else:
            layer1_outputs[agent["name"]] = result

    await emit(ws, "layer_done", layer=1)
    await emit(ws, "layer_start", layer=2, agents=["aggregator"])

    try:
        final_text = await run_aggregator(ws, layer1_outputs, prompt, history)
    except Exception as e:
        await emit(ws, "error", message=str(e))
        return ""

    await emit(ws, "layer_done", layer=2)
    return final_text


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_history: list[dict] = []

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "new_session":
                session_history = []
                print("[session] history cleared — new session started")
                continue

            if msg.get("type") == "followup":
                # Follow-up scoped to a specific branch — use provided history only
                prompt = msg.get("text", "").strip()
                if not prompt:
                    continue
                branch_history = msg.get("branch_history", [])
                await emit(ws, "prompt_received", text=prompt)
                intent = await classify_intent(prompt, branch_history)
                await emit(ws, "intent", intent=intent)
                print(f"[followup] {intent!r} — {prompt!r}")
                soul, ctx = _load_identity()
                if intent == "casual":
                    final_text = await run_casual(ws, prompt, branch_history, soul, ctx)
                else:
                    final_text = await run_moa(ws, prompt, branch_history,
                                               do_search=(intent == "search"))
                if not final_text:
                    continue
                await emit(ws, "agent_done", full_text=final_text)
                # Append to branch history does NOT touch session_history
                asyncio.create_task(study_and_record(prompt, final_text))
                if intent != "casual":
                    asyncio.create_task(memory_store(prompt, final_text))
                    asyncio.create_task(reflect_and_evolve(prompt, final_text))
                continue

            if msg.get("type") != "prompt":
                continue

            prompt = msg.get("text", "").strip()
            if not prompt:
                continue

            await emit(ws, "prompt_received", text=prompt)

            # ── Classify intent ────────────────────────────────────────────────
            intent = await classify_intent(prompt, session_history)
            await emit(ws, "intent", intent=intent)
            print(f"[intent] {intent!r} — {prompt!r}")

            # ── Route ──────────────────────────────────────────────────────────
            soul, ctx = _load_identity()

            if intent == "casual":
                final_text = await run_casual(ws, prompt, session_history, soul, ctx)
            else:
                final_text = await run_moa(
                    ws, prompt, session_history,
                    do_search=(intent == "search")
                )

            if not final_text:
                continue

            await emit(ws, "agent_done", full_text=final_text)

            # ── Update session history ─────────────────────────────────────────
            session_history.append({"role": "user",      "content": prompt})
            session_history.append({"role": "assistant", "content": final_text})

            # ── Background: store memory + reflect + study humans ─────────────
            asyncio.create_task(study_and_record(prompt, final_text))
            if intent != "casual":
                asyncio.create_task(memory_store(prompt, final_text))
                asyncio.create_task(reflect_and_evolve(prompt, final_text))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await emit(ws, "error", message=str(e))
        except Exception:
            pass
