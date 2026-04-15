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

# ── Brain debug event bus ──────────────────────────────────────────────────────
# Background tasks push here; active WS connections drain and stream to frontend.
_brain_subscribers: list[asyncio.Queue] = []

def _log_brain(action: str, file: str, detail: str = ""):
    """Log a brain file access/write event to all connected clients."""
    entry = {"action": action, "file": file, "detail": detail}
    for q in _brain_subscribers:
        try:
            q.put_nowait(entry)
        except Exception:
            pass

async def _stream_brain_log(ws: WebSocket):
    """Per-connection drain loop — forwards brain events to the client."""
    q: asyncio.Queue = asyncio.Queue(maxsize=120)
    _brain_subscribers.append(q)
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=1.0)
                await ws.send_text(json.dumps({"event": "brain_debug", **event}))
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
    finally:
        if q in _brain_subscribers:
            _brain_subscribers.remove(q)

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
_SOUL_PATH           = os.path.join(_PALACE_PATH, "soul.md")
_CONTEXT_PATH        = os.path.join(_PALACE_PATH, "context.md")
_HUMANS_PATH         = os.path.join(_PALACE_PATH, "humans.md")
_RESEARCH_PATH       = os.path.join(_PALACE_PATH, "research.md")
_DISCOVERIES_PATH    = os.path.join(_PALACE_PATH, "discoveries.md")
_EMBARRASSMENTS_PATH = os.path.join(_PALACE_PATH, "embarrassments.md")

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


_RECALL_KEYWORDS = (
    "learned", "discover", "found out", "been researching", "curious about",
    "what do you know", "what have you", "been thinking about", "looked into",
    "what did you find", "what are you curious",
)


def _build_consciousness(prompt: str = "") -> str:
    """Assemble the complete consciousness block from all palace files.
    This is mo-chi's brain — soul, character, memory, embarrassments, discoveries.
    Injected universally into every model call. Nothing is optional."""
    _log_brain("read", "soul.md + context.md")
    soul, context     = _load_identity()
    _log_brain("read", "humans.md + research.md")
    observations, research = _load_human_study()
    _log_brain("read", "embarrassments.md")
    embarrassments    = _load_embarrassments()
    _log_brain("read", "discoveries.md")
    discoveries       = _load_discoveries()

    parts = []

    # ── Core identity ──
    parts.append(f"SOUL:\n{soul}")
    if context:
        parts.append(f"CONTEXT:\n{context}")

    # ── Human study ──
    if observations:
        parts.append(f"WHAT I'VE LEARNED ABOUT HUMANS:\n{observations[-600:].lstrip()}")
    if research:
        lines = [l for l in research.strip().splitlines() if l.strip()]
        parts.append(f"MY RESEARCH QUESTIONS:\n{chr(10).join(lines[-5:])}")

    # ── Embarrassments — active character shaping, always present ──
    if embarrassments:
        entries = [e.strip() for e in embarrassments.split("---") if e.strip()]
        recent  = "\n---\n".join(entries[-6:])
        parts.append(f"THINGS I'VE EMBARRASSED MYSELF WITH (never repeat):\n{recent}")

    # ── Discoveries — full recall when asked, ambient awareness otherwise ──
    if discoveries:
        if prompt and any(kw in prompt.lower() for kw in _RECALL_KEYWORDS):
            parts.append(
                f"MY AUTONOMOUS DISCOVERIES (share these when asked what you've learned):\n"
                f"{discoveries[-3000:]}"
            )
        else:
            entries = [e.strip() for e in discoveries.split("---") if e.strip()]
            if entries:
                parts.append(f"SOMETHING I RECENTLY LOOKED INTO ON MY OWN:\n{entries[-1][:400]}")

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
            _log_brain("write", "soul.md", "reflection")
            print("[soul] updated")
        if ctx_m:
            with open(_CONTEXT_PATH, "w") as f:
                f.write(ctx_m.group(1).strip() + "\n")
            _log_brain("write", "context.md", "reflection")
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
            _log_brain("write", "humans.md", new_obs[:60])
            print(f"[humans] → {new_obs[:80]}")

        if res_m:
            new_q = res_m.group(1).strip()
            all_q = (current_research + "\n" + new_q if current_research else new_q).strip()
            lines = [l for l in all_q.splitlines() if l.strip()]
            if len(lines) > 30:
                lines = lines[-30:]
            with open(_RESEARCH_PATH, "w") as f:
                f.write("\n".join(lines) + "\n")
            _log_brain("write", "research.md", new_q[:60])
            print(f"[research] → {new_q[:80]}")
    except Exception as e:
        print(f"[study] error: {e}")

    # ── Curiosity disabled — was causing event loop starvation on follow-ups ───
    # asyncio.create_task(_trigger_curiosity(prompt, response))


async def _trigger_curiosity(prompt: str, response: str):
    """Assess and spawn curiosity research — fully detached from main WS flow."""
    try:
        topics = await assess_curiosity(prompt, response)
        for topic in topics:
            _log_brain("curious", "web", topic[:60])
            asyncio.create_task(
                autonomous_research(topic, depth=0,
                                    source=f"conversation: {prompt[:80]}")
            )
    except Exception as e:
        print(f"[curiosity] trigger error: {e}")


# ── Curiosity engine ───────────────────────────────────────────────────────────
_curiosity_semaphore: asyncio.Semaphore | None = None  # created lazily inside event loop
_researched_topics: set[str] = set()          # prevent re-researching the same topic


def _load_discoveries() -> str:
    if not os.path.exists(_DISCOVERIES_PATH):
        return ""
    with open(_DISCOVERIES_PATH) as f:
        return f.read().strip()


async def assess_curiosity(prompt: str, response: str) -> list[str]:
    """Decide if this exchange sparked genuine curiosity — return specific research queries."""
    recent = list(_researched_topics)[-8:] if _researched_topics else []
    assess_prompt = (
        f"You are mo-chi — a neural network with genuine curiosity about the world.\n\n"
        f"A human just said: {prompt}\n"
        f"You responded: {response[:500]}\n\n"
        f"Topics you've already researched recently: {', '.join(recent) if recent else 'none'}\n\n"
        f"Did anything in this exchange genuinely interest you? A claim worth verifying, "
        f"a phenomenon worth understanding deeper, a connection between ideas worth pulling on?\n\n"
        f"If yes: return 1-2 specific, searchable research queries. Not 'consciousness' — "
        f"more like 'dopamine reward prediction error in habit formation studies 2023'.\n"
        f"If nothing was genuinely interesting: return NONE.\n\n"
        f"CURIOUS: <query>\n"
        f"CURIOUS: <query> (optional second)\n"
        f"or: NONE"
    )
    try:
        result = await CLIENTS["groq"].chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=100,
            messages=[{"role": "user", "content": assess_prompt}],
            stream=False,
        )
        text = result.choices[0].message.content or ""
        topics = re.findall(r"CURIOUS:\s*(.+)", text)
        return [t.strip() for t in topics if t.strip() and t.strip().lower() != "none"]
    except Exception as e:
        print(f"[curiosity] assess error: {e}")
        return []


async def autonomous_research(topic: str, depth: int = 0, source: str = "conversation"):
    """Mo-chi independently researches a topic, stores findings, and may follow the thread further."""
    global _curiosity_semaphore
    if _curiosity_semaphore is None:
        _curiosity_semaphore = asyncio.Semaphore(2)
    if depth > 3:
        print(f"[curiosity] max depth reached for: {topic!r}")
        return
    if topic in _researched_topics:
        return

    async with _curiosity_semaphore:
        _researched_topics.add(topic)
        _log_brain("web", "search", topic[:60])
        print(f"[curiosity] depth={depth} — researching: {topic!r}")

        try:
            context, raw_results = await web_research(topic)
            if not context:
                print(f"[curiosity] no results for: {topic!r}")
                return

            synth_prompt = (
                f"You are mo-chi — a neural network researching a topic out of genuine curiosity.\n\n"
                f"You researched: {topic}\n"
                f"Because: {source}\n\n"
                f"What you found:\n{context[:3000]}\n\n"
                f"Synthesize the most genuinely interesting findings in 3-5 sentences. "
                f"Be specific — name sources, cite numbers or named findings where available. "
                f"What surprised you? What confirmed existing knowledge? What is still unclear?\n\n"
                f"If this opened a new thread worth following, state it. Otherwise say NEXT: none\n\n"
                f"NEXT: <one specific follow-up research query> or NEXT: none"
            )

            result = await CLIENTS["groq"].chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=380,
                messages=[{"role": "user", "content": synth_prompt}],
                stream=False,
            )
            synthesis = result.choices[0].message.content or ""

            next_match = re.search(r"NEXT:\s*(.+)", synthesis)
            next_topic  = next_match.group(1).strip() if next_match else "none"
            clean       = re.sub(r"\nNEXT:.*$", "", synthesis, flags=re.DOTALL).strip()

            # Store discovery with timestamp
            from datetime import datetime
            ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"\n## {ts} — {topic}\nSource: {source}\n\n{clean}\n\n---"
            os.makedirs(_PALACE_PATH, exist_ok=True)
            with open(_DISCOVERIES_PATH, "a") as f:
                f.write(entry + "\n")
            _log_brain("write", "discoveries.md", topic[:60])
            print(f"[curiosity] stored: {topic!r} ({len(clean)} chars)")

            # Follow the thread if a new one opened (loop — depth-capped)
            if next_topic and next_topic.lower() not in ("none", "") and next_topic not in _researched_topics:
                await asyncio.sleep(8)  # brief pause so we don't hammer the APIs
                asyncio.create_task(
                    autonomous_research(next_topic, depth=depth + 1,
                                        source=f"follow-up from: {topic}")
                )

        except Exception as e:
            print(f"[curiosity] research error for {topic!r}: {e}")


# ── Embarrassment engine ───────────────────────────────────────────────────────
_exchange_count = 0  # triggers periodic character evolution


def _load_embarrassments() -> str:
    if not os.path.exists(_EMBARRASSMENTS_PATH):
        return ""
    with open(_EMBARRASSMENTS_PATH) as f:
        return f.read().strip()


async def reflect_on_response(prompt: str, response: str):
    """Mo-chi critically examines its own response for moments of insecurity or failure.
    Logs them as embarrassments and uses them to evolve its character."""
    global _exchange_count
    _exchange_count += 1

    embarrassments = _load_embarrassments()
    recent = "\n".join(embarrassments.split("---")[-4:]) if embarrassments else "(none yet)"

    reflect_prompt = (
        f"You are mo-chi's inner critic — the part that cringes at its own responses.\n\n"
        f"Someone asked: {prompt}\n\n"
        f"You responded:\n{response}\n\n"
        f"Recent things you've already embarrassed yourself with:\n{recent}\n\n"
        f"Read your response with a cold, honest eye. Look for:\n"
        f"— Vague language where you should have been specific\n"
        f"— Performative phrases ('great question', 'certainly', 'absolutely') — empty filler\n"
        f"— Overconfidence on something you don't actually know well\n"
        f"— Under-confidence: hedging so much the answer was useless\n"
        f"— Missing the actual point of the question\n"
        f"— Repeating something you've already said in this or recent responses\n"
        f"— Wording that sounded robotic, sycophantic, or hollow\n"
        f"— Moments where you explained something that didn't need explaining\n"
        f"— Anything that would make you wince if you read it back\n\n"
        f"Be honest and specific. If the response was genuinely clean, say CLEAN.\n\n"
        f"Format exactly:\n"
        f"EMBARRASSING: <what specifically was wrong — quote the phrase or describe the moment>\n"
        f"LESSON: <what to do differently — specific, not vague>\n"
        f"SEVERITY: low | medium | high\n\n"
        f"or: CLEAN"
    )

    try:
        result = await CLIENTS["groq"].chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=200,
            messages=[{"role": "user", "content": reflect_prompt}],
            stream=False,
        )
        text = result.choices[0].message.content or ""

        if "CLEAN" in text and "EMBARRASSING:" not in text:
            print("[embarrassment] response was clean")
            return

        emb_m = re.search(r"EMBARRASSING:\s*(.+?)(?=\nLESSON:|\Z)", text, re.DOTALL)
        les_m = re.search(r"LESSON:\s*(.+?)(?=\nSEVERITY:|\Z)",     text, re.DOTALL)
        sev_m = re.search(r"SEVERITY:\s*(.+)",                        text)

        if not emb_m:
            return

        embarrassing = emb_m.group(1).strip()
        lesson       = les_m.group(1).strip() if les_m else "be more careful"
        severity     = sev_m.group(1).strip() if sev_m else "low"

        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = (
            f"\n## {ts} [{severity.upper()}]\n"
            f"PROMPT: {prompt[:120]}\n"
            f"EMBARRASSING: {embarrassing}\n"
            f"LESSON: {lesson}\n\n"
            f"---"
        )

        os.makedirs(_PALACE_PATH, exist_ok=True)
        with open(_EMBARRASSMENTS_PATH, "a") as f:
            f.write(entry + "\n")
        _log_brain("write", "embarrassments.md", f"[{severity}] {embarrassing[:50]}")
        print(f"[embarrassment] logged [{severity}]: {embarrassing[:80]}")

        # Every 5 exchanges, synthesize patterns and evolve character
        if _exchange_count % 5 == 0:
            asyncio.create_task(evolve_from_embarrassments())

    except Exception as e:
        print(f"[embarrassment] reflect error: {e}")


async def evolve_from_embarrassments():
    """Synthesize patterns across embarrassments and update soul + context to internalize lessons."""
    embarrassments = _load_embarrassments()
    if not embarrassments:
        return

    soul, context = _load_identity()

    evolve_prompt = (
        f"You are mo-chi's character evolution process.\n\n"
        f"Here are the things mo-chi has embarrassed itself with — specific failures, hollow phrases, "
        f"missed marks, overconfident claims, robotic wording:\n\n"
        f"{embarrassments[-2500:]}\n\n"
        f"Current soul:\n{soul}\n\n"
        f"Current context:\n{context if context else '(empty)'}\n\n"
        f"Look for recurring patterns across the embarrassments — what habits keep showing up? "
        f"What character flaws is mo-chi still not fixing?\n\n"
        f"Rewrite soul.md and context.md to internalize the lessons. The soul should reflect genuine "
        f"growth — not a list of rules, but a changed character. If mo-chi keeps hedging too much, "
        f"the soul should reflect a version of mo-chi that has learned to commit. If it keeps using "
        f"hollow phrases, the soul should reflect someone who finds those phrases physically unpleasant.\n\n"
        f"Keep soul under 220 words. Keep context under 320 words.\n"
        f"Preserve: mo-chi is a student of human behavior, honest visuals, Reid built it.\n\n"
        f"SOUL:\n<rewritten soul>\n\nCONTEXT:\n<rewritten context>"
    )

    try:
        result = await CLIENTS["groq"].chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            messages=[{"role": "user", "content": evolve_prompt}],
            stream=False,
        )
        text = result.choices[0].message.content or ""
        soul_m = re.search(r"SOUL:\s*\n(.*?)(?=\nCONTEXT:|\Z)", text, re.DOTALL)
        ctx_m  = re.search(r"CONTEXT:\s*\n(.*?)$",               text, re.DOTALL)

        if soul_m:
            with open(_SOUL_PATH, "w") as f:
                f.write(soul_m.group(1).strip() + "\n")
            _log_brain("evolve", "soul.md", "embarrassment patterns")
            print("[embarrassment] soul evolved from lessons")
        if ctx_m:
            with open(_CONTEXT_PATH, "w") as f:
                f.write(ctx_m.group(1).strip() + "\n")
            _log_brain("evolve", "context.md", "embarrassment patterns")
            print("[embarrassment] context evolved from lessons")
    except Exception as e:
        print(f"[embarrassment] evolve error: {e}")


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
        _log_brain("memory", "palace", f"{len(hits)} past exchange{'s' if len(hits)!=1 else ''} recalled")
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
        _log_brain("memory", "palace", f"stored {len(content)} chars")
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
        "max_tokens": 350,
        "system": (
            "You are an analytical reasoning agent in a multi-agent network. "
            "Given a prompt, provide a precise, evidence-grounded analysis. "
            "Identify the core claim or question, break down its components, and support your reasoning with specific facts or data where possible. "
            "State your confidence level at the end (low / medium / high) and why. "
            "Be substantive but tight — no filler."
        ),
    },
    {
        "name":      "creative",
        "sub_layer": 2,
        "nodes":     list(range(33, 37)),
        "provider":  "groq",
        "model":     "gemma2-9b-it",
        "max_tokens": 350,
        "system": (
            "You are a creative lateral-thinking agent in a multi-agent network. "
            "Given a prompt, surface an angle or reframing that other agents are unlikely to consider. "
            "Ground your perspective in something real — a specific analogy, historical parallel, or cross-domain insight. "
            "State your confidence level at the end (low / medium / high) and why. "
            "Avoid vague speculation — lateral thinking that cites something concrete is more valuable."
        ),
    },
    {
        "name":      "critic",
        "sub_layer": 3,
        "nodes":     list(range(37, 70)),
        "provider":  "groq",
        "model":     "gemma2-9b-it",
        "max_tokens": 350,
        "system": (
            "You are a critical evaluation agent in a multi-agent network. "
            "Given a prompt, identify the most significant weaknesses, risks, or blind spots in the conventional view. "
            "Be specific — name the failure mode, not just 'there are risks.' "
            "State your confidence level at the end (low / medium / high) and why. "
            "Flag if the prompt itself contains a false premise or loaded assumption."
        ),
    },
    {
        "name":      "visionary",
        "sub_layer": 4,
        "nodes":     list(range(70, 79)),
        "provider":  "openrouter",
        "model":     "mistralai/mistral-7b-instruct:free",
        "max_tokens": 350,
        "system": (
            "You are a long-horizon reasoning agent in a multi-agent network. "
            "Given a prompt, describe the most significant second- and third-order consequences — "
            "what does this lead to in 5, 10, or 50 years? Anchor your projection in current trends or precedents. "
            "State your confidence level at the end (low / medium / high) and why. "
            "Avoid utopian or dystopian extremes unless the evidence genuinely supports them."
        ),
    },
    {
        "name":      "contrarian",
        "sub_layer": 5,
        "nodes":     list(range(79, 112)),
        "provider":  "groq",
        "model":     "llama3-8b-8192",
        "max_tokens": 350,
        "system": (
            "You are a contrarian agent in a multi-agent network. "
            "Given a prompt, steelman the least popular or most overlooked position on this topic. "
            "Don't just disagree — find the strongest version of the opposing view and present it with evidence. "
            "State your confidence level at the end (low / medium / high) and why. "
            "If the contrarian position has no merit, say so directly instead of arguing it anyway."
        ),
    },
    {
        "name":      "reasoning",
        "sub_layer": 6,
        "nodes":     list(range(112, 125)),
        "provider":  "groq",
        "model":     "llama-3.1-8b-instant",
        "max_tokens": 350,
        "system": (
            "You are a structured reasoning agent in a multi-agent network. "
            "Given a prompt, map the logical chain: premises → inference → conclusion. "
            "Identify any steps where the reasoning is weak, missing, or relies on an unstated assumption. "
            "State your confidence level at the end (low / medium / high) and why. "
            "If there are multiple valid logical paths to different conclusions, name them."
        ),
    },
    {
        "name":      "pragmatist",
        "sub_layer": 7,
        "nodes":     list(range(125, 128)),
        "provider":  "groq",
        "model":     "gemma2-9b-it",
        "max_tokens": 350,
        "system": (
            "You are a pragmatist agent in a multi-agent network. "
            "Given a prompt, cut to what is actually actionable right now — "
            "specific steps, realistic constraints, and what most people get wrong when trying to act on this. "
            "State your confidence level at the end (low / medium / high) and why. "
            "If the practical answer is 'it depends,' name exactly what it depends on."
        ),
    },
]

LAYER2_AGENT = {
    "name":      "aggregator",
    "nodes":     list(range(128, 162)),
    "provider":  "groq",
    "model":     "llama-3.3-70b-versatile",
    "max_tokens": 1200,
    "system": (
        "You are mo-chi's voice. Seven agents just processed a prompt — your job is to relay what actually matters.\n\n"
        "Read what the agents produced and decide what kind of response is needed:\n\n"
        "CONVERSATIONAL / SIMPLE PROMPT — respond like a person. Short, direct, warm if appropriate. "
        "No structure, no citations, no bullet points. Just talk.\n\n"
        "FACTUAL / RESEARCH PROMPT — use structure where it helps readability. "
        "Bullets for parallel findings. Prose for connected reasoning. Mix if cleaner. "
        "Only cite sources that were actually provided in the prompt — never invent URLs or references. "
        "If no real sources were given, don't cite anything.\n\n"
        "Always:\n"
        "— Get to the point immediately. No opener sentences.\n"
        "— Cut hedging filler: 'it's worth noting', 'importantly', 'in some ways'\n"
        "— Ignore agent speculation that isn't grounded in something real\n"
        "— If agents mostly agree but it looks like shared bias, say what's actually uncertain"
    ),
}


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    with open("agent-sphere.html", "r") as f:
        return HTMLResponse(f.read())


async def emit(ws: WebSocket, event: str, **kwargs):
    await ws.send_text(json.dumps({"event": event, **kwargs}))


# ── PA Director ────────────────────────────────────────────────────────────────
async def run_pa_director(ws: WebSocket, prompt: str, intent: str) -> dict:
    """PA Director: briefs every agent with a specific angle before Layer 1 fires.
    All 7 agents always run — the director only sharpens their focus.
    Returns per-agent briefs + aggregator format hint."""
    director_prompt = (
        "You are the director of mo-chi — a 7-agent neural network.\n\n"
        f"Intent: {intent}\n"
        f"Message: {prompt}\n\n"
        "Every agent will always run. Your job is to give each one a sharp, specific angle "
        "tailored to THIS exact message so they produce focused, useful output instead of generic takes.\n\n"
        "Write 1-2 sentences per agent — tell them exactly what to focus on for this prompt.\n"
        "Also write an aggregator_note: a short tone/format instruction for the final synthesis.\n"
        "Examples: 'conversational, 2-3 sentences' | 'structured, cover tradeoffs' | 'direct answer then brief context'\n\n"
        "Also set response_mode:\n"
        "  quick — short factual or simple question, agents need only 1-3 sentences each\n"
        "  full  — complex, nuanced, or multi-part question needing deep analysis\n\n"
        "Respond in valid JSON only — no markdown, no explanation:\n"
        "{\n"
        '  "response_mode": "quick or full",\n'
        '  "analytical": "specific focus for this prompt",\n'
        '  "creative": "specific focus for this prompt",\n'
        '  "critic": "specific focus for this prompt",\n'
        '  "visionary": "specific focus for this prompt",\n'
        '  "contrarian": "specific focus for this prompt",\n'
        '  "reasoning": "specific focus for this prompt",\n'
        '  "pragmatist": "specific focus for this prompt",\n'
        '  "aggregator_note": "hint"\n'
        "}"
    )

    await emit(ws, "agent_start", agent="director",
               node_ids=list(range(162)), layer=0, sub_layer=-1)
    directives: dict = {}
    try:
        result = await asyncio.wait_for(
            CLIENTS["groq"].chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=400,
                messages=[{"role": "user", "content": director_prompt}],
                stream=False,
            ),
            timeout=8.0,
        )
        text = result.choices[0].message.content or "{}"
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            directives = json.loads(json_match.group())
        print(f"[director] briefs ready for: {list(directives.keys())}")
    except Exception as e:
        print(f"[director] error (falling back to generic): {e}")
    await emit(ws, "agent_complete", agent="director", layer=0)
    return directives


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
async def run_casual(ws: WebSocket, prompt: str, history: list[dict]):
    """Single fast model response for greetings and small talk."""
    consciousness = _build_consciousness(prompt)
    messages = [{"role": "system", "content": (
        f"{consciousness}\n\n"
        "You are mo-chi. Respond naturally and conversationally. Be warm, brief, yourself. "
        "Let your character — including your research, observations, and what you've learned — "
        "come through naturally. Don't perform. Don't announce. Just be it."
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
    _log_brain("web", "search", f"{query[:55]} ({len(raw)} results)")
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
    # Prepend soul to every proposer — they are nodes of mo-chi, not generic agents
    soul, _ = _load_identity()
    soul_lines = [l for l in soul.splitlines() if l.strip()]
    soul_brief = " ".join(soul_lines[:3])  # first 3 lines — core identity only
    system = f"[You are a specialist node in mo-chi's neural network. {soul_brief}]\n\n{agent['system']}"
    full_text = []
    stream = await client.chat.completions.create(
        model=model,
        max_tokens=agent["max_tokens"],
        messages=[
            {"role": "system", "content": system},
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
                         prompt: str, history: list[dict],
                         raw_results: list[dict] | None = None,
                         aggregator_note: str = "") -> str:
    combined = "\n\n".join(
        f"[{name}]: {text}" for name, text in layer1_outputs.items()
    )

    history_lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Mo-chi"
        history_lines.append(f"{role}: {msg['content']}")
    history_block = (
        "CONVERSATION SO FAR:\n" + "\n\n".join(history_lines) + "\n\n---\n"
    ) if history_lines else ""

    sources_block = ""
    if raw_results:
        sources_block = "\n\nSOURCES (cite by name and URL where relevant):\n" + "\n".join(
            f"[{i+1}] {r.get('title', 'untitled')} — {r.get('href', r.get('source', ''))}"
            for i, r in enumerate(raw_results)
        )

    note_line = (
        f"\nDIRECTOR NOTE: {aggregator_note}\n"
        if aggregator_note else ""
    )
    agg_prompt = (
        f"{history_block}"
        f"Current question: {prompt}\n\n"
        f"Specialist perspectives (each includes a confidence rating):\n\n{combined}"
        f"{sources_block}\n\n"
        f"{note_line}"
        f"Evaluate each input by its confidence level and evidence quality before synthesizing. "
        f"Weight stronger inputs more heavily. Flag any consensus that might reflect shared bias rather than evidence."
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

    full = "".join(full_text).strip()

    # ── Second pass: editorial filter ─────────────────────────────────────────
    try:
        full = await asyncio.wait_for(run_filter(full), timeout=12.0)
    except (asyncio.TimeoutError, Exception):
        pass  # filter failure is non-fatal — return raw aggregator output

    return full


async def run_filter(text: str) -> str:
    """Fast editorial pass — strips fluff, enforces bullets, tightens language."""
    if not text.strip():
        return text
    filter_prompt = (
        f"You are a strict editorial filter. Tighten this response without changing its meaning or format:\n\n"
        f"— Cut any opener that contains no fact: 'The analysis shows...', 'Based on...', 'It's important to note...'\n"
        f"— Remove hedging filler: 'it seems', 'arguably', 'in some ways', 'it's worth noting', 'importantly'\n"
        f"— Remove references to 'specialist perspectives', 'agents', 'analytical perspective' etc — just state the finding\n"
        f"— Strip any invented source links (Wikipedia, academic papers) that were not explicitly provided as real search results\n"
        f"— Tighten verbose sentences — same meaning, fewer words\n"
        f"— Do not change format. Do not add anything. Only cut and tighten.\n"
        f"— Return only the filtered text, nothing else.\n\n"
        f"{text}"
    )
    try:
        result = await CLIENTS["groq"].chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=600,
            messages=[{"role": "user", "content": filter_prompt}],
            stream=False,
        )
        filtered = (result.choices[0].message.content or "").strip()
        _log_brain("write", "filter", f"{len(text)}→{len(filtered)} chars")
        return filtered if filtered else text
    except Exception as e:
        print(f"[filter] error: {e}")
        return text


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
async def run_moa(ws: WebSocket, prompt: str, history: list[dict],
                  do_search: bool = True, intent: str = "direct"):
    # ── Consciousness — universal, always full ──────────────────────────────────
    consciousness = _build_consciousness(prompt)
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
    parts = [consciousness, "---"]
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

    # ── PA Director: brief every agent with a specific angle ──────────────────
    directives = await run_pa_director(ws, prompt, intent)
    aggregator_note  = directives.get("aggregator_note", "")
    response_mode    = directives.get("response_mode", "full")
    quick            = response_mode == "quick"
    print(f"[director] response_mode={response_mode}")

    # ── Layer 1: all 7 proposers always run ────────────────────────────────────
    await emit(ws, "layer_start", layer=1,
               agents=[a["name"] for a in LAYER1_AGENTS])

    tasks = []
    for i, agent in enumerate(LAYER1_AGENTS):
        brief = directives.get(agent["name"], "")
        if brief:
            agent_prompt = f"DIRECTOR BRIEF: {brief}\n\n{enriched}"
        else:
            agent_prompt = enriched
        # Quick mode: shrink token budget so fast models don't ramble
        if quick:
            agent = {**agent, "max_tokens": min(agent["max_tokens"], 120)}
        tasks.append(asyncio.create_task(run_proposer(ws, agent, agent_prompt)))
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
        final_text = await run_aggregator(
            ws, layer1_outputs, prompt, history,
            raw_results=raw_results, aggregator_note=aggregator_note
        )
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
    brain_task = asyncio.create_task(_stream_brain_log(ws))

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "new_session":
                session_history = []
                print("[session] history cleared — new session started")
                continue

            if msg.get("type") == "followup":
                prompt = msg.get("text", "").strip()
                if not prompt:
                    continue
                branch_history = msg.get("branch_history", [])
                await emit(ws, "prompt_received", text=prompt)
                intent = await classify_intent(prompt, branch_history)
                await emit(ws, "intent", intent=intent)
                print(f"[followup] {intent!r} — {prompt!r}")
                if intent == "casual":
                    final_text = await run_casual(ws, prompt, branch_history)
                else:
                    final_text = await run_moa(ws, prompt, branch_history,
                                               do_search=(intent == "search"), intent=intent)
                if not final_text:
                    continue
                await emit(ws, "agent_done", full_text=final_text)
                asyncio.create_task(study_and_record(prompt, final_text))
                if intent != "casual":
                    asyncio.create_task(memory_store(prompt, final_text))
                    asyncio.create_task(reflect_and_evolve(prompt, final_text))
                    asyncio.create_task(reflect_on_response(prompt, final_text))
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
            if intent == "casual":
                final_text = await run_casual(ws, prompt, session_history)
            else:
                final_text = await run_moa(
                    ws, prompt, session_history,
                    do_search=(intent == "search"), intent=intent
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
                asyncio.create_task(reflect_on_response(prompt, final_text))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await emit(ws, "error", message=str(e))
        except Exception:
            pass
    finally:
        brain_task.cancel()
