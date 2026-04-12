"""
mo-chi · FastAPI + WebSocket backend
Single-agent (option 1) — streams Claude events to the sphere visualization.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import anthropic

load_dotenv()

app = FastAPI(title="mo-chi")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Serve the frontend at /
@app.get("/")
async def root():
    with open("agent-sphere.html", "r") as f:
        return HTMLResponse(f.read())


async def emit(ws: WebSocket, event: str, **kwargs):
    """Send a typed JSON event to the frontend."""
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
                if not prompt:
                    continue
                await run_agent(ws, prompt)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await emit(ws, "error", message=str(e))
        except Exception:
            pass


async def run_agent(ws: WebSocket, prompt: str):
    """
    Single-agent forward pass (option 1).

    Events emitted:
      prompt_received  – server got the prompt
      node_activate    – a node index should start glowing  { node_id, layer }
      token            – a streamed token from the LLM      { text }
      agent_done       – LLM finished                       { full_text }
      error            – something went wrong               { message }
    """

    await emit(ws, "prompt_received", text=prompt)

    # Small stagger: send a handful of node activations as "pre-processing"
    # In option 2 these will map to real parallel agents; for now they fire
    # proportionally as tokens stream in.
    TOTAL_NODES = 42          # matches geodesic detail=2
    activated = set()
    full_text = []
    token_count = 0

    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                full_text.append(text)
                token_count += 1

                # Activate a new node roughly every 3 tokens, ripple outward
                if token_count % 3 == 0:
                    node_id = token_count // 3 % TOTAL_NODES
                    if node_id not in activated:
                        activated.add(node_id)
                        await emit(ws, "node_activate", node_id=node_id, layer=0)

                await emit(ws, "token", text=text)

                # Yield to event loop so WebSocket sends aren't batched
                await asyncio.sleep(0)

        await emit(ws, "agent_done", full_text="".join(full_text))

    except anthropic.APIError as e:
        await emit(ws, "error", message=f"API error: {e}")
    except Exception as e:
        await emit(ws, "error", message=str(e))
