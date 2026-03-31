"""
main.py — The central nervous system of varyAI

This is the FastAPI backend server. It receives messages from the
frontend, orchestrates the entire pipeline, and streams responses back.

The pipeline for every single message:
1. Receive user message from frontend
2. Add it to conversation history
3. Build enriched messages with profile context
4. Stream response from Gemini back to frontend
5. Save assistant response to history
6. Trigger extraction in background
7. Refresh system prompt with updated profile

AI Engineering Concept — Why FastAPI?
FastAPI is built on top of Starlette and uses Python's async/await
system. This matters for LLM applications because API calls are
I/O bound — while waiting for Gemini to respond, the server can
handle other requests instead of blocking. FastAPI also has native
support for streaming responses, which is essential for the
typewriter effect in the UI.

AI Engineering Concept — Why a local server?
Instead of calling the Gemini API directly from the browser,
we route through a local Python server. This is important because:
1. API keys must never be exposed in frontend code (browser)
2. We need to run Python code (extraction, SQLite) on the backend
3. We can add logic, validation, and error handling in one place
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.profile_store import initialize_database, get_profile_summary
from backend.llm_client import initialize_model, stream_response, refresh_system_prompt
from backend.extraction import extract_and_save
from backend.prompt_builder import build_conversation_messages


# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────

# The Gemini model instance — initialized once, reused for all requests
# We store it here so extraction.py can reuse it too (dependency injection)
model = None

# In-memory conversation history for the current session
# Each entry: {"role": "user" | "assistant", "content": "..."}
# This gets cleared when the server restarts — persistence across
# sessions comes from the profile, not the raw conversation history
conversation_history = []


# ─────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs setup code before the server starts accepting requests,
    and cleanup code when it shuts down.

    AI Engineering Concept — Lifespan events:
    We initialize expensive resources (database, model) once at startup
    rather than on every request. Creating a database connection or
    initializing an LLM model on every request would be slow and wasteful.
    This is standard practice in production API servers.
    """
    global model

    print("\n🚀 Starting varyAI backend...")

    # Step 1: Initialize the database (creates tables if they don't exist)
    initialize_database()

    # Step 2: Initialize the Gemini model with current profile
    model = initialize_model()

    print("✓ varyAI is ready\n")

    yield  # Server runs here, handling requests

    # Cleanup on shutdown (if needed in future)
    print("\n👋 varyAI backend shutting down")


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="varyAI",
    description="A local-first AI assistant with shared memory across LLMs",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware — allows the frontend (running on a different port)
# to make requests to this backend without being blocked by the browser
# In production you'd restrict this to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    The shape of data the frontend sends with every message.
    Pydantic automatically validates this — if 'message' is missing
    or not a string, FastAPI returns a 400 error automatically.
    """
    message: str


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Simple endpoint to verify the server is running.
    The frontend calls this on load to confirm backend is up.
    """
    return {"status": "ok", "message": "varyAI backend is running"}


@app.get("/profile")
async def get_profile():
    """
    Returns the current user profile as formatted text.
    The frontend uses this to show what varyAI knows about you.
    """
    summary = get_profile_summary()
    return {"profile": summary}


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    The main endpoint. Receives a user message, streams back
    the LLM response, and triggers background extraction.

    Returns a StreamingResponse — tokens arrive at the frontend
    as they're generated, enabling the typewriter effect.

    AI Engineering Concept — async generators for streaming:
    FastAPI's StreamingResponse accepts an async generator.
    We wrap our synchronous Gemini stream in an async generator
    so FastAPI can stream it properly over HTTP as Server-Sent Events.
    """
    global model, conversation_history

    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Build the full messages array (history + new message)
    messages = build_conversation_messages(conversation_history, user_message)

    # Collect the full assistant response as we stream it
    # We need the complete response for extraction after streaming finishes
    full_response = []

    async def generate():
        """
        Async generator that streams tokens to the frontend
        and collects the full response for post-processing.

        AI Engineering Concept — Why collect while streaming?
        We need the complete response text to run extraction on it.
        But we can't wait for the full response before streaming —
        that would defeat the purpose. So we stream AND collect
        simultaneously, then run extraction after streaming completes.
        """
        global model

        try:
            # Stream tokens from Gemini
            for chunk in stream_response(model, messages):
                full_response.append(chunk)
                yield chunk

            # Streaming complete — now run post-processing
            complete_response = "".join(full_response)

            # Add both messages to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_message
            })
            conversation_history.append({
                "role": "assistant",
                "content": complete_response
            })

            # Run extraction on the latest exchange
            # We pass only the last 2 messages (current exchange) for extraction
            # rather than full history — we only want facts from THIS exchange
            current_exchange = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": complete_response}
            ]

            # Extract facts in background — don't block the response
            # asyncio.to_thread runs synchronous code in a thread pool
            # so it doesn't block the async event loop
            await asyncio.to_thread(extract_and_save, current_exchange, model)

            # Refresh the system prompt with newly extracted facts
            model = refresh_system_prompt(model)

        except Exception as e:
            print(f"⚠ Error during generation: {e}")
            # Don't show extraction errors to the user
            print(f"⚠ Error during generation: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.delete("/conversation")
async def clear_conversation():
    """
    Clears the in-memory conversation history for the current session.
    Does NOT clear the profile — that persists across sessions.
    """
    global conversation_history
    conversation_history = []
    return {"message": "Conversation cleared"}


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────

# Serve the frontend HTML file as a static file
# This means you only need to run the Python server —
# no separate frontend server needed for v1
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")