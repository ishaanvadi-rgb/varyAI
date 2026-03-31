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
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.profile_store import (
    initialize_database,
    get_profile_summary,
    save_facts,
    delete_fact,
    create_conversation,
    save_message,
    update_conversation_title,
    get_conversation_messages,
    get_all_conversations,
    delete_conversation
)
from backend.llm_client import initialize_model, stream_response, refresh_system_prompt
from backend.extraction import extract_and_save
from backend.prompt_builder import build_conversation_messages
from backend.retrieval import initialize_retrieval, sync_existing_facts


# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────

# The Gemini model instance — initialized once, reused for all requests
# We store it here so extraction.py can reuse it too (dependency injection)
model = None

# In-memory conversation history for the current session
# Each entry: {"role": "user" | "assistant", "content": "..."}
conversation_history = []

# Current active conversation ID
# None means no conversation started yet
current_conversation_id = None


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
    initialize_retrieval()    # ← add this
    sync_existing_facts()     # ← and this

    # Step 2: Initialize the model with current profile
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
    Now includes model_key so the user can choose which LLM to use.
    """
    message:   str
    model_key: str = "llama-3.3-70b"  # defaults to Llama if not specified


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


@app.get("/profile/full")
async def get_full_profile():
    """
    Returns the full profile with fact IDs for the profile editor.
    """
    from backend.profile_store import get_full_profile_with_ids
    return {"profile": get_full_profile_with_ids()}


@app.delete("/profile/fact/{fact_id}")
async def delete_fact_endpoint(fact_id: int):
    """
    Delete a specific fact from the profile by its ID.
    Also removes it from the ChromaDB vector index.
    """
    delete_fact(fact_id)
    return {"message": "Fact deleted"}


@app.post("/profile/fact")
async def add_fact(request: dict):
    """
    Manually add a fact to the profile.
    Allows users to explicitly tell varyAI something about themselves.
    """
    category = request.get("category")
    content = request.get("content")

    if not category or not content:
        raise HTTPException(status_code=400, detail="category and content required")

    if category not in ["preferences", "projects", "skills", "history"]:
        raise HTTPException(status_code=400, detail="Invalid category")

    save_facts([{"category": category, "content": content}])
    return {"message": "Fact added"}


@app.get("/profile/export")
async def export_profile():
    """
    Export the full profile as a downloadable JSON file.
    This is the portability feature — take your memory anywhere.
    """
    from backend.profile_store import get_full_profile_with_ids
    profile = get_full_profile_with_ids()

    # Clean up for export — remove internal IDs, keep only content
    export_data = {
        "varyai_export": True,
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "profile": {
            category: [fact["content"] for fact in facts]
            for category, facts in profile.items()
        }
    }

    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": "attachment; filename=varyai-profile.json"
        }
    )


@app.post("/profile/import")
async def import_profile(request: dict):
    """
    Import a previously exported profile JSON.
    Merges with existing profile — doesn't overwrite.
    """
    try:
        data = request

        if not data.get("varyai_export"):
            raise HTTPException(status_code=400, detail="Invalid varyAI export file")

        profile = data.get("profile", {})
        imported_count = 0

        for category, facts in profile.items():
            if category not in ["preferences", "projects", "skills", "history"]:
                continue
            for content in facts:
                if content and isinstance(content, str):
                    save_facts([{"category": category, "content": content}])
                    imported_count += 1

        return {"message": f"Imported {imported_count} facts successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/conversations")
async def get_conversations():
    """
    Returns all past conversations for the sidebar history panel.
    """
    conversations = get_all_conversations()
    return {"conversations": conversations}


@app.get("/conversations/{conversation_id}")
async def load_conversation(conversation_id: int):
    """
    Load a past conversation's messages and restore it as the
    active conversation.

    This is how users resume past conversations — the frontend
    calls this, gets the message history, and restores the UI.
    The conversation becomes the active one so new messages
    are appended to it.
    """
    global conversation_history, current_conversation_id

    messages = get_conversation_messages(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Restore as active conversation
    conversation_history = messages
    current_conversation_id = conversation_id

    return {"messages": messages}


@app.delete("/conversations/{conversation_id}")
async def remove_conversation(conversation_id: int):
    """
    Delete a specific past conversation.
    """
    global conversation_history, current_conversation_id

    delete_conversation(conversation_id)

    # If deleting active conversation, reset state
    if current_conversation_id == conversation_id:
        conversation_history = []
        current_conversation_id = None

    return {"message": "Conversation deleted"}


@app.get("/models")
async def get_models():
    """
    Returns the list of available LLMs for the frontend dropdown.
    Called once when the frontend loads.
    """
    from backend.llm_client import get_available_models
    return {"models": get_available_models()}


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
    global model, conversation_history, current_conversation_id

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
            for chunk in stream_response(model, messages, request.model_key):
                full_response.append(chunk)
                yield chunk

            # Streaming complete
            complete_response = "".join(full_response)

            # Save to in-memory history
            conversation_history.append({
                "role": "user",
                "content": user_message
            })
            conversation_history.append({
                "role": "assistant",
                "content": complete_response
            })

            # Persist to SQLite
            # Create conversation on first message
            if current_conversation_id is None:
                globals()['current_conversation_id'] = create_conversation(request.model_key)
                # Use first message as conversation title
                update_conversation_title(current_conversation_id, user_message)

            save_message(current_conversation_id, "user", user_message)
            save_message(current_conversation_id, "assistant", complete_response)

            # Run extraction in background
            current_exchange = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": complete_response}
            ]
            await asyncio.to_thread(extract_and_save, current_exchange, model)
            model = refresh_system_prompt(model)

        except Exception as e:
            # Don't show generation errors to the user
            print(f"⚠ Error during generation: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.delete("/conversation")
async def clear_conversation():
    """
    Clears the active conversation and starts fresh.
    The past conversation is already saved to SQLite.
    """
    global conversation_history, current_conversation_id
    conversation_history = []
    current_conversation_id = None
    return {"message": "Conversation cleared"}


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────

# Serve the frontend HTML file as a static file
# This means you only need to run the Python server —
# no separate frontend server needed for v1
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")