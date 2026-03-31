"""
llm_client.py — The multi-LLM router for varyAI (v4)

This module now supports multiple LLMs through a unified interface.
All models currently run on Groq's free tier, giving users the ability
to switch between different model architectures while keeping the same
memory profile.

AI Engineering Concept — The adapter pattern at scale:
In v1 we isolated LLM code here as a future-proofing measure.
In v4 that investment pays off — we add multi-model support by
changing only this file. Everything else (extraction, retrieval,
prompt building) is completely unchanged.

AI Engineering Concept — Model diversity:
Different LLMs have genuinely different strengths:
- Llama 3.3 70B: Best overall reasoning, most capable
- Gemma 2 9B: Faster, more concise, good for quick questions
- Mixtral 8x7B: Mixture of experts architecture, strong at analysis

Having the same memory profile work across all three means users
can choose the right model for the right task without losing context.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from backend.prompt_builder import build_system_prompt

load_dotenv()

# ─────────────────────────────────────────────────────
# Available models registry
# This is the single source of truth for supported models.
# Adding a new model is just adding an entry here.
# ─────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "llama-3.3-70b": {
        "id":          "llama-3.3-70b-versatile",
        "name":        "Llama 3.3 · 70B",
        "provider":    "Groq",
        "description": "Most capable. Best for complex reasoning and detailed responses.",
        "temperature": 0.7,
    },
    "llama-3.1-8b": {
        "id":          "llama-3.1-8b-instant",
        "name":        "Llama 3.1 · 8B",
        "provider":    "Groq",
        "description": "Lightning fast. Best for quick questions and simple tasks.",
        "temperature": 0.7,
    },
    "llama-4-scout": {
        "id":          "meta-llama/llama-4-scout-17b-16e-instruct",
        "name":        "Llama 4 Scout · 17B",
        "provider":    "Groq",
        "description": "Meta's latest model. Fast, multimodal, great all-rounder.",
        "temperature": 0.7,
    },
}

# Default model if none specified
DEFAULT_MODEL = "llama-3.3-70b"


def initialize_model():
    """
    Initialize and return the Groq client.

    The client is model-agnostic — the model is chosen per request
    in stream_response(). This means switching models mid-conversation
    requires no reinitialization.

    Returns:
        An initialized Groq client

    Raises:
        ValueError: If GROQ_API_KEY is not found in environment
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Make sure you have a .env file with your API key. "
            "See .env.example for the format."
        )

    client = Groq(api_key=api_key)
    print(f"✓ Groq client initialized ({len(AVAILABLE_MODELS)} models available)")
    return client


def get_available_models() -> list[dict]:
    """
    Return the list of available models for the frontend dropdown.

    Returns:
        List of model info dicts with key, name, and description
    """
    return [
        {
            "key":         key,
            "name":        model["name"],
            "description": model["description"],
        }
        for key, model in AVAILABLE_MODELS.items()
    ]


def stream_response(client, messages: list[dict], model_key: str = DEFAULT_MODEL):
    """
    Send a conversation to the chosen model and stream the response.

    Args:
        client:    The initialized Groq client
        messages:  Full conversation history in standard chat format
        model_key: Key from AVAILABLE_MODELS (e.g. "llama-3.3-70b")

    Yields:
        str: Individual text chunks as they stream from the API

    AI Engineering Concept — Model-agnostic interface:
    The rest of the codebase passes messages in a standard format
    and gets back a stream of text chunks. It doesn't need to know
    which model is being used — that detail is fully encapsulated here.
    This is the power of a well-designed adapter layer.
    """

    # Resolve model config — fall back to default if key not found
    model_config = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS[DEFAULT_MODEL])

    # Use recent context for richer RAG retrieval
    recent_context = " ".join([m["content"] for m in messages[-4:]])
    latest_query = recent_context if recent_context else ""

    # Build full messages with system prompt
    full_messages = [
        {"role": "system", "content": build_system_prompt(query=latest_query)}
    ] + messages

    # Stream response from chosen model
    response = client.chat.completions.create(
        model=model_config["id"],
        messages=full_messages,
        stream=True,
        temperature=model_config["temperature"],
        max_tokens=2048,
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def refresh_system_prompt(client):
    """
    System prompt is rebuilt fresh on every request.
    Returns the same client unchanged.
    """
    print("✓ System prompt will refresh on next request")
    return client