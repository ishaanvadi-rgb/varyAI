"""
llm_client.py — The Groq API connection layer for varyAI

Groq runs open-source models (Llama, Mistral) on custom hardware
called LPUs (Language Processing Units) — making it significantly
faster than GPU-based providers and completely free to use.

AI Engineering Concept — Why Groq for varyAI?
Groq is OpenAI-API-compatible, free, fast, and runs Llama —
an open-source model, meaning no single company controls it.
This aligns perfectly with varyAI's philosophy: open, local-first,
and not locked into any one vendor's ecosystem.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from backend.prompt_builder import build_system_prompt

load_dotenv()

# The model we use — Llama 3.3 70B is Groq's most capable free model
MODEL = "llama-3.3-70b-versatile"


def initialize_model():
    """
    Initialize and return the Groq client.

    No model is loaded here — Groq runs models on their servers.
    We just create an authenticated client that's ready to make calls.

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
    print(f"✓ Groq client initialized (model: {MODEL})")
    return client


def stream_response(client, messages: list[dict]):
    """
    Send a conversation to Groq and stream the response back.

    The system prompt is prepended as the first message with
    role "system" — standard OpenAI-compatible format.

    Args:
        client:   The initialized Groq client from initialize_model()
        messages: Full conversation history in standard chat format

    Yields:
        str: Individual text chunks as they stream from the API
    """

    # Use last 2 exchanges for richer retrieval context
    # More context = better semantic matching
    recent_context = " ".join([m["content"] for m in messages[-4:]])
    latest_query = recent_context if recent_context else ""

    full_messages = [
        {"role": "system", "content": build_system_prompt(query=latest_query)}
    ] + messages

    # Stream the response from Groq
    response = client.chat.completions.create(
        model=MODEL,
        messages=full_messages,
        stream=True,
        temperature=0.7,
        max_tokens=2048,
    )

    # Yield each chunk as it arrives
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def refresh_system_prompt(client):
    """
    System prompt is rebuilt fresh on every request in stream_response(),
    so no reinitialization needed. Returns the same client unchanged.

    This function exists to keep main.py's interface consistent —
    it calls this after every extraction without knowing which LLM
    is underneath.
    """
    print("✓ System prompt will refresh on next request")
    return client