"""
llm_client.py — Multi-provider LLM router for varyAI

This module supports multiple AI providers through a unified interface.
Each provider uses the OpenAI-compatible API format, meaning the same
code works for all of them — only the base_url and api_key change.

AI Engineering Concept — Provider abstraction:
By isolating all provider-specific details here, the rest of varyAI
(extraction, retrieval, prompt building) never needs to know which
provider is being used. This is the adapter pattern at scale.

Supported providers:
- Groq          → Fast inference, Llama models
- Google        → Gemini 2.5 Flash/Pro, frontier quality free
- NVIDIA NIM    → DeepSeek R1, Qwen3, Kimi K2, free tier
- OpenRouter    → Gateway to 29+ models including GPT-4o
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from backend.prompt_builder import build_system_prompt

load_dotenv()

# ─────────────────────────────────────────────────────
# Provider registry
# Each provider needs a base_url and an env variable
# for its API key. All use OpenAI-compatible format.
# ─────────────────────────────────────────────────────
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key":  "GROQ_API_KEY",
        "name":     "Groq",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key":  "OPENROUTER_API_KEY",
        "name":     "OpenRouter",
    },
}

# ─────────────────────────────────────────────────────
# Model registry
# Add any model here — just reference the provider key
# and the exact model ID the provider expects.
# ─────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    # ── Groq ─────────────────────────────────────────
    "llama-3.3-70b": {
        "id":          "llama-3.3-70b-versatile",
        "name":        "Llama 3.3 · 70B",
        "provider":    "groq",
        "description": "Fast and capable. Best all-rounder.",
        "temperature": 0.7,
    },
    "llama-3.1-8b": {
        "id":          "llama-3.1-8b-instant",
        "name":        "Llama 3.1 · 8B",
        "provider":    "groq",
        "description": "Fastest responses. Best for quick questions.",
        "temperature": 0.7,
    },
    "llama-4-scout": {
        "id":          "meta-llama/llama-4-scout-17b-16e-instruct",
        "name":        "Llama 4 Scout · 17B",
        "provider":    "groq",
        "description": "Meta's latest. Fast, multimodal, great all-rounder.",
        "temperature": 0.7,
    },

    # ── OpenRouter ────────────────────────────────────
    "gpt-4o": {
        "id":          "openai/gpt-4o",
        "name":        "GPT-4o",
        "provider":    "openrouter",
        "description": "OpenAI's flagship. Best for writing and reasoning.",
        "temperature": 0.7,
    },
   
    "llama-3.3-free": {
        "id":          "meta-llama/llama-3.3-70b-instruct:free",
        "name":        "Llama 3.3 · 70B (OR)",
        "provider":    "openrouter",
        "description": "Llama 3.3 via OpenRouter free tier.",
        "temperature": 0.7,
    },
    "openrouter-free": {
        "id":          "openrouter/auto",
        "name":        "Auto (Best Free)",
        "provider":    "openrouter",
        "description": "Automatically picks the best available free model.",
        "temperature": 0.7,
    },
}

DEFAULT_MODEL = "llama-3.3-70b"


def get_provider_client(provider_key: str) -> OpenAI | None:
    """
    Create and return an OpenAI-compatible client for a provider.

    Returns None if the provider's API key is not configured —
    this is how we determine which models to show as available
    in the frontend dropdown.

    AI Engineering Concept — OpenAI compatibility:
    Almost every major LLM provider now implements the OpenAI
    API format. This means we can use the same OpenAI Python SDK
    for all of them — just changing base_url and api_key.
    This is why provider abstraction is so powerful here.

    Args:
        provider_key: Key from PROVIDERS dict

    Returns:
        Initialized OpenAI client or None if key missing
    """
    provider = PROVIDERS.get(provider_key)
    if not provider:
        return None

    api_key = os.getenv(provider["env_key"])
    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url=provider["base_url"]
    )


def initialize_model() -> dict:
    """
    Initialize clients for all configured providers.

    Instead of returning a single client, we now return a dict
    of provider_key -> client for all providers that have
    API keys configured.

    Returns:
        Dict of provider_key -> OpenAI client
    """
    clients = {}

    for provider_key, provider in PROVIDERS.items():
        client = get_provider_client(provider_key)
        if client:
            clients[provider_key] = client
            print(f"✓ {provider['name']} client initialized")
        else:
            print(f"○ {provider['name']} — no API key configured (skipping)")

    if not clients:
        raise ValueError(
            "No API keys found. Add at least one provider key to your .env file.\n"
            "See .env.example for available providers."
        )

    return clients


def get_available_models(clients: dict) -> list[dict]:
    """
    Return models available based on which provider clients
    are initialized. Models whose provider has no API key
    are excluded from the list.

    Args:
        clients: Dict of provider_key -> client from initialize_model()

    Returns:
        List of available model info dicts for the frontend dropdown
    """
    available = []

    for key, model in AVAILABLE_MODELS.items():
        if model["provider"] in clients:
            available.append({
                "key":         key,
                "name":        model["name"],
                "provider":    PROVIDERS[model["provider"]]["name"],
                "description": model["description"],
            })

    return available


def stream_response(clients: dict, messages: list[dict], model_key: str = DEFAULT_MODEL):
    """
    Send a conversation to the chosen model and stream the response.

    Automatically selects the right provider client based on
    the model's provider configuration.

    Args:
        clients:   Dict of provider_key -> client
        messages:  Full conversation history
        model_key: Key from AVAILABLE_MODELS

    Yields:
        str: Text chunks as they stream from the API
    """
    model_config = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS[DEFAULT_MODEL])
    provider_key = model_config["provider"]
    client = clients.get(provider_key)

    if not client:
        yield f"[Error: {PROVIDERS[provider_key]['name']} API key not configured]"
        return

    # Build richer query context for RAG retrieval
    recent_context = " ".join([m["content"] for m in messages[-4:]])
    latest_query = recent_context if recent_context else ""

    full_messages = [
        {"role": "system", "content": build_system_prompt(query=latest_query)}
    ] + messages

    # OpenRouter free tier has credit limits — use smaller max_tokens
    max_tokens = 1024 if model_config["provider"] == "openrouter" else 2048

    try:
        response = client.chat.completions.create(
            model=model_config["id"],
            messages=full_messages,
            stream=True,
            temperature=model_config["temperature"],
            max_tokens=max_tokens,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    except Exception as e:
        yield f"\n[Error: {str(e)}]"


def refresh_system_prompt(clients: dict) -> dict:
    """
    System prompt is rebuilt fresh on every request.
    Returns the same clients dict unchanged.
    """
    print("✓ System prompt will refresh on next request")
    return clients
