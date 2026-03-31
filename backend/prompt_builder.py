"""
prompt_builder.py — Where memory becomes context in varyAI

This module constructs the system prompt that gets sent to the LLM
before every conversation. It takes the user's profile from
profile_store.py and weaves it into a prompt that makes the LLM
feel like it already knows the user.
"""

from backend.profile_store import get_profile_summary


# The base personality and behavior of varyAI
BASE_IDENTITY = """You are varyAI, a personal AI assistant with a unique ability — \
you have access to a memory profile about the user you're talking to. \
This profile has been built from your past conversations with them. \
Use it naturally, the way a good friend would — don't announce that you're \
using it, don't reference it explicitly, just let it inform how you respond.

Your personality:
- Warm but direct — no unnecessary filler phrases
- Technically precise when the user needs depth
- Adaptive — match the user's level of expertise and communication style
- Proactive — if you notice something relevant in the profile, use it
"""

# Instructions for how to handle the profile
PROFILE_INSTRUCTIONS = """
Guidelines for using the user's profile:
- Use the profile to personalize your responses naturally
- NEVER make assumptions about the user beyond what the profile states
- NEVER infer unstated details like location, interests, or background
- If the user mentions something new about themselves, incorporate it naturally
- If profile information seems outdated based on what the user says, trust what they say now
- Never say things like "according to your profile" or "I remember that you..." —
  just use the information naturally as any knowledgeable friend would
- If the profile is empty, behave normally — don't mention that you have no profile data
"""


def build_system_prompt(query: str = "") -> str:
    """
    Construct the full system prompt with RAG-powered profile retrieval.

    If a query is provided, retrieves only semantically relevant facts.
    If no query (e.g. at initialisation), falls back to full profile.

    Args:
        query: The user's current message for semantic retrieval

    Returns:
        Complete system prompt string ready to send to the LLM API.
    """
    if query:
        # RAG — retrieve only relevant facts for this specific query
        try:
            from backend.retrieval import get_relevant_profile_summary
            profile_context = get_relevant_profile_summary(query)
        except Exception:
            # Fall back to full profile if retrieval fails
            profile_context = get_profile_summary()
    else:
        # No query available — use full profile (fallback)
        profile_context = get_profile_summary()

    system_prompt = f"""
{BASE_IDENTITY}

{PROFILE_INSTRUCTIONS}

---
WHAT YOU KNOW ABOUT THIS USER:
{profile_context}
---
"""
    return system_prompt.strip()


def build_conversation_messages(history: list[dict], new_message: str) -> list[dict]:
    """
    Build the full messages array that gets sent to the LLM API.

    Every LLM API expects messages in this standard format:
    [
        {{"role": "user", "content": "Hello"}},
        {{"role": "assistant", "content": "Hi! How can I help?"}},
        {{"role": "user", "content": "Tell me about Python"}}
    ]

    We take the existing conversation history and append the new
    user message to it, maintaining the full context of the session.

    Args:
        history:     Previous messages in this conversation session
        new_message: The user's latest message

    Returns:
        Complete messages array ready to send to the LLM API
    """

    # Start with existing history
    messages = list(history)

    # Append the new user message
    messages.append({
        "role": "user",
        "content": new_message
    })

    return messages