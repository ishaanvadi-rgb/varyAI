"""
prompt_builder.py — Where memory becomes context in varyAI

This module constructs the system prompt that gets sent to the LLM
before every conversation. It takes the user's profile from
profile_store.py and weaves it into a prompt that makes the LLM
feel like it already knows the user.

AI Engineering Concept — What is a system prompt?
Every LLM API call has three possible roles:
- system:    Instructions to the LLM about how to behave. The user never
             sees this. It sets the context, personality, and constraints.
- user:      The actual message from the user.
- assistant: The LLM's response.

The system prompt is our most powerful tool. By injecting the user's
profile here, the LLM reads it as foundational context before it even
sees the user's first message. This is why it feels like the LLM
"already knows" the user — because from the LLM's perspective, it does.

AI Engineering Concept — Prompt structure matters:
The order and structure of information in a prompt significantly affects
LLM behavior. We put the most important context (who the user is) first,
followed by behavioral instructions, followed by the profile data.
This mirrors how you'd brief a human assistant — identity first,
then instructions, then specifics.
"""

from backend.profile_store import get_profile_summary


# The base personality and behavior of varyAI
# This is what makes it feel like a coherent product rather than
# a raw API call. Every LLM product you've used has something like this
# running silently before your first message.
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
- If the user mentions something new about themselves, incorporate it naturally
- If profile information seems outdated based on what the user says, trust what they say now
- Never say things like "according to your profile" or "I remember that you..." — 
  just use the information naturally as any knowledgeable friend would
- If the profile is empty, behave normally — don't mention that you have no profile data
"""


def build_system_prompt() -> str:
    """
    Construct the full system prompt by combining:
    1. varyAI's base identity and personality
    2. Instructions for using the profile
    3. The user's actual profile data

    This is called fresh before every conversation starts,
    ensuring the LLM always has the most up-to-date profile.

    Returns:
        A complete system prompt string ready to send to the LLM API.

    AI Engineering Concept — Why rebuild every time?
    We rebuild the system prompt before every conversation rather than
    caching it because the profile updates in real-time. If we cached it,
    facts extracted from the current conversation wouldn't be available
    until the next session. Fresh prompt = always current memory.
    """

    # Get the formatted profile string from profile_store
    profile_summary = get_profile_summary()

    # Assemble the full system prompt
    # The structure is deliberate:
    # 1. Who varyAI is (identity)
    # 2. How to use the profile (instructions)  
    # 3. What the profile contains (data)
    system_prompt = f"""
{BASE_IDENTITY}

{PROFILE_INSTRUCTIONS}

---
WHAT YOU KNOW ABOUT THIS USER:
{profile_summary}
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

    AI Engineering Concept — Why maintain full history?
    LLMs are stateless — they have no memory between API calls.
    The only way to maintain conversation context is to send the
    ENTIRE conversation history with every single API call.
    This is why long conversations become expensive — every message
    sends all previous messages again. This is also why context
    windows (maximum tokens per call) matter so much in production.
    """

    # Start with existing history
    messages = list(history)

    # Append the new user message
    messages.append({
        "role": "user",
        "content": new_message
    })

    return messages