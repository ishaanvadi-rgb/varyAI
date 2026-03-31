"""
extraction.py — The AI learning engine of varyAI

This module is responsible for extracting facts about the user
from conversations in real-time. After every message exchange,
this runs silently in the background and updates the local profile.

AI Engineering Concept — Why a separate LLM call for extraction?
We could try to make the main LLM extract facts AND respond to the user
in one call, but that creates messy, unreliable outputs. Instead, we use
a dedicated extraction call with a tightly controlled prompt that returns
clean structured JSON. This is called "separation of concerns" — one LLM
call does one job well.

AI Engineering Concept — Structured outputs:
We instruct the LLM to respond ONLY in JSON format. This is a core
pattern in AI engineering — when you need reliable, parseable output
from an LLM, you constrain its response format strictly in the prompt.
In production systems this is handled by tools like Pydantic + instructor,
but for v1 we do it manually to understand the fundamentals.
"""

import json
from backend.profile_store import save_facts


# This is the prompt that drives the extraction engine.
# It's the most important piece of prompt engineering in the entire project.
#
# Notice what makes it effective:
# 1. Very specific role — "expert at identifying personal information"
# 2. Clear categories with definitions
# 3. Explicit JSON schema with an example
# 4. Clear rules about what NOT to extract
# 5. Hard instruction: return empty list if nothing found
#
# AI Engineering Concept — Prompt engineering for structured output:
# The more specific and constrained your prompt, the more reliable the output.
# Vague prompts = unpredictable outputs. This is especially true when you
# need the LLM to return structured data like JSON.

EXTRACTION_PROMPT = """
You are a precise fact extractor. Your job is to extract ONLY explicitly stated facts about the user from conversations.

STRICT RULES:
- ONLY extract facts the user directly stated about themselves
- NEVER infer, assume, or extrapolate beyond what was explicitly said
- NEVER extract facts about general topics or the AI's responses
- If the user says "I am a student", extract "Is a student" — nothing more
- If unsure whether something was explicitly stated, skip it

Categories:
- preferences: Explicitly stated likes, dislikes, tools, habits
- projects: Explicitly mentioned current work or plans
- skills: Explicitly stated abilities or knowledge areas
- history: Explicitly stated background facts about themselves

Return ONLY this JSON, no explanation, no markdown:
{
    "facts": [
        {"category": "history", "content": "Second year chemical engineering student at IIT Roorkee"}
    ]
}

If nothing was explicitly stated about the user, return: {"facts": []}

CONVERSATION:
{conversation}
"""


def format_conversation(messages: list[dict]) -> str:
    """
    Convert the conversation history into a readable string
    that can be inserted into the extraction prompt.

    Input format (standard chat format used by all LLM APIs):
    [
        {"role": "user", "content": "I'm building a project in Python"},
        {"role": "assistant", "content": "That sounds great! What are you building?"},
        {"role": "user", "content": "An AI memory system called varyAI"}
    ]

    Output format (readable for the extraction LLM):
    User: I'm building a project in Python
    Assistant: That sounds great! What are you building?
    User: An AI memory system called varyAI
    """
    lines = []
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_and_save(messages: list[dict], clients) -> None:
    """
    Extract facts from conversation using Groq as the extraction engine.
    Falls back to first available client if Groq isn't configured.

    Args:
        messages: Full conversation history in standard chat format
        clients:  Dict of provider_key -> client, or a single client instance
    """
    # Use Groq for extraction — fast and reliable for structured output
    # Fall back to first available provider if Groq not configured
    if isinstance(clients, dict):
        client = clients.get("groq") or next(iter(clients.values()))
    else:
        client = clients

    conversation_text = format_conversation(messages)
    prompt = EXTRACTION_PROMPT.replace("{conversation}", conversation_text)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )

        raw_output = response.choices[0].message.content.strip()

        # Robustly extract JSON from the response
        # Llama sometimes wraps output in ```json ... ``` or adds explanation text
        # This finds the JSON object regardless of what surrounds it
        import re
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if not json_match:
            print("⚠ No JSON found in extraction response")
            return

        clean_json = json_match.group(0).strip()
        extracted = json.loads(clean_json)

        if "facts" in extracted and isinstance(extracted["facts"], list):
            facts = extracted["facts"]
            if facts:
                save_facts(facts)
                print(f"✓ Extracted and saved {len(facts)} fact(s)")
            else:
                print("✓ Extraction complete — no new facts found")
        else:
            print("⚠ Extraction returned unexpected format, skipping")

    except json.JSONDecodeError as e:
        print(f"⚠ Extraction failed — invalid JSON: {e}")

    except Exception as e:
        print(f"⚠ Extraction error: {e}")