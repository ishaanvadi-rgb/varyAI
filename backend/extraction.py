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
You are an expert at identifying personal information about a user from conversations.
Your job is to extract facts that would help an AI assistant know the user better in future conversations.

Analyze the conversation below and extract facts about the user into these categories:

- preferences: How they like to work, communicate, tools they use, things they like/dislike
- projects: What they are currently building, working on, or planning
- skills: Their technical abilities, knowledge areas, experience levels
- history: Decisions they've made, things they've mentioned that give context about them

Return ONLY a JSON object in this exact format, nothing else, no explanation, no markdown:
{
    "facts": [
        {"category": "skills", "content": "Knows Python well but is relearning syntax"},
        {"category": "projects", "content": "Building varyAI, a local-first AI memory system"}
    ]
}

Rules:
- Only extract facts ABOUT THE USER, not general information
- Only extract meaningful, reusable facts — not conversation-specific details
- If the user corrects something, extract the correction not the original
- If nothing meaningful can be extracted, return: {"facts": []}
- Keep each fact concise — one clear statement per fact
- Never invent facts — only extract what was explicitly said or clearly implied

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


def extract_and_save(messages: list[dict], client) -> None:
    """
    The main extraction function. Takes the full conversation history,
    asks Llama via Groq to extract facts, and saves them to the profile.

    Args:
        messages: Full conversation history in standard chat format
        client:   The initialized Groq client instance from llm_client.py
    """

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