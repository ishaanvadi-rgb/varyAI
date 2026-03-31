"""
profile_store.py — The memory foundation of varyAI

This module handles all interactions with the local SQLite database.
SQLite is a file-based database — no server needed, no setup, just a .db file
sitting on your machine. This is what makes varyAI truly local-first.

The profile is organized into 4 categories:
- preferences: how you like to work, communicate, what tools you use
- projects: what you're currently building or working on
- history: past decisions, things you've mentioned before
- skills: your expertise level in different areas

AI Engineering Concept — Why SQLite for v1?
In v1 we inject the full profile into every prompt (simple but works).
In v3 we'll swap this for a vector database (ChromaDB) that lets us
retrieve only the RELEVANT facts using semantic search — that's RAG.
For now, SQLite keeps things simple and learnable.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

# The database file lives at the project root
# Path(__file__) is this file's location, .parent goes up to backend/, 
# .parent again goes to project root
DB_PATH = Path(__file__).parent.parent / "varyai.db"


def get_connection() -> sqlite3.Connection:
    """
    Create and return a connection to the SQLite database.
    
    sqlite3.connect() creates the file if it doesn't exist yet.
    detect_types enables Python to parse stored dates back into datetime objects.
    """
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    # row_factory makes rows behave like dictionaries (access by column name)
    # instead of tuples (access by index). Much more readable.
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """
    Create the profile table if it doesn't already exist.
    Called once when the backend starts up.
    
    Each row in the table is one fact about the user, for example:
    - category: "skills", content: "Knows Python, learning AI engineering"
    - category: "projects", content: "Building varyAI, a local-first AI memory system"
    - category: "preferences", content: "Prefers detailed explanations before code"
    """
    conn = get_connection()
    
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profile (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        
        # This table tracks every conversation session
        # Useful for context and future features
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                llm         TEXT NOT NULL,
                started_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        
        conn.commit()
        print("✓ Database initialized at", DB_PATH)
        
    finally:
        conn.close()


def get_full_profile() -> dict:
    """
Retrieve all profile facts, grouped by category.
Returns a dictionary which get_profile_summary() then converts
into a formatted string for prompt injection.

In v3, we'll replace this with get_relevant_profile(query)
which fetches only semantically relevant facts — that's RAG.
"""

    conn = get_connection()
    
    try:
        rows = conn.execute("""
            SELECT category, content 
            FROM profile 
            ORDER BY category, created_at ASC
        """).fetchall()
        
        # Group facts by category
        profile = {
            "preferences": [],
            "projects": [],
            "skills": [],
            "history": []
        }
        
        for row in rows:
            category = row["category"]
            if category in profile:
                profile[category].append(row["content"])
        
        return profile
    
    finally:
        conn.close()


def save_facts(facts: list[dict]) -> None:
    """
    Save a list of newly extracted facts to the profile.
    
    Called by extraction.py after every conversation exchange.
    Each fact is a dict with 'category' and 'content' keys:
    
    Example input:
    [
        {"category": "skills", "content": "Learning React for the first time"},
        {"category": "projects", "content": "Building varyAI with FastAPI backend"}
    ]
    
    AI Engineering Concept — Deduplication:
    In v1 we don't deduplicate, so similar facts might get stored multiple times.
    In a future version, before saving we'd embed the new fact and check if a
    semantically similar fact already exists in the database. If similarity
    score > threshold, we update instead of insert.
    """
    if not facts:
        return
    
    conn = get_connection()
    
    try:
        for fact in facts:
            # Basic validation — skip malformed facts
            if "category" not in fact or "content" not in fact:
                continue
            if fact["category"] not in ["preferences", "projects", "skills", "history"]:
                continue
                
            conn.execute("""
                INSERT INTO profile (category, content, created_at, updated_at)
                VALUES (?, ?, datetime('now'), datetime('now'))
            """, (fact["category"], fact["content"]))
        
        conn.commit()
        print(f"✓ Saved {len(facts)} new fact(s) to profile")
        
    finally:
        conn.close()


def get_profile_summary() -> str:
    """
    Return the profile as a formatted string, ready to be injected
    into a system prompt.
    
    Example output:
    
    PREFERENCES:
    - Prefers detailed explanations before seeing code
    - Uses VS Code as primary editor
    
    PROJECTS:
    - Building varyAI, a local-first AI memory layer
    
    SKILLS:
    - Knows Python well, learning AI engineering
    - New to React and frontend development
    
    HISTORY:
    - Decided to use Gemini API for v1
    - Chose Poetry over pip for dependency management
    """
    profile = get_full_profile()
    
    # If profile is completely empty, return a note
    # This happens on first ever conversation
    if not any(profile.values()):
        return "No profile data yet. This is the user's first conversation."
    
    lines = []
    
    for category, facts in profile.items():
        if facts:  # Only include categories that have data
            lines.append(f"\n{category.upper()}:")
            for fact in facts:
                lines.append(f"  - {fact}")
    
    return "\n".join(lines)


def clear_profile() -> None:
    """
    Delete all profile data. Useful for testing and resetting.
    In the final UI, this will be a button the user can press.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM profile")
        conn.commit()
        print("✓ Profile cleared")
    finally:
        conn.close()