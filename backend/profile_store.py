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
    Create all tables if they don't already exist.
    Called once when the backend starts up.

    Tables:
    - profile:       Individual facts about the user
    - conversations: Metadata about each conversation session
    - messages:      Individual messages within each conversation
    """
    conn = get_connection()

    try:
        # User profile facts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profile (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            )
        """)

        # Conversation sessions
        # Each session is one continuous conversation with a title
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT DEFAULT 'New Conversation',
                model_key  TEXT DEFAULT 'llama-3.3-70b',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Individual messages within a conversation
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
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
    Save extracted facts to the profile with conflict detection.

    Before saving each fact, checks if a semantically similar fact
    already exists in the same category. If it does, the old fact
    is replaced with the new one — preventing contradictory facts
    from building up over time.

    AI Engineering Concept — Why replace rather than keep both?
    If the user said "Lives in Delhi" in January and "Moved to Mumbai"
    in March, keeping both creates a contradiction. The newer fact
    is always more accurate, so we replace. In a more sophisticated
    system, we'd also store the old fact with a timestamp for audit
    purposes — but for v1 replacement is clean and correct.
    """
    if not facts:
        return

    conn = get_connection()

    try:
        saved_count = 0
        replaced_count = 0

        for fact in facts:
            if "category" not in fact or "content" not in fact:
                continue
            if fact["category"] not in ["preferences", "projects", "skills", "history"]:
                continue

            # Step 1 — Exact duplicate check
            existing_exact = conn.execute("""
                SELECT id FROM profile
                WHERE category = ?
                AND LOWER(content) = LOWER(?)
            """, (fact["category"], fact["content"])).fetchone()

            if existing_exact:
                continue  # Exact duplicate, skip silently

            # Step 2 — Semantic conflict check
            # Check if a similar fact exists using embedding similarity
            try:
                from backend.retrieval import find_conflicting_fact
                conflicting_id = find_conflicting_fact(
                    fact["content"],
                    fact["category"]
                )
            except Exception:
                conflicting_id = None

            if conflicting_id:
                # Replace the conflicting fact with the new one
                # conflicting_id is the ChromaDB ID which equals the SQLite row ID
                old_id = int(conflicting_id)

                conn.execute("""
                    UPDATE profile
                    SET content = ?, updated_at = datetime('now')
                    WHERE id = ?
                """, (fact["content"], old_id))

                # Update the vector index with new embedding
                try:
                    from backend.retrieval import index_fact, delete_fact_from_index
                    delete_fact_from_index(old_id)
                    index_fact(old_id, fact["content"], fact["category"])
                except Exception as e:
                    print(f"⚠ Could not update vector index: {e}")

                replaced_count += 1
                print(f"✓ Replaced conflicting fact: '{fact['content']}'")

            else:
                # No conflict — insert as new fact
                cursor = conn.execute("""
                    INSERT INTO profile (category, content, created_at, updated_at)
                    VALUES (?, ?, datetime('now'), datetime('now'))
                """, (fact["category"], fact["content"]))

                try:
                    from backend.retrieval import index_fact
                    index_fact(cursor.lastrowid, fact["content"], fact["category"])
                except Exception as e:
                    print(f"⚠ Could not index fact in vector store: {e}")

                saved_count += 1

        conn.commit()

        if saved_count > 0:
            print(f"✓ Saved {saved_count} new fact(s) to profile")
        if replaced_count > 0:
            print(f"✓ Replaced {replaced_count} conflicting fact(s)")
        if saved_count == 0 and replaced_count == 0:
            print("✓ Extraction complete — no new facts found")

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


def delete_fact(fact_id: int) -> None:
    """
    Delete a specific fact from SQLite and ChromaDB.

    Args:
        fact_id: The SQLite row ID of the fact to delete
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM profile WHERE id = ?", (fact_id,))
        conn.commit()

        # Also remove from vector index
        try:
            from backend.retrieval import delete_fact_from_index
            delete_fact_from_index(fact_id)
        except Exception as e:
            print(f"⚠ Could not remove fact from vector index: {e}")

        print(f"✓ Deleted fact {fact_id}")
    finally:
        conn.close()


def get_full_profile_with_ids() -> dict:
    """
    Like get_full_profile() but includes the SQLite row ID for each fact.
    Needed by the profile editor so it knows which fact to delete.

    Returns:
        Dict of category -> list of {id, content} dicts
    """
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT id, category, content
            FROM profile
            ORDER BY category, created_at ASC
        """).fetchall()

        profile = {
            "preferences": [],
            "projects": [],
            "skills": [],
            "history": []
        }

        for row in rows:
            category = row["category"]
            if category in profile:
                profile[category].append({
                    "id": row["id"],
                    "content": row["content"]
                })

        return profile
    finally:
        conn.close()


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


def create_conversation(model_key: str = "llama-3.3-70b") -> int:
    """
    Create a new conversation session and return its ID.

    Called when the user starts a new chat. The title starts as
    'New Conversation' and gets updated after the first message
    using the user's first message as the title.

    Args:
        model_key: The LLM model being used for this conversation

    Returns:
        The ID of the newly created conversation
    """
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO conversations (title, model_key, created_at, updated_at)
            VALUES ('New Conversation', ?, datetime('now'), datetime('now'))
        """, (model_key,))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def save_message(conversation_id: int, role: str, content: str) -> None:
    """
    Save a single message to a conversation.

    Called after every user message and assistant response
    to persist the full conversation history.

    Args:
        conversation_id: The ID of the conversation this message belongs to
        role:            'user' or 'assistant'
        content:         The message text
    """
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (conversation_id, role, content))

        # Update conversation's updated_at timestamp
        conn.execute("""
            UPDATE conversations
            SET updated_at = datetime('now')
            WHERE id = ?
        """, (conversation_id,))

        conn.commit()
    finally:
        conn.close()


def update_conversation_title(conversation_id: int, title: str) -> None:
    """
    Update the conversation title based on the first user message.

    We truncate to 50 characters to keep the sidebar clean.

    Args:
        conversation_id: The conversation to update
        title:           The new title (first user message, truncated)
    """
    conn = get_connection()
    try:
        # Truncate and clean the title
        clean_title = title.strip()[:50]
        if len(title.strip()) > 50:
            clean_title += "..."

        conn.execute("""
            UPDATE conversations
            SET title = ?, updated_at = datetime('now')
            WHERE id = ?
        """, (clean_title, conversation_id))
        conn.commit()
    finally:
        conn.close()


def get_conversation_messages(conversation_id: int) -> list[dict]:
    """
    Retrieve all messages for a conversation in chronological order.

    Used to restore conversation history when the user resumes
    a past conversation.

    Args:
        conversation_id: The conversation to retrieve

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
        """, (conversation_id,)).fetchall()

        return [{"role": row["role"], "content": row["content"]} for row in rows]
    finally:
        conn.close()


def get_all_conversations() -> list[dict]:
    """
    Retrieve all conversations ordered by most recent first.

    Used to populate the conversation history panel in the sidebar.

    Returns:
        List of conversation dicts with id, title, model_key, updated_at
    """
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT id, title, model_key, updated_at
            FROM conversations
            ORDER BY updated_at DESC
        """).fetchall()

        return [
            {
                "id":         row["id"],
                "title":      row["title"],
                "model_key":  row["model_key"],
                "updated_at": row["updated_at"]
            }
            for row in rows
        ]
    finally:
        conn.close()


def delete_conversation(conversation_id: int) -> None:
    """
    Delete a conversation and all its messages.

    Args:
        conversation_id: The conversation to delete
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()