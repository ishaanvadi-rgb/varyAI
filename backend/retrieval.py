"""
retrieval.py — The semantic retrieval layer for varyAI (v3)

This module replaces the v1 "inject everything" approach with
intelligent retrieval — fetching only the profile facts that are
semantically relevant to what the user is currently talking about.

AI Engineering Concept — What is RAG?
RAG stands for Retrieval Augmented Generation. The pattern is:
1. RETRIEVE: Find relevant context from a knowledge store
2. AUGMENT:  Inject that context into the prompt
3. GENERATE: Let the LLM generate a response with that context

In our case:
- Knowledge store = user's profile (SQLite + ChromaDB)
- Retrieval = semantic search using embeddings
- Augmentation = injecting relevant facts into system prompt
- Generation = Llama responding with personalized context

AI Engineering Concept — What are embeddings?
An embedding is a list of numbers (a vector) that represents the
meaning of a piece of text. Texts with similar meanings have similar
vectors — their numbers are close to each other in high-dimensional
space. This is what makes semantic search possible: instead of
matching keywords, we match meanings.

Example:
"I love Python programming"  → [0.2, 0.8, 0.1, 0.9, 0.3, ...]
"My favourite language is Python" → [0.3, 0.7, 0.2, 0.8, 0.4, ...]
"I enjoy cooking pasta"      → [0.9, 0.1, 0.8, 0.1, 0.7, ...]

The first two are close in vector space (similar meaning).
The third is far away (different meaning).

AI Engineering Concept — Why ChromaDB?
ChromaDB is a vector database — a database optimised for storing
and searching embeddings. It can find the N most similar vectors
to a query vector extremely efficiently, even across millions of
entries. We use it to store one embedding per profile fact.

AI Engineering Concept — Why local embeddings?
We use sentence-transformers to generate embeddings locally on
your machine instead of calling an API. This means:
- No cost per embedding
- No data sent to external servers for this step
- Works offline
- Fast enough for our use case
The model (all-MiniLM-L6-v2) is small (80MB) but produces
high quality embeddings for semantic similarity tasks.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────

# Path where ChromaDB stores its data — local, alongside SQLite
CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"

# The embedding model — runs locally, no API needed
# all-MiniLM-L6-v2 is a well-regarded small model that produces
# 384-dimensional embeddings. Good balance of speed and quality.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# How many relevant facts to retrieve per query
# 5-8 is the sweet spot — enough context without noise
TOP_K = 8

# Module-level singletons — initialised once, reused across calls
# Creating these objects is expensive, so we do it once at startup
_embedding_model = None
_chroma_client = None
_collection = None


def initialize_retrieval() -> None:
    """
    Initialize the embedding model and ChromaDB collection.
    Called once at server startup alongside initialize_database().

    AI Engineering Concept — Lazy vs eager initialisation:
    We initialise everything eagerly at startup rather than
    lazily on first use. This means the first user message
    isn't slow due to model loading — the cost is paid once
    at startup where it's expected and acceptable.
    """
    global _embedding_model, _chroma_client, _collection

    print("⏳ Loading embedding model (first run may take a moment)...")

    # Load the local embedding model
    # On first run this downloads ~80MB, after that it's cached
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialize ChromaDB with persistent storage
    # persist_directory means data survives server restarts
    _chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
    )

    # Get or create our facts collection
    # A collection is like a table — stores embeddings + metadata
    _collection = _chroma_client.get_or_create_collection(
        name="profile_facts",
        # Cosine similarity is the standard metric for semantic search
        # It measures the angle between vectors (0 = identical, 2 = opposite)
        metadata={"hnsw:space": "cosine"}
    )

    print(f"✓ Retrieval layer initialized ({_collection.count()} facts indexed)")


def embed_text(text: str) -> list[float]:
    """
    Convert a piece of text into an embedding vector.

    Args:
        text: Any string to embed

    Returns:
        A list of floats representing the text's meaning
        (384 dimensions for all-MiniLM-L6-v2)

    AI Engineering Concept — Normalisation:
    We normalise embeddings to unit length before storing them.
    This makes cosine similarity equivalent to dot product similarity,
    which ChromaDB can compute more efficiently.
    """
    embedding = _embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def index_fact(fact_id: int, content: str, category: str) -> None:
    """
    Add a new profile fact to the ChromaDB vector index.

    Called by profile_store.save_facts() every time a new fact
    is saved — so the vector index stays in sync with SQLite.

    Args:
        fact_id:  The SQLite row ID of the fact (used as ChromaDB ID)
        content:  The text content of the fact
        category: The category (preferences, projects, skills, history)

    AI Engineering Concept — Dual write:
    We write to both SQLite AND ChromaDB for every fact.
    SQLite stores the structured data (easy to read, query, edit).
    ChromaDB stores the embedding (enables semantic search).
    This duplication is intentional — each database does what
    it's optimised for. SQLite for structure, ChromaDB for search.
    """
    embedding = embed_text(content)

    _collection.add(
        ids=[str(fact_id)],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"category": category}]
    )


def retrieve_relevant_facts(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Find the most semantically relevant profile facts for a given query.

    This is the core of RAG — given what the user just said,
    find which profile facts are most relevant to inject.

    Args:
        query:  The user's current message (or recent conversation)
        top_k:  Maximum number of facts to retrieve

    Returns:
        List of dicts with 'content' and 'category' keys,
        ordered by relevance (most relevant first)

    AI Engineering Concept — Semantic search process:
    1. Embed the query into a vector
    2. Ask ChromaDB to find the N vectors closest to the query vector
    3. Return the corresponding facts

    The "closeness" is measured by cosine similarity:
    - Score of 1.0 = identical meaning
    - Score of 0.0 = completely unrelated
    We filter out facts below a minimum similarity threshold
    to avoid injecting irrelevant context.

    AI Engineering Concept — Query construction:
    We embed the user's message as-is. In more advanced systems
    you might expand the query — adding context from recent
    conversation turns to improve retrieval quality. This is
    called HyDE (Hypothetical Document Embeddings) or query
    expansion. We keep it simple for v3.
    """
    if _collection is None:
        return []

    # If profile is empty, return nothing
    if _collection.count() == 0:
        return []

    # Embed the user's query
    query_embedding = embed_text(query)

    # Search ChromaDB for the most similar facts
    # n_results is capped at collection size to avoid errors
    n_results = min(top_k, _collection.count())

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Parse and filter results
    # ChromaDB returns distances (lower = more similar for cosine)
    # We convert to similarity score: similarity = 1 - distance
    relevant_facts = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        similarity = 1 - distance

        # Only include facts above minimum similarity threshold
        # 0.25 is a reasonable threshold — filters noise without
        # being too aggressive on a small profile
        if similarity >= 0.15:
            relevant_facts.append({
                "content": doc,
                "category": meta["category"],
                "similarity": round(similarity, 3)  # useful for debugging
            })

    return relevant_facts


def get_relevant_profile_summary(query: str) -> str:
    """
    Retrieve relevant facts and format them as a string
    ready for injection into the system prompt.

    This replaces get_profile_summary() from profile_store.py
    for the RAG-powered flow.

    Args:
        query: The user's current message

    Returns:
        Formatted string of relevant facts, or a note if
        the profile is empty.

    AI Engineering Concept — Context window efficiency:
    In v1 we might inject 500 tokens of profile context per message.
    With RAG we inject ~50-100 tokens of highly relevant context.
    Over 1000 messages, this saves ~450,000 tokens — significant
    cost savings at scale, and better response quality too because
    the LLM isn't distracted by irrelevant context.
    """
    facts = retrieve_relevant_facts(query)

    if not facts:
        return "No relevant profile data found for this query."

    # Group by category for clean formatting
    grouped = {}
    for fact in facts:
        cat = fact["category"].upper()
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(fact["content"])

    lines = ["[Relevant profile context for this message:]"]
    for category, contents in grouped.items():
        lines.append(f"\n{category}:")
        for content in contents:
            lines.append(f"  - {content}")

    return "\n".join(lines)


def sync_existing_facts() -> None:
    """
    Index any facts that are in SQLite but not yet in ChromaDB.

    Called at startup to handle the case where facts were saved
    before the vector index existed (e.g. upgrading from v1 to v3).

    AI Engineering Concept — Index synchronisation:
    In production systems, keeping multiple data stores in sync
    is a real engineering challenge. Our approach is simple:
    on startup, check what's in SQLite and index anything missing
    from ChromaDB. For a single-user local app this is sufficient.
    """
    from backend.profile_store import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, content, category FROM profile"
        ).fetchall()

        # Get already-indexed IDs from ChromaDB
        if _collection.count() > 0:
            existing = _collection.get()
            indexed_ids = set(existing["ids"])
        else:
            indexed_ids = set()

        # Index any facts not yet in ChromaDB
        new_count = 0
        for row in rows:
            fact_id = str(row["id"])
            if fact_id not in indexed_ids:
                index_fact(row["id"], row["content"], row["category"])
                new_count += 1

        if new_count > 0:
            print(f"✓ Indexed {new_count} existing fact(s) into vector store")

    finally:
        conn.close()


def find_conflicting_fact(content: str, category: str, threshold: float = 0.75) -> str | None:
    """
    Check if a semantically similar fact already exists in the profile.

    If similarity is above threshold, we consider it a conflict —
    the new fact is an update to the existing one, not a new fact.

    Args:
        content:   The new fact being considered for saving
        category:  The category to search within
        threshold: Similarity score above which we consider it a conflict
                   0.75 is high enough to catch genuine conflicts
                   without false positives on unrelated facts

    Returns:
        The ChromaDB document ID of the conflicting fact, or None

    AI Engineering Concept — Semantic deduplication:
    Simple string matching catches exact duplicates ("Has Python experience"
    == "Has Python experience"). But it misses paraphrases like
    "Knows Python well" vs "Has experience with Python" — these mean
    the same thing but are textually different.
    Embedding similarity catches both cases because similar meanings
    produce similar vectors, regardless of exact wording.
    """
    if _collection is None or _collection.count() == 0:
        return None

    query_embedding = embed_text(content)

    # Search only within the same category
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, _collection.count()),
        where={"category": category},
        include=["documents", "metadatas", "distances"]
    )

    if not results["ids"][0]:
        return None

    for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
        similarity = 1 - distance
        if similarity >= threshold:
            return doc_id  # Return ID of conflicting fact

    return None


def delete_fact_from_index(fact_id: int) -> None:
    """
    Remove a fact from the ChromaDB vector index.

    Args:
        fact_id: The SQLite row ID (used as ChromaDB document ID)
    """
    if _collection is None:
        return
    try:
        _collection.delete(ids=[str(fact_id)])
        print(f"✓ Removed fact {fact_id} from vector index")
    except Exception as e:
        print(f"⚠ Could not remove fact from index: {e}")