"""
importer.py — External conversation importer for varyAI

This module handles importing conversations from other AI platforms
by fetching and parsing their public share links.

Currently supported:
- ChatGPT (chatgpt.com/share/...)

AI Engineering Concept — Why scraping and not an API?
OpenAI, Anthropic, and Google don't provide public APIs to fetch
conversation history from share links. The share page renders as
public HTML — scraping is the only programmatic way to access it.

Important limitations:
- This relies on the HTML structure of each platform's share page
- If the platform changes their HTML, the parser breaks
- This is against most platforms' ToS for automated access
- We disclose this clearly in the UI and README

Despite these limitations, this is the most direct path to solving
the real user problem: bringing context from other AI tools into varyAI.
"""

import httpx
from bs4 import BeautifulSoup
import json
import re


# ─────────────────────────────────────────────────────
# Platform detection
# ─────────────────────────────────────────────────────

def detect_platform(url: str) -> str | None:
    """
    Detect which AI platform a share link comes from.

    Args:
        url: The share link URL

    Returns:
        Platform name string or None if unrecognized
    """
    if "chatgpt.com/share" in url or "chat.openai.com/share" in url:
        return "chatgpt"
    if "claude.ai/share" in url:
        return "claude"
    return None


# ─────────────────────────────────────────────────────
# Main import function
# ─────────────────────────────────────────────────────

async def import_from_url(url: str) -> dict:
    """
    Fetch and parse a conversation from an AI platform share link.

    Args:
        url: The share link URL to import from

    Returns:
        Dict with:
        - messages: list of {role, content} dicts
        - platform: which platform it came from
        - title: conversation title if available
        - error: error message if import failed

    AI Engineering Concept — Async HTTP:
    We use httpx instead of the standard requests library because
    FastAPI is async. Using a synchronous HTTP library inside an
    async endpoint would block the event loop. httpx provides
    the same API as requests but works properly with async/await.
    """
    platform = detect_platform(url)

    if not platform:
        return {
            "error": "Unrecognized URL. Currently supported: ChatGPT share links (chatgpt.com/share/...)",
            "messages": [],
            "platform": None,
            "title": None
        }

    try:
        # Fetch the share page
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={
                # Mimic a real browser to avoid being blocked
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        ) as client:
            response = await client.get(url)

            if response.status_code == 404:
                return {
                    "error": "Link not found. The conversation may have been deleted or made private.",
                    "messages": [],
                    "platform": platform,
                    "title": None
                }

            if response.status_code != 200:
                return {
                    "error": f"Could not fetch the link (status {response.status_code}). Try again later.",
                    "messages": [],
                    "platform": platform,
                    "title": None
                }

            html = response.text

        # Parse based on platform
        if platform == "chatgpt":
            return parse_chatgpt(html, url)
        elif platform == "claude":
            return parse_claude(html, url)

    except httpx.TimeoutException:
        return {
            "error": "Request timed out. The platform may be slow or blocking requests.",
            "messages": [],
            "platform": platform,
            "title": None
        }
    except Exception as e:
        return {
            "error": f"Import failed: {str(e)}",
            "messages": [],
            "platform": platform,
            "title": None
        }


# ─────────────────────────────────────────────────────
# ChatGPT parser
# ─────────────────────────────────────────────────────

def parse_chatgpt(html: str, url: str) -> dict:
    """
    Parse a ChatGPT share page to extract the conversation.

    ChatGPT has migrated through several formats:
    - 2023–2024: __NEXT_DATA__ JSON (Next.js SSR)
    - 2025+:     React Router streamed flat array (streamController.enqueue)

    We try the newest format first, then fall back to older methods.
    """
    # Method 1: React Router flat array format (2025+)
    result = _parse_react_router_format(html)
    if result:
        return result

    soup = BeautifulSoup(html, "html.parser")

    # Method 2: __NEXT_DATA__ JSON (2023–2024)
    next_data_script = soup.find("script", id="__NEXT_DATA__")
    if next_data_script:
        try:
            data = json.loads(next_data_script.string)
            messages = extract_messages_from_next_data(data)
            if messages:
                title = extract_title_from_next_data(data)
                return {
                    "messages": messages,
                    "platform": "chatgpt",
                    "title": title or "ChatGPT Conversation",
                    "error": None
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # Method 3: HTML data attributes
    messages = extract_messages_from_html(soup)
    if messages:
        title_tag = soup.find("title")
        title = title_tag.text if title_tag else "ChatGPT Conversation"
        return {
            "messages": messages,
            "platform": "chatgpt",
            "title": title,
            "error": None
        }

    # Method 4: Plain text patterns
    messages = extract_messages_from_text(soup)
    if messages:
        return {
            "messages": messages,
            "platform": "chatgpt",
            "title": "ChatGPT Conversation",
            "error": None
        }

    return {
        "error": "Could not parse the conversation. ChatGPT may have updated their page structure.",
        "messages": [],
        "platform": "chatgpt",
        "title": None
    }


def _parse_react_router_format(html: str) -> dict | None:
    """
    Parse ChatGPT's React Router streamed flat array format (2025+).

    ChatGPT now embeds conversation data as a flat serialized array inside
    a streamController.enqueue() call. The format is:
    - A flat JSON array where every value is stored at an index
    - Object keys are "_N" where arr[N] is the actual key string
    - Object values are integers that index into the same flat array
    - List items are also integer indices into the array
    - Special negative integers: -5=None, -2=False, -1=True

    This is React's Server Components wire format, used for streaming
    page state from server to client without a separate API call.
    """
    # Find the enqueue call — the conversation is the largest one
    match = re.search(
        r'streamController\.enqueue\("((?:[^"\\]|\\.)*)"\)',
        html
    )
    if not match:
        return None

    try:
        raw = match.group(1)
        # The content is a JSON-encoded string: decode it to get the array string
        arr_str = json.loads('"' + raw + '"')
        arr = json.loads(arr_str)
    except Exception:
        return None

    if not isinstance(arr, list) or len(arr) < 20:
        return None

    # Special negative index values
    SPECIAL = {-5: None, -4: None, -3: None, -2: False, -1: True}

    def resolve(val):
        """Resolve an integer index to its value in the flat array."""
        if isinstance(val, int):
            if val < 0:
                return SPECIAL.get(val)
            return arr[val] if val < len(arr) else None
        return val  # already a direct value (string, float, dict, list)

    def find_key(s):
        """Find the index of a key name string in the flat array."""
        try:
            return arr.index(s)
        except ValueError:
            return None

    # Locate key string indices — each key name is stored once in the array
    lc_ki        = find_key("linear_conversation")
    message_ki   = find_key("message")
    author_ki    = find_key("author")
    role_ki      = find_key("role")
    content_ki   = find_key("content")
    parts_ki     = find_key("parts")
    title_ki     = find_key("title")

    if None in (lc_ki, message_ki, author_ki, role_ki, content_ki, parts_ki):
        return None

    # Find the linear_conversation list (ordered node indices)
    lc_list = None
    for item in arr:
        if isinstance(item, dict):
            v_idx = item.get(f"_{lc_ki}")
            if v_idx is not None:
                v = resolve(v_idx)
                if isinstance(v, list) and len(v) > 0:
                    lc_list = v
                    break

    if not lc_list:
        return None

    # Find the conversation title
    title = "ChatGPT Conversation"
    if title_ki is not None:
        for item in arr:
            if isinstance(item, dict):
                v_idx = item.get(f"_{title_ki}")
                if v_idx is not None:
                    t = resolve(v_idx)
                    if isinstance(t, str) and t and len(t) < 300:
                        title = t
                        break

    # Walk the linear_conversation nodes in order and extract messages
    messages = []
    mk = f"_{message_ki}"
    ak = f"_{author_ki}"
    rk = f"_{role_ki}"
    ck = f"_{content_ki}"
    pk = f"_{parts_ki}"

    for node_idx in lc_list:
        # Each item in lc_list is an index pointing to a node shell dict
        node = resolve(node_idx)
        if not isinstance(node, dict) or mk not in node:
            continue

        # Get the message object
        msg = resolve(node[mk])
        if not isinstance(msg, dict):
            continue

        # Get the author object and role
        if ak not in msg:
            continue
        author = resolve(msg[ak])
        if not isinstance(author, dict) or rk not in author:
            continue

        role = resolve(author[rk])
        if role not in ("user", "assistant"):
            continue

        # Get the content object and parts
        if ck not in msg:
            continue
        content = resolve(msg[ck])
        if not isinstance(content, dict) or pk not in content:
            continue

        parts = resolve(content[pk])
        if not isinstance(parts, list) or not parts:
            continue

        text = resolve(parts[0])
        if not isinstance(text, str) or not text.strip():
            continue

        messages.append({"role": role, "content": text.strip()})

    if not messages:
        return None

    return {
        "messages": messages,
        "platform": "chatgpt",
        "title": title,
        "error": None
    }


def extract_messages_from_next_data(data: dict) -> list[dict]:
    """
    Navigate the __NEXT_DATA__ JSON structure to find messages.

    The structure varies between ChatGPT versions but typically
    the conversation is nested under props.pageProps.
    """
    messages = []

    try:
        # Navigate to conversation data
        page_props = data.get("props", {}).get("pageProps", {})

        # Try different possible locations for the conversation
        conversation = (
            page_props.get("conversation") or
            page_props.get("serverResponse", {}).get("data", {}).get("conversation") or
            page_props.get("data", {}).get("conversation")
        )

        if not conversation:
            return []

        # Extract linear messages from the conversation tree
        mapping = conversation.get("mapping", {})
        if not mapping:
            return []

        # Find the root node and traverse the message tree
        # ChatGPT stores conversations as a tree (for branching)
        # We extract the main branch linearly
        messages = traverse_message_tree(mapping)

    except (KeyError, TypeError, AttributeError):
        pass

    return messages


def traverse_message_tree(mapping: dict) -> list[dict]:
    """
    Traverse ChatGPT's conversation tree structure to extract
    messages in order.

    ChatGPT stores conversations as a tree to support branching
    (when you edit a message and get alternative responses).
    We follow the main branch from root to leaf.
    """
    messages = []

    # Find root node (node with no parent)
    root_id = None
    for node_id, node in mapping.items():
        if not node.get("parent"):
            root_id = node_id
            break

    if not root_id:
        return []

    # Traverse from root following first child
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id, {})
        message = node.get("message")

        if message:
            role = message.get("author", {}).get("role")
            content = message.get("content", {})

            # Extract text from content parts
            text = ""
            if isinstance(content, dict):
                parts = content.get("parts", [])
                text = " ".join([p for p in parts if isinstance(p, str)])
            elif isinstance(content, str):
                text = content

            # Only include user and assistant messages with content
            if role in ("user", "assistant") and text.strip():
                messages.append({
                    "role": role,
                    "content": text.strip()
                })

        # Move to first child
        children = node.get("children", [])
        current_id = children[0] if children else None

    return messages


def extract_title_from_next_data(data: dict) -> str | None:
    """Extract conversation title from Next.js data."""
    try:
        page_props = data.get("props", {}).get("pageProps", {})
        conversation = (
            page_props.get("conversation") or
            page_props.get("serverResponse", {}).get("data", {}).get("conversation")
        )
        if conversation:
            return conversation.get("title")
    except (KeyError, TypeError):
        pass
    return None


def extract_messages_from_html(soup: BeautifulSoup) -> list[dict]:
    """
    Fallback: extract messages directly from HTML structure.
    ChatGPT renders messages with data-message-author-role attributes.
    """
    messages = []

    # Look for message containers with role attributes
    message_divs = soup.find_all(
        attrs={"data-message-author-role": True}
    )

    for div in message_divs:
        role = div.get("data-message-author-role")
        if role not in ("user", "assistant"):
            continue

        # Get text content
        text = div.get_text(separator="\n", strip=True)
        if text:
            messages.append({
                "role": role,
                "content": text
            })

    return messages


def extract_messages_from_text(soup: BeautifulSoup) -> list[dict]:
    """
    Last resort: look for any conversation-like patterns in the page text.
    Less accurate but catches cases where structure has changed.
    """
    messages = []
    text = soup.get_text(separator="\n")

    # Look for patterns like "You said:" / "ChatGPT said:"
    patterns = [
        (r"You said:\s*\n(.*?)(?=ChatGPT said:|You said:|$)", "user"),
        (r"ChatGPT said:\s*\n(.*?)(?=You said:|ChatGPT said:|$)", "assistant"),
    ]

    for pattern, role in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            content = match.strip()
            if content and len(content) > 10:
                messages.append({"role": role, "content": content})

    # Sort by appearance order (rough)
    return messages


# ─────────────────────────────────────────────────────
# Claude parser (placeholder)
# ─────────────────────────────────────────────────────

def parse_claude(html: str, url: str) -> dict:
    """
    Claude share link parser.

    Claude doesn't currently have public share links in the same
    way ChatGPT does, so this is a placeholder for future support.
    """
    return {
        "error": "Claude share links are not yet supported. Claude doesn't expose conversation data in share pages.",
        "messages": [],
        "platform": "claude",
        "title": None
    }