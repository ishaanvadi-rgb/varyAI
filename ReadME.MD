# varyAI

Recently, I switched from Gemini Pro to Claude.

Claude didn't know who I was. It didn't know what I was working on, what I'd already figured out, or what kind of answers I actually find useful. I had to start from scratch — re-introducing myself, re-explaining my projects, rebuilding context I'd spent weeks developing with another AI.

That friction bothered me. So I built varyAI.

---

## The idea

**Your profile travels with you. The AI is just the engine.**

varyAI remembers you across every conversation and every model. Switch from Llama to GPT-4o to DeepSeek mid-conversation — your context stays intact. The AI varies. You don't.

---

## What it does

**Memory that builds itself**
As you chat, varyAI quietly learns facts about you — your projects, preferences, skills, background. It stores everything locally on your device and uses it to personalize every response, across every model you use.

**Switch models mid-conversation**
Pick any model from the dropdown. Your memory profile stays the same. The new model already knows you.

**Import from ChatGPT**
Paste a ChatGPT share link. varyAI fetches the conversation, loads it, and you continue it with any model you choose — no copy-pasting, no re-explaining.

**Your profile, your control**
Everything is stored on your machine in a local database. You can see exactly what varyAI knows about you, edit any fact, delete anything, or export your entire profile as JSON. No profile is built on any AI provider's servers — the API is stateless, it processes your message and forgets it.

---

## How the memory works

After every message exchange, varyAI runs a silent background call asking the LLM:
*"What did you just learn about this user?"*

The response gets saved locally. Before your next message, it searches that database for facts relevant to what you're asking — and includes them in the prompt silently. The model feels like it knows you because, from its perspective, it does.

This is called RAG (Retrieval-Augmented Generation). I built it from scratch using ChromaDB for vector search and sentence-transformers for local embeddings.

---

## Models available

| Model | Provider |
|---|---|
| Llama 3.3 70B | Groq (free) |
| Llama 3.1 8B | Groq (free) |
| Llama 4 Scout 17B | Groq (free) |
| GPT-4o | OpenRouter (free tier) |
| DeepSeek V3 | OpenRouter (free tier) |
| Auto — best available free model | OpenRouter (free tier) |

---

## Setup

You'll need Python 3.10+ and Poetry.
```bash
git clone https://github.com/YOUR_USERNAME/varyAI.git
cd varyAI
poetry install
cp .env.example .env
```

Add at least one free API key to `.env`:
- **Groq** → [console.groq.com](https://console.groq.com)
- **OpenRouter** → [openrouter.ai](https://openrouter.ai)
```bash
make dev
```

Open `http://localhost:8000`

---

