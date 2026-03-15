# meomory

A memory prosthetic for LLM agents — not a memory system, a state restoration system.

LLM agents are stateless. Every conversation starts from zero. RAG gives them a library, but a library isn't memory — it doesn't learn, doesn't forget, and returns the same results forever.

meomory replaces static vector search with **DGD associative memory** that learns from experience: memories that get used are strengthened, unused ones naturally decay. Feedback comes from observing agent behavior — no human labeling needed.

## How it works

```
User message arrives
  → Pre-conscious activation: DGD matches relevant memories
  → Inject into LLM context (LLM doesn't know this happened)
  → LLM responds normally
  → Observer judges which memories were actually used
  → DGD strengthens used associations, others decay
  → Next query: rankings shift based on accumulated experience
```

## Results

MemoryBench DialSim-bigbang (805 fragments, 14 questions, 20 rounds):

| Method | P@1 |
|--------|-----|
| Cosine (static, never learns) | 57.1% |
| DGD + ground truth feedback | 78.6% (+21.5pp) |
| **DGD + auto LLM judge feedback** | **71.4% (+14.3pp)** |

The auto-feedback version uses no human labels — an LLM observer determines which memories the agent referenced, and DGD learns from that signal alone.

## Key ideas

- **Memory ≠ library.** A library stores and searches. Memory compresses, learns, forgets, and activates passively.
- **DGD as standalone memory module.** Extracted from the Hope architecture (Nested Learning, NeurIPS 2025) and repurposed as an external retrieval module — this application is novel.
- **Auto-feedback loop.** Observe agent behavior → infer which memories were used → update associations. The agent is unaware it's being observed.
- **Pre-conscious activation.** Memories surface before the LLM starts thinking, not when it decides to search.

## Quick start

```bash
cd /Users/fangzhiyu/run/meomory
source .venv/bin/activate

# Query knowledge base (cosine)
python3 -m src.tool "主动遗忘"

# Query with DGD (interactive, learns from your feedback)
python3 -m src.tool_dgd

# Full loop demo (match → inject → DeepSeek responds → auto feedback → DGD updates)
MEOMORY_LLM_URL="https://api.deepseek.com/v1" \
MEOMORY_LLM_KEY="your-key" \
MEOMORY_LLM_MODEL="deepseek-chat" \
python3 scripts/demo_cycle.py
```

## Project structure

```
src/
  parser.py          # Session JSONL parser
  compressor.py      # L2→L1→L0 compression (rule + LLM)
  embedder.py        # qwen3-embedding client (Ollama)
  store.py           # Vector store (brute-force cosine)
  matcher.py         # Memory matcher with threshold gating
  projection.py      # Random projection 4096→256
  dgd.py             # DGD associative memory
  feedback.py        # Auto-feedback layer
  llm_client.py      # LLM API client
  tool.py            # recall_memory CLI (cosine)
  tool_dgd.py        # recall_memory CLI (DGD, interactive)

scripts/
  demo_cycle.py              # Full memory loop demo
  bench_v2.py                # LLM judge benchmark (parallel)
  index_knowledge_base.py    # Index knowledge base docs

docs/
  architecture.md            # Full architecture doc
  experiment-report-*.md     # Experiment reports
```

## Name

meomory = me + memory. Originally a typo. Kept it.
