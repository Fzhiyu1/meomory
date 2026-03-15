# meomory: Memory Prosthetic for LLM Agents

## Problem

Current LLM agents are stateless. Every conversation starts from zero. The entire "memory" industry is essentially RAG — static vector similarity search over stored text chunks. RAG doesn't learn, doesn't forget, and returns the same results for the same query forever. It's a library, not memory.

## Core Insight

Real memory has three properties that RAG lacks:
1. **It learns from experience** — frequently useful associations get stronger
2. **It forgets** — unused associations naturally decay
3. **It activates passively** — you don't decide to "search" your memory; relevant memories surface automatically when cues appear

## Architecture

### Memory Hierarchy

```
L0 — Always in context (zero retrieval cost)
│    System prompt, identity, core rules
│    Analogy: CPU registers
│
L1 — Pre-conscious activation layer (near-zero cost)
│    DGD associative memory: M(key) → value
│    Online learning via Delta Gradient Descent
│    O(1) matrix multiply, not O(n) search
│    Analogy: L1/L2 cache
│
L2 — Vector search fallback (O(log n))
│    Cosine similarity over embeddings
│    Static, no learning — the "library" layer
│    Analogy: disk
│
L3 — Cold storage (raw session archives)
     Original conversation logs
     Analogy: tape backup
```

### Message Flow

```
User message arrives
       ↓
[Pre-conscious Activation Layer]     ← LLM doesn't see this
  Extract cues from message
  DGD: M @ query_embedding → activation vector
  Find nearest memory fragments to activation
  Inject matched fragments into context
       ↓
LLM processes (with memories already present)
  May also call recall_memory tool (active recall)
       ↓
LLM responds
       ↓
[Post-conscious Observation Layer]   ← LLM doesn't see this
  Embed response
  Compare response embedding vs each injected memory
  similarity > threshold → memory was "used" → positive signal
  similarity < threshold → memory was ignored → no update
       ↓
DGD updates M matrix
  Strengthen: query direction → used memory direction
  Natural decay: unused associations weaken over time
       ↓
M evolves. Next query through updated M produces different rankings.
```

### DGD (Delta Gradient Descent) as Memory Module

DGD is NOT a language model. It's a single weight matrix M that maps query vectors to activation vectors.

**Core formula:**
```
M_new = M_old × (αI − η·k·kᵀ) − η·error·kᵀ

Where:
  M     = [dim × dim] weight matrix (the "memory")
  k     = query embedding (projected to lower dim)
  error = M@k − target (prediction vs actual useful memory)
  α     = forgetting rate (controls decay of old associations)
  η     = learning rate
```

Two components:
- **Forgetting term** `M × (αI − η·k·kᵀ)`: decay old associations in the query direction
- **Learning term** `η·error·kᵀ`: write new association toward the correct memory

**Origin:** DGD comes from the Nested Learning paper (Behrouz et al., Google Research, NeurIPS 2025) where it's used inside the Hope architecture for test-time weight modification. We extracted it as a standalone external memory retrieval module — this application is novel.

### Auto-Feedback Loop

The key innovation: feedback signal comes from **observing the agent's behavior**, not from human labels.

```python
# After agent responds:
response_embedding = embed(agent_response[:500])

for each injected_memory:
    similarity = cosine(response_embedding, memory_embedding)
    if similarity > 0.5:
        # Agent's response is similar to this memory
        # → Agent likely referenced it → positive feedback
        dgd.update(query_vec, memory_vec)  # strengthen association
    else:
        # Agent ignored this memory → no update
        # (natural decay in DGD handles weakening over time)
```

No human in the loop. The agent doesn't know it's being observed.

### Dimensionality

- Raw embeddings: 4096-dim (qwen3-embedding)
- Random projection: 4096 → 256 (Johnson-Lindenstrauss, preserves cosine ordering)
- M matrix: 256 × 256 = 65K parameters
- Memory fragments: stored as 256-dim projected vectors

### Comparison with Existing Approaches

| Property | RAG | DGD Memory | Fine-tuning |
|----------|-----|------------|-------------|
| Learning | None | Online, per-query | Offline, batch |
| Forgetting | None (manual delete) | Natural decay via α | Catastrophic |
| Retrieval | O(n) similarity search | O(1) matrix multiply | N/A (in weights) |
| Personalization | None | Emerges from usage | Requires training data |
| Feedback signal | None | Auto (observe agent) | Labeled data |
| State | Stateless | M matrix evolves | Weights change |
| Cost | Cheap | Cheap (one matmul) | Expensive (GPU) |

### Connection to Neuroscience / Cognitive Science

| System concept | Brain analogy |
|----------------|---------------|
| Main LLM (frozen weights) | Neocortex (reasoning, generation) |
| DGD memory module | Hippocampus (associative memory, pattern completion) |
| Pre-conscious activation | Spreading activation before conscious awareness |
| Auto-feedback loop | Synaptic consolidation during experience |
| L0/L1/L2/L3 hierarchy | Working memory / episodic / semantic / archival |
| Forgetting term in DGD | Retrieval-induced forgetting, active suppression |

### Connection to DeepSeek Engram

DeepSeek's Engram paper (Jan 2026) independently validated similar architectural principles from a pure engineering path:

| Engram | meomory |
|--------|---------|
| Conditional memory (static lookup) | DGD (online learning lookup) |
| N-gram hash → O(1) retrieval | M @ query → O(1) retrieval |
| Async prefetch before computation | Pre-conscious activation before LLM |
| Context-aware gating (αₜ) | Auto-feedback (strengthen/ignore) |
| Multi-level cache (HBM/DRAM/SSD) | L0/L1/L2/L3 hierarchy |
| 20-25% sparsity allocation law | ~20% context budget for memory injection |

Engram provides large-capacity static storage. DGD provides online learning. They are complementary — one handles "how much to store", the other handles "how to adapt".

## Experimental Results

### Benchmark: MemoryBench DialSim-BigBang

805 Big Bang Theory conversation fragments, 14 questions with ground truth answers.

**DGD + Ground Truth Feedback (20 rounds):**

| Round | P@1 | P@3 | P@5 |
|-------|-----|-----|-----|
| 1 | 64.3% | 71.4% | 78.6% |
| 10 | 71.4% | 85.7% | 85.7% |
| 20 | **78.6%** | **85.7%** | **92.9%** |

Cosine baseline: ~57% P@1 (static, no improvement over rounds)

**DGD + Auto Feedback (in progress):**

| Round | P@1 | Feedback accuracy |
|-------|-----|-------------------|
| 1 | 57.1% | 68% |
| 2 | 64.3% | 65% |

Learning curve is rising under noisy auto-feedback. Experiment ongoing.

## Limitations

1. **Small test set** — 14 questions is not statistically significant
2. **Auto-feedback is noisy** — embedding similarity ≠ true usage; ~65-68% accuracy
3. **Not deployed** — all results are from offline benchmarks, not real agent usage
4. **No CMS yet** — single timescale DGD, no multi-frequency memory layers
5. **M matrix capacity** — 256×256 may be too small for thousands of memories

## Future Directions

1. **Gateway integration** — plug into OpenClaw as pre-conscious activation middleware
2. **CMS (Continuous Memory System)** — multi-frequency DGD layers (per-conversation / daily / weekly)
3. **Engram-style static layer** — N-gram hash for stable knowledge, DGD for adaptive patterns
4. **Thinking-as-cue** — use LLM's reasoning/thinking output as deeper retrieval cues for next turn
5. **Larger benchmarks** — validate on LoCoMo, LongMemEval with hundreds of questions

## Tech Stack

- Python 3.14
- qwen3-embedding 8B Q4_K_M (Ollama on RTX 4060)
- GPT-5.4 via cliproxy relay
- Pure Python DGD (no numpy, no FAISS — data scale is small)
- Random projection for dimensionality reduction (4096 → 256)

## Repository

```
meomory/
├── src/
│   ├── parser.py          # Session JSONL parser
│   ├── compressor.py      # L2→L1→L0 compression (rule + LLM)
│   ├── embedder.py        # qwen3-embedding client
│   ├── store.py           # Vector store (brute-force cosine)
│   ├── matcher.py         # Memory matcher with threshold gating
│   ├── projection.py      # Random projection (4096→256)
│   ├── dgd.py             # DGD associative memory module
│   ├── feedback.py        # Auto-feedback layer
│   ├── llm_client.py      # GPT-5.4 API client
│   ├── tool.py            # recall_memory CLI (cosine)
│   └── tool_dgd.py        # recall_memory CLI (DGD, interactive)
├── scripts/
│   ├── demo_cycle.py      # Full loop demo (match→inject→respond→feedback→update)
│   ├── bench_dgd.py       # Cosine vs DGD benchmark
│   ├── bench_memorybench.py    # MemoryBench evaluation
│   └── bench_auto_feedback.py  # Auto-feedback learning curve experiment
├── memory/
│   ├── l1/                # L1 fragments (JSON)
│   ├── l0/                # L0 summaries (JSON)
│   └── vectors/           # Vector store + DGD model + projection matrix
└── docs/
    ├── architecture.md    # This document
    └── plans/             # Implementation plans
```

## Theoretical Foundation

This system's design decisions are grounded in a knowledge base of memory theory:

- **Actionable State** — memory's purpose is "what to do next", not "what was said"
- **Associative Memory** — M(key)→value; DGD/CMS are engineering implementations
- **Lossy Compression** — compression implies forgetting; forgetting is a feature, not a bug
- **Active Forgetting** — retrieval control, not deletion; context gating is runtime implementation
- **External Working Memory** — compress, correct, replay
- **Memory Prosthetic** — we're not building memory, we're building state restoration
- **Pre-conscious Activation** — memories should surface passively, not be actively searched
- **Memory vs Library** — vector DB is a library (no learning, no forgetting); associative memory approaches real memory
