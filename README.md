# meomory

A memory prosthesis for LLM agents that learns from usage. Not better search — associative memory that improves with every interaction.

> **Blog (中文)**: [我让三个 AI 互相竞争进化，两天后它们发明了一个我看不懂的算法](https://juejin.cn/spost/7618056233687924782) | [知乎](https://zhuanlan.zhihu.com/p/2017403193101926520)

## What is this

LLM agents are brilliant amnesiacs — every conversation starts from scratch. Existing memory systems (RAG, vector search) are libraries: they store things but always return the same results for the same query.

meomory replaces static retrieval with **online-learning associative memory**. A 256×256 weight matrix that adjusts after every feedback signal — the more you use it, the better it gets.

**The algorithm wasn't written by a human. Three AI models discovered it through evolutionary search.**

## Results

Embedding: qwen3-embedding 8B → random projection to 256d. The DGD learning layer is a 256×256 weight matrix on top.

### Train=Test evaluation (early, flawed)

Our initial evaluation trained and tested on the same questions. The numbers looked impressive but **did not reflect real-world generalization**:

| Method | LoCoMo P@1 (1976 Q) | LongMemEval P@1 (470 Q) |
|--------|:---:|:---:|
| Cosine similarity (no learning) | 22.4% | 41.7% |
| BM25 keyword matching | 30.9% | 48.3% |
| Hand-written DGD | 32.8% | 56.0% |
| AI-evolved algorithm (train=test) | 60.1% | 95.5% |

When we tested on **unseen questions** (train on 500, test on 1476), the evolved algorithm scored **0.9%** — worse than cosine baseline. The 60.1% was a fitness illusion caused by overfitting to the evaluation set.

### Online streaming evaluation (current, honest)

We redesigned the evaluation: each question is tested **before** the algorithm learns from it. Questions are shuffled across conversations to expose cross-topic interference.

| Method | Online P@1 (1976 Q, shuffled) |
|--------|:---:|
| Cosine similarity (no learning) | 22.4% |
| **AI-evolved algorithm (v6)** | **26.0%** |

The real gain from online learning is **+3.6pp** over cosine — modest but genuine. The evolved algorithm (gen107) discovered experience replay, row normalization, and learning rate decay to mitigate catastrophic interference.

### What we learned

- **Shared weight matrices cause catastrophic interference**: learning query A degrades retrieval for query B
- **Fitness function design matters more than algorithm design**: train=test rewards memorization, online streaming rewards generalization
- **The evolved algorithms are useful for repeated query patterns**: same user asking similar questions over time. Not useful for completely unseen queries.
- **Evolution independently discovered experience replay, residual connections, and adaptive learning rates** — not because it "invented" them, but because selection pressure guided it toward known effective techniques.

## Evolution

Three AI models evolving unsupervised on a server:

```
qwen3.5:9b (30%)   — mass exploration, local model
DeepSeek (20%)      — precise optimization
Opus 4.6 (50%)     — highest quality, primary driver
```

**v4 (train=test):** From 26.8% to 87% in two hours — fast but illusory. Evolved Adam + Momentum + dual-channel + BatchNorm.

**v6 (online streaming + shuffled):** From 22% to 26% overnight — slow but real. Evolved experience replay + row normalization + residual connections. Completely different algorithm direction under different selection pressure.

## Usage

**As a reranker plugin (plug into any memory system):**

```python
from src.mem0_integration.dgd_reranker import DGDReranker

reranker = DGDReranker({"dim": 256})

# Rerank after retrieval
results = your_memory_system.search(query)
reranked = reranker.rerank(query, results, top_k=5)

# Feedback: tell it which memory was correct
reranker.feedback(query, correct_memory)
# Next time, similar queries rank better
```

**Run evolutionary search (discover better algorithms):**

```bash
python scripts/run_funsearch.py \
  --ollama-host http://localhost:11434 \
  --deepseek-key sk-xxx \
  --iterations 0 --rounds 1 --samples 12 \
  --seed-from multi-arch
```

## Project Structure

```
src/funsearch/        — Evolution framework (island model + multi-model + sandbox + numpy)
src/evolution/        — Prompt evolution (Judge prompt 28.9% → 90.9%)
src/mem0_integration/ — Mem0 reranker plugin
src/dgd.py           — Original DGD associative memory
src/bench/            — Benchmark framework (65 experiments, 2446 questions)
scripts/dashboard.py  — Real-time evolution dashboard
experiments/          — All experiment results + evolution population data
```

## Inspiration

This project grew from my [cognitive science knowledge base](https://github.com/Fzhiyu1/cognitive-network) (49 concept cards). An AI read my notes on memory, forgetting, and preconscious activation, then auto-linked them to the DGD algorithm from the Hope/Nested Learning paper (NeurIPS 2025). I had [reproduced that paper](https://github.com/Fzhiyu1/hope-gpt) but focused on a different module — the AI found the piece I overlooked.

## Name

meomory = me + memory。打错了，留下了。

---

*[Zhiyu Fang](https://github.com/Fzhiyu1)*
