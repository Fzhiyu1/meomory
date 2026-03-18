# meomory

A memory prosthesis for LLM agents that learns from usage. Not better search — associative memory that improves with every interaction.

> **Blog (中文)**: [我让三个 AI 互相竞争进化，两天后它们发明了一个我看不懂的算法](https://juejin.cn/spost/7618056233687924782) | [知乎](https://zhuanlan.zhihu.com/p/2017403193101926520)

## What is this

LLM agents are brilliant amnesiacs — every conversation starts from scratch. Existing memory systems (RAG, vector search) are libraries: they store things but always return the same results for the same query.

meomory replaces static retrieval with **online-learning associative memory**. A 256×256 weight matrix that adjusts after every feedback signal — the more you use it, the better it gets.

**The algorithm wasn't written by a human. Three AI models invented it through evolutionary search.**

## Results

| Method | LoCoMo (1976 Q) | LongMemEval (470 Q) |
|--------|:---:|:---:|
| Cosine similarity (no learning) | 22.4% | 41.7% |
| BM25 keyword matching | 30.9% | 48.3% |
| Hand-written DGD | 32.8% | 56.0% |
| **AI-evolved algorithm** | **56.6%** | **92.3%** |

92.3% on LongMemEval approaches current SOTA (Hindsight+TEMPR 91.4%, EverMemOS 93.0%) — but those systems use GPT-4o + knowledge graphs + multi-strategy fusion. We use a single matrix.

## Evolution

Three AI models evolving unsupervised on a server for 24 hours:

```
qwen3.5:9b (60%)  — mass exploration, discovered base strategies
DeepSeek (30%)    — precise optimization, came from behind to win
GPT-5.4 (10%)    — innovation scout, first breakthrough
```

From 26.8% to 88.4% in two hours (500 Q subset), 56.6% on full validation. The evolved algorithm independently reinvented Hebbian learning (1949), Momentum (1980s), Adam optimizer (2014), and BatchNorm (2015) — LLMs discovered neural network layer structure without being told about neural networks.

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
  --iterations 0 --rounds 10 --samples 6
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
