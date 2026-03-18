#!/usr/bin/env python3
"""DGD Update Rule FunSearch CLI

用法:
    # 纯本地 (Ollama)
    .venv/bin/python scripts/run_funsearch.py --ollama-host http://localhost:11434

    # 多模型混合 (本地 + DeepSeek + GPT-5.4)
    .venv/bin/python scripts/run_funsearch.py \
        --ollama-host http://100.94.126.19:11434 \
        --deepseek-key sk-xxx \
        --gpt54-url https://cliproxy.fzhiyu2333.top/v1 \
        --gpt54-key cpa_xxx \
        --model-weights 60,30,10

    # 从 v1 种子继续
    .venv/bin/python scripts/run_funsearch.py \
        --seed-from experiments/funsearch/best.json \
        --data-dir experiments/funsearch-v3
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.funsearch.runner import run_funsearch


def main():
    parser = argparse.ArgumentParser(description="DGD Update Rule FunSearch v4")

    # 进化参数
    parser.add_argument("--iterations", type=int, default=0, help="0 = infinite")
    parser.add_argument("--islands", type=int, default=5)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed-from", default=None, help="JSON file or 'multi-arch' for 5 architecture seeds")
    parser.add_argument("--seed-mode", default="file", choices=["file", "multi-arch"],
        help="file: load from JSON; multi-arch: use 5 architecture seeds from seeds.py")
    parser.add_argument("--data-dir", default="experiments/funsearch")

    # 评测参数
    parser.add_argument("--dataset", default="LoCoMo-full-all")
    parser.add_argument("--max-questions", type=int, default=9999, help="v6 default: full dataset")
    parser.add_argument("--rounds", type=int, default=1, help="v6 default: 1 round (online streaming)")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.01)

    # 模型配置
    parser.add_argument("--ollama-host", default=None, help="Ollama API URL (e.g. http://localhost:11434)")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--deepseek-key", default=os.environ.get("MEOMORY_LLM_KEY", ""))
    parser.add_argument("--gpt54-url", default="", help="GPT-5.4 API URL (e.g. https://cliproxy.fzhiyu2333.top/v1)")
    parser.add_argument("--gpt54-key", default="", help="GPT-5.4 API key")
    parser.add_argument("--model-weights", default="60,30,10", help="ollama,deepseek,gpt54 weights")

    args = parser.parse_args()

    asyncio.run(run_funsearch(
        num_islands=args.islands,
        max_iterations=args.iterations,
        samples_per_iteration=args.samples,
        eval_dataset=args.dataset,
        eval_max_questions=args.max_questions,
        eval_rounds=args.rounds,
        dim=args.dim,
        alpha=args.alpha,
        eta=args.eta,
        data_dir=args.data_dir,
        seed_from=args.seed_from,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        deepseek_key=args.deepseek_key,
        gpt54_url=args.gpt54_url,
        gpt54_key=args.gpt54_key,
        model_weights=args.model_weights,
    ))


if __name__ == "__main__":
    main()
