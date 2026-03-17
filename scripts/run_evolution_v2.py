#!/usr/bin/env python3
"""Judge Prompt 进化搜索 v2：直接测 relevance，不模拟 agent。

用法:
    MEOMORY_LLM_KEY=sk-xxx .venv/bin/python scripts/run_evolution_v2.py
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution.runner_v2 import run_evolution_v2


def main():
    parser = argparse.ArgumentParser(description="Judge Prompt Evolution v2 (relevance)")
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="experiments/evolution-v2")
    args = parser.parse_args()

    api_key = os.environ.get("MEOMORY_LLM_KEY", "")
    if not api_key:
        print("Error: MEOMORY_LLM_KEY not set")
        sys.exit(1)

    asyncio.run(run_evolution_v2(
        pop_size=args.pop_size,
        n_generations=args.generations,
        n_eval_samples=args.samples,
        api_key=api_key,
        seed=args.seed,
        data_dir=args.data_dir,
    ))


if __name__ == "__main__":
    main()
