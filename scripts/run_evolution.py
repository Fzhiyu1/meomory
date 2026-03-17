#!/usr/bin/env python3
"""Judge Prompt 进化搜索 CLI

用法:
    MEOMORY_LLM_KEY=sk-xxx .venv/bin/python scripts/run_evolution.py
    MEOMORY_LLM_KEY=sk-xxx .venv/bin/python scripts/run_evolution.py --pop-size 10 --generations 15 --samples 200
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution.runner import run_evolution


def main():
    parser = argparse.ArgumentParser(description="Judge Prompt Evolution")
    parser.add_argument("--pop-size", type=int, default=10, help="Population size")
    parser.add_argument("--generations", type=int, default=15, help="Number of generations")
    parser.add_argument("--samples", type=int, default=200, help="Eval samples per generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="experiments/evolution")
    args = parser.parse_args()

    api_key = os.environ.get("MEOMORY_LLM_KEY", "")
    if not api_key:
        print("Error: MEOMORY_LLM_KEY not set")
        sys.exit(1)

    best = asyncio.run(run_evolution(
        pop_size=args.pop_size,
        n_generations=args.generations,
        n_eval_samples=args.samples,
        api_key=api_key,
        seed=args.seed,
        data_dir=args.data_dir,
    ))

    print(f"\nBest prompt saved to {args.data_dir}/best.json")


if __name__ == "__main__":
    main()
