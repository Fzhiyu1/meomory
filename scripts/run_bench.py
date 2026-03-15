#!/usr/bin/env python3
"""统一实验 CLI

用法:
    python3 scripts/run_bench.py experiments/configs/example.yaml
    python3 scripts/run_bench.py experiments/configs/example.yaml --only cosine-bigbang
"""
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.bench.runner import run_all


def _resolve_env_vars(obj):
    """递归替换配置中的 ${ENV_VAR} 为环境变量值"""
    if isinstance(obj, str):
        def replacer(m):
            var_name = m.group(1)
            val = os.environ.get(var_name)
            if val is None:
                print(f"  Warning: env var {var_name} not set, using empty string")
                return ""
            return val
        return re.sub(r"\$\{(\w+)\}", replacer, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/run_bench.py <config.yaml> [--only <name>]")
        sys.exit(1)

    config_path = sys.argv[1]
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        only = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    with open(config_path) as f:
        data = yaml.safe_load(f)

    data = _resolve_env_vars(data)

    experiments = data.get("experiments", [data] if "name" in data else [])

    if only:
        experiments = [e for e in experiments if e["name"] == only]

    if not experiments:
        print("No experiments to run")
        sys.exit(1)

    print(f"Running {len(experiments)} experiment(s) from {config_path}")
    asyncio.run(run_all(experiments))


if __name__ == "__main__":
    main()
