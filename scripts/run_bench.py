#!/usr/bin/env python3
"""统一实验 CLI

用法:
    python3 scripts/run_bench.py experiments/configs/example.yaml
    python3 scripts/run_bench.py experiments/configs/example.yaml --only cosine-bigbang
"""
import asyncio
import io
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.bench.runner import run_all


class TeeWriter:
    """同时写 stdout 和日志文件"""

    def __init__(self, log_path, original):
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.original = original

    def write(self, text):
        self.original.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


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

    # 自动保存日志到 experiments/logs/{config_name}-{timestamp}.log
    log_dir = Path(__file__).parent.parent / "experiments" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_stem = Path(config_path).stem
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{config_stem}-{timestamp}.log"

    tee = TeeWriter(log_path, sys.stdout)
    sys.stdout = tee
    sys.stderr = TeeWriter(log_path, sys.__stderr__)

    print(f"Running {len(experiments)} experiment(s) from {config_path}")
    print(f"Log: {log_path}")
    try:
        asyncio.run(run_all(experiments))
    finally:
        sys.stdout = tee.original
        tee.close()
        print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
