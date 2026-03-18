"""FunSearch 主循环：采样 -> 编译 -> 评测 -> 注册，循环。

v5: 并行 LLM 采样 + 混合 fitness (70% clean + 30% noisy@80%) + 异步全量验证。
"""
import asyncio
import os
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from multiprocessing import Process, Queue
from pathlib import Path

from src.bench.backends import DeepSeekBackend, OllamaBackend, LLMBackend
from src.funsearch.database import ProgramsDatabase, Program, ScoresPerTest, _reduce_score
from src.funsearch.sandbox import compile_update_function, _calls_ancestor
from src.funsearch.evaluator import evaluate_update_fn
from src.funsearch.sampler import sample_new_program
from src.funsearch.specification import INITIAL_IMPLEMENTATION, EVOLVE_FUNCTION_NAME


def _eval_in_subprocess(code: str, dataset_name: str, dim: int, alpha: float,
                        eta: float, n_rounds: int, max_questions: int,
                        random_offset: int = 0,
                        noise_accuracy: float = 1.0) -> dict | None:
    """在子进程中评测一个 update 函数或类。

    random_offset > 0 时，从数据集的 offset 位置开始取 max_questions 题（循环取），
    每次迭代用不同 offset 防止过拟合固定子集。
    noise_accuracy < 1.0 时，以该概率给正确反馈，用于噪声鲁棒性评测。
    """
    from src.funsearch.sandbox import compile_update_function as _compile_fn
    from src.funsearch.sandbox import compile_class as _compile_cls
    from src.funsearch.evaluator import evaluate_update_fn as _eval

    code_stripped = code.strip()
    if code_stripped.startswith("class "):
        obj = _compile_cls(code_stripped)
        if obj is None:
            return None
    else:
        obj = _compile_fn(code_stripped)
        if obj is None:
            return None

    return _eval(obj, dataset_name=dataset_name, dim=dim, alpha=alpha,
                 eta=eta, n_rounds=n_rounds, max_questions=max_questions,
                 random_offset=random_offset, noise_accuracy=noise_accuracy)


# 混合 fitness 常量（模块级，seed 和主循环共用）
NOISE_ACCURACY = 0.8
CLEAN_WEIGHT = 0.7
NOISE_WEIGHT = 0.3


def _eval_mixed_in_subprocess(code: str, dataset_name: str, dim: int, alpha: float,
                               eta: float, n_rounds: int, max_questions: int,
                               random_offset: int = 0) -> dict | None:
    """混合 fitness 评测：clean + noisy，加权合并。用于 seed 和主循环。"""
    clean = _eval_in_subprocess(code, dataset_name, dim, alpha, eta, n_rounds, max_questions,
                                 random_offset=random_offset, noise_accuracy=1.0)
    noisy = _eval_in_subprocess(code, dataset_name, dim, alpha, eta, n_rounds, max_questions,
                                 random_offset=random_offset, noise_accuracy=NOISE_ACCURACY)
    if clean is None or noisy is None:
        return None
    if clean.get("error") or noisy.get("error"):
        return clean
    mixed_p1 = CLEAN_WEIGHT * clean["p_at_1"] + NOISE_WEIGHT * noisy["p_at_1"]
    spt = {}
    clean_spt = clean.get("scores_per_test", {})
    noisy_spt = noisy.get("scores_per_test", {})
    for k in clean_spt:
        spt[k] = CLEAN_WEIGHT * clean_spt.get(k, 0) + NOISE_WEIGHT * noisy_spt.get(k, 0)
    spt["clean_overall"] = clean["p_at_1"]
    spt["noisy_overall"] = noisy["p_at_1"]
    result = dict(clean)
    result["p_at_1"] = mixed_p1
    result["clean_p1"] = clean["p_at_1"]
    result["noisy_p1"] = noisy["p_at_1"]
    result["scores_per_test"] = spt
    return result


def _auto_git_commit(data_dir: str, iteration: int):
    """自动 git commit 实验数据（静默失败）。"""
    try:
        repo_root = Path(data_dir)
        while repo_root != repo_root.parent:
            if (repo_root / ".git").exists():
                break
            repo_root = repo_root.parent
        else:
            return
        subprocess.run(
            ["git", "add", str(data_dir)],
            cwd=repo_root, capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", f"auto: iter {iteration} checkpoint"],
            cwd=repo_root, capture_output=True, timeout=10,
        )
    except Exception:
        pass


class ModelEnsemble:
    """多模型混合采样器。按权重随机选模型。"""

    def __init__(self, models: list[tuple[str, LLMBackend, float]]):
        """
        Args:
            models: [(name, backend, weight), ...]
                e.g. [("9b", ollama, 0.6), ("deepseek", ds, 0.3), ("gpt54", gpt, 0.1)]
        """
        self.models = models
        self.names = [m[0] for m in models]
        self.backends = [m[1] for m in models]
        self.weights = [m[2] for m in models]
        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        # 统计
        self.call_counts = {name: 0 for name in self.names}
        self.fail_counts = {name: 0 for name in self.names}

    def pick(self) -> tuple[str, LLMBackend]:
        """按权重随机选一个模型。"""
        r = random.random()
        cumsum = 0
        for i, w in enumerate(self.weights):
            cumsum += w
            if r <= cumsum:
                self.call_counts[self.names[i]] += 1
                return self.names[i], self.backends[i]
        self.call_counts[self.names[-1]] += 1
        return self.names[-1], self.backends[-1]

    def record_fail(self, name: str):
        self.fail_counts[name] = self.fail_counts.get(name, 0) + 1

    def stats(self) -> str:
        parts = []
        for name in self.names:
            calls = self.call_counts[name]
            fails = self.fail_counts[name]
            parts.append(f"{name}:{calls}({fails}f)")
        return " ".join(parts)


async def run_funsearch(
    num_islands: int = 5,
    max_iterations: int = 0,  # 0 = 无限
    samples_per_iteration: int = 6,
    eval_dataset: str = "LoCoMo-full-all",
    eval_max_questions: int = 500,
    eval_rounds: int = 10,
    dim: int = 256,
    alpha: float = 1.0,
    eta: float = 0.01,
    data_dir: str = "experiments/funsearch",
    seed_from: str | None = None,
    # 多模型配置
    ollama_host: str | None = None,
    ollama_model: str = "qwen3.5:9b",
    deepseek_key: str = "",
    gpt54_url: str = "",
    gpt54_key: str = "",
    model_weights: str = "60,30,10",  # ollama,deepseek,gpt54
):
    """运行 FunSearch 进化循环。

    支持多模型混合：Ollama(本地) + DeepSeek(API) + GPT-5.4(cliproxy)。
    max_iterations=0 表示无限循环，Ctrl+C 优雅退出。
    """
    # 构建模型集合
    models = []
    weights = [float(w) for w in model_weights.split(",")]

    if ollama_host:
        models.append(("9b", OllamaBackend(host=ollama_host, model=ollama_model), weights[0] if len(weights) > 0 else 60))
    if deepseek_key:
        models.append(("deepseek", DeepSeekBackend(api_key=deepseek_key), weights[1] if len(weights) > 1 else 30))
    if gpt54_url and gpt54_key:
        models.append(("gpt54", DeepSeekBackend(api_key=gpt54_key, url=gpt54_url, model="gpt-5.4"), weights[2] if len(weights) > 2 else 10))

    # 如果没有任何模型配置，fallback 到 DeepSeek
    if not models:
        if deepseek_key:
            models.append(("deepseek", DeepSeekBackend(api_key=deepseek_key), 100))
        else:
            raise ValueError("No LLM backend configured. Set --ollama-host, --deepseek-key, or --gpt54-url + --gpt54-key")

    ensemble = ModelEnsemble(models)
    print(f"Model ensemble: {[f'{m[0]}({m[2]:.0f}%)' for m in models]}")

    db = ProgramsDatabase(num_islands=num_islands, data_dir=data_dir)

    # ---- 初始化种子 ----
    start_iter = len(db.all_programs)
    if start_iter == 0:
        seed_programs = []
        if seed_from:
            import json as _json
            print(f"Loading seeds from {seed_from}...")
            with open(seed_from) as _f:
                seed_data = _json.load(_f)
            seen_codes = set()
            for sd in seed_data:
                code = sd["code"].strip()
                if code not in seen_codes:
                    seen_codes.add(code)
                    seed_programs.append(code)
            print(f"  Loaded {len(seed_programs)} unique seeds")
        else:
            seed_programs = [INITIAL_IMPLEMENTATION.strip()]

        print(f"Evaluating {len(seed_programs)} seeds (parallel, mixed fitness)...")
        seed_futures = []
        with ProcessPoolExecutor(max_workers=min(len(seed_programs), 12)) as pool:
            for i, code in enumerate(seed_programs):
                f = pool.submit(
                    _eval_mixed_in_subprocess, code,
                    eval_dataset, dim, alpha, eta, eval_rounds, eval_max_questions,
                )
                seed_futures.append((i, code, f))
            for i, code, f in seed_futures:
                try:
                    result = f.result(timeout=300)
                except Exception:
                    result = None
                seed = Program(id=f"seed-{i:03d}", code=code, generation=0)
                if result and not result.get("error"):
                    spt = result.get("scores_per_test", {"overall": result["p_at_1"]})
                    score = _reduce_score(spt)
                    seed.score = score
                    seed.eval_details = result
                    seed.scores_per_test = dict(spt)
                else:
                    score = 0.0
                    spt = {"overall": 0.0}
                    seed.score = 0.0
                island_id = i % num_islands
                seed.island_id = island_id
                db.register(seed, island_id=island_id, scores_per_test=spt)
                print(f"  Seed {i}: P@1={score:.1%} → island {island_id}")
        db.snapshot()
        db.save()
        start_iter = 1

    infinite = (max_iterations == 0)
    iter_limit = float('inf') if infinite else max_iterations
    mode_str = "infinite" if infinite else f"{max_iterations} iterations"

    print(f"\nStarting FunSearch v5 ({mode_str}, {num_islands} islands)")
    print(f"  Dataset: {eval_dataset}, dim={dim}, α={alpha}, η={eta}")
    print(f"  Eval: {eval_max_questions} questions × {eval_rounds} rounds")
    print(f"  Mixed fitness: {0.7:.0%} clean + {0.3:.0%} noisy (80% accuracy)")
    print(f"  Samples/iter: {samples_per_iteration} (parallel LLM)")
    print(f"  Models: {ensemble.stats()}")
    print(f"  Already evaluated: {len(db.all_programs)} programs")

    best_ever = db.get_best()
    if best_ever:
        print(f"  Current best: {best_ever.score:.1%} ({best_ever.id})")

    iteration = start_iter
    try:
        while iteration <= iter_limit:
            t0 = time.time()

            parents, island_id = db.get_prompt_programs()
            if not parents:
                print(f"  Iter {iteration}: No parents, skipping")
                iteration += 1
                continue

            # 并行生成候选（多模型混合，asyncio.gather）
            candidates = []
            sample_tasks = []
            sample_models = []
            for s in range(samples_per_iteration):
                model_name, backend = ensemble.pick()
                sample_models.append(model_name)
                sample_tasks.append(
                    sample_new_program(parents, backend, island_id, iteration, s)
                )

            results = await asyncio.gather(*sample_tasks, return_exceptions=True)
            for s, (model_name, result) in enumerate(zip(sample_models, results)):
                if isinstance(result, Exception) or result is None:
                    ensemble.record_fail(model_name)
                    continue
                program = result
                if _calls_ancestor(program.code, EVOLVE_FUNCTION_NAME):
                    continue
                program.eval_details["model"] = model_name
                candidates.append(program)

            # 并行评测（带强制超时：子进程超 10 分钟直接 kill）
            successes = 0
            EVAL_TIMEOUT = 600  # 10 分钟（允许单次操作 ≤100ms 的复杂算法）

            def _eval_mixed_worker(code, ds, d, a, e, r, mq, q_out, offset=0):
                """在独立进程中跑混合 fitness，结果通过 Queue 返回。"""
                try:
                    result = _eval_mixed_in_subprocess(code, ds, d, a, e, r, mq, random_offset=offset)
                    q_out.put(result)
                except Exception:
                    q_out.put(None)

            if candidates:
                procs = []
                # 每迭代用不同的随机 offset，防止过拟合固定子集
                iter_offset = iteration * 500  # 每迭代偏移 500 题
                for prog in candidates:
                    q = Queue()
                    p = Process(target=_eval_mixed_worker, args=(
                        prog.code, eval_dataset, dim, alpha, eta, eval_rounds, eval_max_questions, q,
                        iter_offset,
                    ))
                    p.start()
                    procs.append((prog, p, q))

                for prog, p, q in procs:
                    p.join(timeout=EVAL_TIMEOUT)
                    if p.is_alive():
                        p.kill()
                        p.join()
                        result = None
                    else:
                        try:
                            result = q.get_nowait()
                        except Exception:
                            result = None

                        if result is None or result.get("error"):
                            prog.score = 0.0
                            prog.eval_details["error"] = "eval_failed"
                            prog.scores_per_test = {"overall": 0.0}
                            continue

                        spt = result.get("scores_per_test", {"overall": result["p_at_1"]})
                        score = _reduce_score(spt)
                        prog.score = score
                        prog.eval_details.update(result)
                        prog.scores_per_test = dict(spt)

                        db.register(prog, island_id=island_id, scores_per_test=spt)
                        successes += 1

                        if score > (best_ever.score if best_ever else 0):
                            best_ever = prog
                            model = prog.eval_details.get("model", "?")
                            clean_s = result.get("clean_p1", score)
                            noisy_s = result.get("noisy_p1", 0)
                            print(f"  ★ NEW BEST: {score:.1%} (clean={clean_s:.1%}, noisy={noisy_s:.1%}, {prog.id}, model={model})")
                            db.save()

                            # 异步全量验证（不阻塞主循环）
                            def _bg_full_eval(prog_ref, code, ds, d, a, e, r):
                                full_result = _eval_in_subprocess(code, ds, d, a, e, r, 9999)
                                if full_result and not full_result.get("error"):
                                    full_p1 = full_result["p_at_1"]
                                    ratio = score / full_p1 if full_p1 > 0 else float('inf')
                                    print(f"    → 全量 P@1: {full_p1:.1%} (膨胀比: {ratio:.1f}x)")
                                    prog_ref.eval_details["full_p1"] = full_p1
                                    prog_ref.eval_details["overfit_ratio"] = round(ratio, 2)
                                    db.save()
                                else:
                                    print(f"    → 全量验证失败")

                            bg = Process(target=_bg_full_eval, args=(
                                prog, prog.code, eval_dataset, dim, alpha, eta, eval_rounds,
                            ))
                            bg.start()
                            print(f"    → 全量验证已启动(后台)")

            # 不再需要自动扩大评测集——每迭代随机 500 题 + NEW BEST 时全量验证

            elapsed = time.time() - t0
            best = db.get_best()

            # 每次打印
            island_bests = [f"{isl.best_score:.1%}" for isl in db.islands]
            print(
                f"  Iter {iteration:3d}: {successes}/{len(candidates)} valid, "
                f"best={best.score:.1%}, islands=[{', '.join(island_bests)}] ({elapsed:.0f}s)"
            )

            # 保存策略
            if successes > 0:
                db.save()
            if iteration % 5 == 0:
                db.snapshot()
                db.save()
            if iteration % 10 == 0:
                # 自动 git commit 实验数据
                _auto_git_commit(data_dir, iteration)
            if iteration % 50 == 0:
                print(f"  [Stats] {ensemble.stats()}")

            iteration += 1

    except KeyboardInterrupt:
        print(f"\n  Interrupted at iteration {iteration}. Saving...")

    # 最终保存
    db.snapshot()
    db.save()

    print(f"\n{'='*60}")
    print("  FunSearch Complete")
    print(f"{'='*60}")
    best = db.get_best()
    if best:
        print(f"  Best P@1: {best.score:.1%}")
        print(f"  Best ID: {best.id}")
        print(f"  Total programs: {len(db.all_programs)}")
        print(f"  Model stats: {ensemble.stats()}")
        if best.scores_per_test:
            per_conv = {k: f"{v:.1%}" for k, v in best.scores_per_test.items()}
            print(f"  Scores: {per_conv}")
        print(f"  Code:\n{best.code}")
    return best
