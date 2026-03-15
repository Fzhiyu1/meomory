"""实验运行器"""
import json
import time
from pathlib import Path

from src.embedder import get_embedding
from src.projection import create_projection_matrix, project
from src.bench.datasets import Dataset
from src.bench.methods import create_method, BM25Method, _norm
from src.bench.backends import create_backend

RESULTS_DIR = Path(__file__).parent.parent.parent / "experiments" / "results"


def _evaluate(q_vecs, proj_vecs, questions, query_fn):
    """计算 P@1/3/5"""
    hits = {1: 0, 3: 0, 5: 0}
    details = []
    for qi, q in enumerate(questions):
        answer_set = set(q["answer_indices"])
        scores = query_fn(q_vecs[qi])
        top_ids = [idx for _, idx in scores[:10]]
        for k in [1, 3, 5]:
            if any(idx in answer_set for idx in top_ids[:k]):
                hits[k] += 1
        best_rank = ">10"
        for r, idx in enumerate(top_ids):
            if idx in answer_set:
                best_rank = r + 1
                break
        details.append({"qi": qi, "rank": best_rank, "top_5": top_ids[:5]})
    total = len(questions)
    return {k: hits[k] / total if total > 0 else 0 for k in hits}, details


async def run_experiment(config: dict) -> dict:
    """运行一次完整实验"""
    name = config["name"]
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {name}")
    print(f"{'=' * 60}")

    # 加载数据集
    ds_name = config["dataset"]
    if ds_name.startswith("DialSim-"):
        sub = ds_name.split("-", 1)[1].lower()
        dataset = Dataset.load_dialsim(sub)
    elif ds_name.startswith("LoCoMo-"):
        index = int(ds_name.split("-", 1)[1])
        dataset = Dataset.load_locomo(index)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    print(f"  {dataset}")

    # 创建方法和后端
    method = create_method(config)
    backend = create_backend(config.get("llm_backend"))

    # Embedding（带缓存）
    dim = config.get("dgd", {}).get("dim", 256)
    n_rounds = config.get("rounds", 1)
    eval_every = config.get("eval_every", 2)

    cache_dir = Path(__file__).parent.parent.parent / "experiments" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset.name}_embed.json"

    if cache_file.exists():
        print(f"  Loading cached embeddings from {cache_file.name}")
        import json as _json
        with open(cache_file) as _f:
            cached = _json.load(_f)
        frag_vecs = cached["frag_vecs"]
        q_vecs_raw = cached["q_vecs"]
    else:
        print(f"  Embedding {len(dataset.fragments)} fragments...")
        frag_vecs = []
        for i, f in enumerate(dataset.fragments):
            frag_vecs.append(get_embedding(f["body"][:400]))
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(dataset.fragments)}")

        print(f"  Embedding {len(dataset.questions)} questions...")
        q_vecs_raw = [get_embedding(q["question"][:300]) for q in dataset.questions]

        # 保存缓存
        import json as _json
        with open(cache_file, "w") as _f:
            _json.dump({"frag_vecs": frag_vecs, "q_vecs": q_vecs_raw}, _f)
        print(f"  Cached to {cache_file.name}")

    proj = create_projection_matrix(4096, dim, seed=42)
    proj_vecs = [_norm(project(v, proj)) for v in frag_vecs]
    q_vecs = [_norm(project(v, proj)) for v in q_vecs_raw]

    # Setup method
    method.setup(proj_vecs)
    if isinstance(method, BM25Method):
        method.setup_bm25(dataset.fragments)

    # 实验结果
    result = {
        "name": name,
        "config": config,
        "dataset": dataset.name,
        "n_fragments": len(dataset.fragments),
        "n_questions": len(dataset.questions),
        "rounds": [],
    }

    # 对于不学习的方法（cosine/bm25），只跑 1 轮
    if config["method"] in ("cosine", "bm25"):
        if config["method"] == "bm25":
            # BM25 用文本查询
            hits = {1: 0, 3: 0, 5: 0}
            details = []
            for qi, q in enumerate(dataset.questions):
                answer_set = set(q["answer_indices"])
                scores = method.query_bm25(q["question"])
                top_ids = [idx for _, idx in scores[:10]]
                for k in [1, 3, 5]:
                    if any(idx in answer_set for idx in top_ids[:k]):
                        hits[k] += 1
                best_rank = ">10"
                for r, idx in enumerate(top_ids):
                    if idx in answer_set:
                        best_rank = r + 1
                        break
                details.append({"qi": qi, "rank": best_rank})
            total = len(dataset.questions)
            metrics = {k: hits[k] / total if total > 0 else 0 for k in hits}
        else:
            metrics, details = _evaluate(q_vecs, proj_vecs, dataset.questions, method.query)

        print(f"  {config['method']}: P@1={metrics[1]:.1%} P@3={metrics[3]:.1%} P@5={metrics[5]:.1%}")
        result["rounds"].append({"round": 0, "metrics": metrics, "details": details})
    else:
        # 学习型方法，跑 N 轮
        for r in range(1, n_rounds + 1):
            t0 = time.time()
            total_updates = 0
            total_correct = 0
            errors = 0

            # 并发执行所有题的 feedback（DGD 更新在 gather 后批量进行）
            async def _process_one(qi, q):
                scores = method.query(q_vecs[qi])
                top_indices = [idx for _, idx in scores[:3]]
                fb = await method.feedback(
                    q_vecs[qi], top_indices, q["answer_indices"],
                    dataset.fragments, proj_vecs, backend,
                    question_text=q.get("question", ""),
                )
                return fb

            import asyncio
            fbs = await asyncio.gather(*[
                _process_one(qi, q) for qi, q in enumerate(dataset.questions)
            ])
            for fb in fbs:
                total_updates += fb.get("updates", 0)
                total_correct += fb.get("correct", 0)
                if fb.get("error"):
                    errors += 1

            elapsed = time.time() - t0

            if r % eval_every == 0 or r == 1:
                metrics, details = _evaluate(q_vecs, proj_vecs, dataset.questions, method.query)
                accuracy = total_correct / total_updates if total_updates > 0 else 0
                round_data = {
                    "round": r,
                    "metrics": metrics,
                    "updates": total_updates,
                    "correct": total_correct,
                    "feedback_accuracy": round(accuracy, 3),
                    "errors": errors,
                    "elapsed": round(elapsed, 1),
                }
                result["rounds"].append(round_data)
                print(
                    f"  Round {r:2d}: P@1={metrics[1]:.1%} P@3={metrics[3]:.1%} P@5={metrics[5]:.1%}  "
                    f"(upd:{total_updates} acc:{accuracy:.0%} err:{errors} {elapsed:.1f}s)"
                )
            else:
                # 仍然记录但不 evaluate
                accuracy = total_correct / total_updates if total_updates > 0 else 0
                result["rounds"].append({
                    "round": r,
                    "updates": total_updates,
                    "correct": total_correct,
                    "feedback_accuracy": round(accuracy, 3),
                    "errors": errors,
                    "elapsed": round(elapsed, 1),
                })

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {result_path}")

    return result


async def run_all(configs: list[dict]) -> list[dict]:
    """顺序运行多个实验"""
    results = []
    for config in configs:
        result = await run_experiment(config)
        results.append(result)

    # 汇总表
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  {'Name':<30} {'P@1':>8} {'P@3':>8} {'P@5':>8}")
    print(f"  {'-' * 54}")
    for r in results:
        if r["rounds"]:
            last = [rd for rd in r["rounds"] if "metrics" in rd]
            if last:
                m = last[-1]["metrics"]
                print(f"  {r['name']:<30} {m[1]:>7.1%} {m[3]:>7.1%} {m[5]:>7.1%}")

    return results
