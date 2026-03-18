"""冷启动蒸馏实验：LLM 标注 train set → DGD 学习 → test set 验证泛化。

实验设计：
  1. LoCoMo 1976 题 split → 500 train + 1476 test
  2. 条件 A: 纯 cosine baseline（无 DGD）
  3. 条件 B: DGD + GT 反馈 train only → test 上评测
  4. 条件 C: DGD + GT 反馈 all（上限参考）

如果 B 在 test 上比 A 高，说明 DGD 真的在泛化，不是死记硬背。

Usage: .venv/bin/python scripts/experiment_cold_start.py
"""
import json
import time
import numpy as np
from pathlib import Path

from src.funsearch.sandbox import compile_class
from src.funsearch.evaluator import _load_dataset_and_embeddings


def run_experiment():
    print("=" * 60)
    print("冷启动蒸馏实验：DGD 泛化能力验证")
    print("=" * 60)

    # Load dataset and gen059 code
    best = json.load(open("experiments/funsearch-v4/best.json"))
    code = best[0]["code"]
    prog_id = best[0]["id"]
    print(f"算法: {prog_id}")

    ds, proj_vecs_np, q_vecs_np, proj_vecs_list, q_vecs_list = \
        _load_dataset_and_embeddings("LoCoMo-full-all", 256)
    questions = ds.questions
    n_frags = len(proj_vecs_list)
    total_q = len(questions)
    print(f"片段: {n_frags}, 问题: {total_q}")

    # Split: first 500 = train, rest = test
    train_size = 500
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, total_q))
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    print()

    def evaluate_p1(cls_code, train_rounds, test_questions_idx):
        """训练 train_rounds 轮后在 test_questions_idx 上评测 P@1。"""
        cls = compile_class(cls_code)
        obj = cls(dim=256, alpha=1.0, eta=0.01)

        # Train: GT feedback on train set only
        for r in range(train_rounds):
            for qi in train_indices:
                q = questions[qi]
                target_idx = q["answer_indices"][0]
                obj.update(q_vecs_list[qi], proj_vecs_list[target_idx])

        # Evaluate on test set
        hits = 0
        for qi in test_questions_idx:
            q = questions[qi]
            answer_set = set(q["answer_indices"])
            act = obj.query(q_vecs_list[qi])
            act_np = np.array(act, dtype=np.float32)
            act_norm = np.linalg.norm(act_np)
            if act_norm > 0:
                act_np = act_np / act_norm
            scores = proj_vecs_np @ act_np
            top1_idx = int(np.argmax(scores))
            if top1_idx in answer_set:
                hits += 1
        return hits / len(test_questions_idx)

    def cosine_baseline(test_questions_idx):
        """纯 cosine baseline（无 DGD）。"""
        hits = 0
        for qi in test_questions_idx:
            q = questions[qi]
            answer_set = set(q["answer_indices"])
            q_np = q_vecs_np[qi]
            scores = proj_vecs_np @ q_np
            top1_idx = int(np.argmax(scores))
            if top1_idx in answer_set:
                hits += 1
        return hits / len(test_questions_idx)

    # ---- 实验 A: Cosine baseline ----
    print("实验 A: Cosine baseline（无 DGD）")
    t0 = time.time()
    cos_train = cosine_baseline(train_indices)
    cos_test = cosine_baseline(test_indices)
    cos_all = cosine_baseline(list(range(total_q)))
    print(f"  Train P@1: {cos_train:.1%}")
    print(f"  Test  P@1: {cos_test:.1%}")
    print(f"  All   P@1: {cos_all:.1%}")
    print(f"  ({time.time() - t0:.0f}s)")
    print()

    # ---- 实验 B: DGD + GT on train only → test ----
    print("实验 B: DGD 冷启动（GT on 500 train → test 1476）")
    for rounds in [1, 3, 5, 10]:
        t0 = time.time()
        p1_train = evaluate_p1(code, rounds, train_indices)
        p1_test = evaluate_p1(code, rounds, test_indices)
        dt = time.time() - t0
        delta_train = p1_train - cos_train
        delta_test = p1_test - cos_test
        print(f"  {rounds:2d} 轮: Train={p1_train:.1%} (+{delta_train:.1%}) | Test={p1_test:.1%} (+{delta_test:.1%}) | ({dt:.0f}s)")
    print()

    # ---- 实验 C: DGD + GT on ALL（上限参考）----
    print("实验 C: DGD 全量 GT（上限参考，10 轮）")
    t0 = time.time()
    cls = compile_class(code)
    obj = cls(dim=256, alpha=1.0, eta=0.01)
    for r in range(10):
        for qi in range(total_q):
            q = questions[qi]
            target_idx = q["answer_indices"][0]
            obj.update(q_vecs_list[qi], proj_vecs_list[target_idx])
    # Evaluate on all
    hits_train = 0
    hits_test = 0
    for qi in range(total_q):
        q = questions[qi]
        answer_set = set(q["answer_indices"])
        act = obj.query(q_vecs_list[qi])
        act_np = np.array(act, dtype=np.float32)
        act_norm = np.linalg.norm(act_np)
        if act_norm > 0:
            act_np = act_np / act_norm
        scores = proj_vecs_np @ act_np
        top1_idx = int(np.argmax(scores))
        if top1_idx in answer_set:
            if qi < train_size:
                hits_train += 1
            else:
                hits_test += 1
    p1_train_c = hits_train / len(train_indices)
    p1_test_c = hits_test / len(test_indices)
    p1_all_c = (hits_train + hits_test) / total_q
    print(f"  Train P@1: {p1_train_c:.1%}")
    print(f"  Test  P@1: {p1_test_c:.1%}")
    print(f"  All   P@1: {p1_all_c:.1%}")
    print(f"  ({time.time() - t0:.0f}s)")
    print()

    # ---- 总结 ----
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"                   Train(500)    Test(1476)")
    print(f"  A: Cosine        {cos_train:.1%}         {cos_test:.1%}")
    print(f"  B: 冷启动 10轮   (见上)         (见上)")
    print(f"  C: 全量 GT       {p1_train_c:.1%}         {p1_test_c:.1%}")
    print()
    print("如果 B_test > A_test → DGD 在泛化（不是作弊）")
    print("如果 B_test ≈ A_test → DGD 只记住了 train，没泛化")


if __name__ == "__main__":
    run_experiment()
