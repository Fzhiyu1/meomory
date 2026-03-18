# Judge Prompt 进化搜索系统 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用进化算法（GA/DE）自动搜索更优的 Judge prompt，将 feedback accuracy 从当前 33-43% 提升到 50%+。

**Architecture:** 参考 EvoPrompt (ICLR 2024) 的 GA/DE 策略，但只取进化核心逻辑（~200行），评测函数直接复用 meomory 已有的 benchmark pipeline。种群库持久化到 JSON，支持断点续跑。一个 Judge prompt 包含 system_prompt + user_prompt_template 两部分，都参与进化。

**Tech Stack:** Python 3.14, httpx (DeepSeek API), asyncio, json（零额外依赖）

**当前 baseline：**
- JUDGE_SYSTEM: `src/bench/methods.py:8-13`
- Judge user prompt: `src/bench/methods.py:169-172`
- LoCoMo feedback accuracy: **33.6%**
- LongMemEval feedback accuracy: **44.1%**

---

### Task 1: Judge Prompt 评测函数

**Files:**
- Create: `src/evolution/__init__.py`
- Create: `src/evolution/evaluator.py`

**作用：** 给定一个 Judge prompt（system + user template），在采样数据上评测其 feedback accuracy。

**Step 1: 创建 `src/evolution/__init__.py`**

```python
# empty
```

**Step 2: 创建 `src/evolution/evaluator.py`**

```python
"""Judge prompt 评测器。

给定一个 Judge prompt，在采样题目上测量 feedback accuracy。
accuracy = Judge 判断"有用"的片段确实在 GT answer_indices 中的比例。
"""
import asyncio
import json
import random

from src.bench.backends import DeepSeekBackend


async def evaluate_judge_prompt(
    system_prompt: str,
    user_prompt_template: str,
    samples: list[dict],
    backend: DeepSeekBackend,
    concurrency: int = 30,
) -> dict:
    """评测一个 Judge prompt 的 feedback accuracy。

    Args:
        system_prompt: Judge 的 system prompt
        user_prompt_template: Judge 的 user prompt 模板，
            可用变量: {mem_list}, {response}, {question}
        samples: 预构建的评测样本列表，每个包含:
            - question: 问题文本
            - top_bodies: top-3 检索到的片段文本列表
            - answer_indices_local: 在 top_bodies 中哪些是正确的 (0-based set)
        backend: DeepSeek API 后端
        concurrency: 最大并发数

    Returns:
        {"accuracy": float, "total": int, "correct": int, "errors": int, "details": list}
    """
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _eval_one(sample):
        async with sem:
            top_bodies = sample["top_bodies"]
            question = sample["question"]
            gt_local = sample["answer_indices_local"]  # set of 0-based indices

            # Step 1: Agent 回答
            injection = "\n".join(f"- {b}" for b in top_bodies)
            agent_prompt = f"[Context]\n{injection}\n\n[Question] {question}\n\nAnswer briefly based on context."
            try:
                response = await backend.chat(agent_prompt, max_tokens=200)
            except Exception:
                return {"error": True}

            # Step 2: Judge 判断
            mem_list = "\n".join(f"[{i}] {b}" for i, b in enumerate(top_bodies))
            judge_user = user_prompt_template.format(
                mem_list=mem_list,
                response=response[:500],
                question=question,
            )
            try:
                raw = await backend.chat(judge_user, system=system_prompt, max_tokens=50)
                text = raw.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                used = json.loads(text)
                if not isinstance(used, list):
                    used = []
            except Exception:
                return {"error": True}

            # Step 3: 计算准确率
            correct = 0
            total = 0
            for idx in used:
                if isinstance(idx, int) and 0 <= idx < len(top_bodies):
                    total += 1
                    if idx in gt_local:
                        correct += 1

            return {"correct": correct, "total": total, "error": False}

    tasks = [_eval_one(s) for s in samples]
    raw_results = await asyncio.gather(*tasks)

    total_updates = 0
    total_correct = 0
    errors = 0
    for r in raw_results:
        if r.get("error"):
            errors += 1
        else:
            total_updates += r["total"]
            total_correct += r["correct"]

    accuracy = total_correct / total_updates if total_updates > 0 else 0
    return {
        "accuracy": accuracy,
        "total_updates": total_updates,
        "total_correct": total_correct,
        "errors": errors,
        "n_samples": len(samples),
    }


def build_eval_samples(
    dataset_name: str = "LoCoMo-full-all",
    n_samples: int = 200,
    seed: int = 42,
) -> list[dict]:
    """从数据集构建评测样本。

    每个样本 = 一道题 + cosine top-3 + GT 标签。
    使用缓存的 embedding 避免重复计算。
    """
    import math
    from src.bench.datasets import Dataset
    from src.projection import create_projection_matrix, project

    # 加载数据集
    if dataset_name == "LoCoMo-full-all":
        ds = Dataset.load_locomo_full_all()
    elif dataset_name.startswith("LongMemEval"):
        ds = Dataset.load_longmemeval(dataset_name.split("-")[-1])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 加载缓存的 embedding
    from pathlib import Path
    cache_file = Path("experiments/cache") / f"{ds.name}_embed.json"
    with open(cache_file) as f:
        cached = json.load(f)
    frag_vecs = cached["frag_vecs"]
    q_vecs_raw = cached["q_vecs"]

    # 投影
    proj = create_projection_matrix(4096, 256, seed=42)

    def _norm(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v

    def _cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    proj_vecs = [_norm(project(v, proj)) for v in frag_vecs]
    q_vecs = [_norm(project(v, proj)) for v in q_vecs_raw]

    # 采样
    rng = random.Random(seed)
    indices = list(range(len(ds.questions)))
    rng.shuffle(indices)
    indices = indices[:n_samples]

    samples = []
    for qi in indices:
        q = ds.questions[qi]
        # cosine top-3
        scores = [(_cosine(q_vecs[qi], pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        top_3_idx = [idx for _, idx in scores[:3]]

        top_bodies = [ds.fragments[idx]["body"][:200] for idx in top_3_idx]
        answer_set = set(q["answer_indices"])
        # 哪些 top-3 在 GT 中
        answer_indices_local = {i for i, idx in enumerate(top_3_idx) if idx in answer_set}

        samples.append({
            "question": q["question"],
            "top_bodies": top_bodies,
            "answer_indices_local": answer_indices_local,
            "global_indices": top_3_idx,
        })

    return samples
```

**Step 3: 验证评测函数**

```bash
MEOMORY_LLM_KEY=${DEEPSEEK_API_KEY} \
.venv/bin/python -c "
import asyncio
from src.evolution.evaluator import evaluate_judge_prompt, build_eval_samples
from src.bench.backends import DeepSeekBackend

# 当前 baseline prompt
SYSTEM = '''You are a judge. Given injected memories and an agent's response, determine which memories the response actually referenced or used.
Rules:
- Only mark a memory as \"used\" if the response clearly draws on information from that memory
- Paraphrasing counts as usage
- Mentioning the same topic is NOT enough
- Output ONLY a JSON array of indices (0-based), e.g. [0, 2]. Output [] if none were used.'''

USER_TPL = 'Injected memories:\n{mem_list}\n\nAgent response:\n{response}\n\nWhich memories were used? Output JSON array only.'

backend = DeepSeekBackend(api_key='${DEEPSEEK_API_KEY}')
samples = build_eval_samples('LoCoMo-full-all', n_samples=50, seed=42)
print(f'Built {len(samples)} samples')

result = asyncio.run(evaluate_judge_prompt(SYSTEM, USER_TPL, samples, backend))
print(f'Baseline accuracy: {result[\"accuracy\"]:.1%} ({result[\"total_correct\"]}/{result[\"total_updates\"]}, {result[\"errors\"]} errors)')
"
```

Expected: accuracy ~33% (matching our experiment results).

**Step 4: Commit**

```bash
git add src/evolution/__init__.py src/evolution/evaluator.py
git commit -m "feat(evolution): judge prompt evaluator with sample builder"
```

---

### Task 2: 种群库

**Files:**
- Create: `src/evolution/population.py`

**作用：** 持久化管理所有候选 prompt 的存储、谱系追踪、选择。

**Step 1: 创建 `src/evolution/population.py`**

```python
"""种群库：持久化存储、谱系追踪、选择操作。"""
import json
import time
from pathlib import Path


class Individual:
    """一个候选 Judge prompt。"""

    def __init__(
        self,
        id: str,
        system_prompt: str,
        user_prompt_template: str,
        fitness: float = 0.0,
        generation: int = 0,
        parent_ids: list[str] = None,
        mutation_type: str = "seed",
        eval_details: dict = None,
    ):
        self.id = id
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.fitness = fitness
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.mutation_type = mutation_type
        self.eval_details = eval_details or {}
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {
            "id": self.id,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_type": self.mutation_type,
            "eval_details": self.eval_details,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        ind = cls(
            id=d["id"],
            system_prompt=d["system_prompt"],
            user_prompt_template=d["user_prompt_template"],
            fitness=d.get("fitness", 0),
            generation=d.get("generation", 0),
            parent_ids=d.get("parent_ids", []),
            mutation_type=d.get("mutation_type", "unknown"),
            eval_details=d.get("eval_details", {}),
        )
        ind.created_at = d.get("created_at", "")
        return ind


class Population:
    """种群库，支持持久化和谱系追踪。"""

    def __init__(self, data_dir: str = "experiments/evolution"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.population_file = self.data_dir / "population.json"
        self.history_file = self.data_dir / "history.json"
        self.best_file = self.data_dir / "best.json"

        self.current: list[Individual] = []
        self.history: list[dict] = []  # 每代快照
        self.all_evaluated: dict[str, Individual] = {}  # id → Individual
        self._load()

    def _load(self):
        """从磁盘恢复状态。"""
        if self.population_file.exists():
            with open(self.population_file) as f:
                data = json.load(f)
            self.current = [Individual.from_dict(d) for d in data.get("current", [])]
            self.all_evaluated = {
                d["id"]: Individual.from_dict(d)
                for d in data.get("all_evaluated", [])
            }
        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)

    def save(self):
        """持久化到磁盘。"""
        with open(self.population_file, "w") as f:
            json.dump({
                "current": [ind.to_dict() for ind in self.current],
                "all_evaluated": [ind.to_dict() for ind in self.all_evaluated.values()],
            }, f, ensure_ascii=False, indent=2)

        with open(self.history_file, "w") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        # 更新 best top-10
        top10 = sorted(self.all_evaluated.values(), key=lambda x: x.fitness, reverse=True)[:10]
        with open(self.best_file, "w") as f:
            json.dump([ind.to_dict() for ind in top10], f, ensure_ascii=False, indent=2)

    def add(self, ind: Individual):
        """添加已评测的个体。"""
        self.all_evaluated[ind.id] = ind

    def snapshot(self, generation: int):
        """记录当代快照。"""
        fitnesses = [ind.fitness for ind in self.current]
        self.history.append({
            "generation": generation,
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "population_size": len(self.current),
            "best_id": max(self.current, key=lambda x: x.fitness).id if self.current else "",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def get_best(self) -> Individual | None:
        if not self.all_evaluated:
            return None
        return max(self.all_evaluated.values(), key=lambda x: x.fitness)

    @property
    def generation(self) -> int:
        return len(self.history)
```

**Step 2: Commit**

```bash
git add src/evolution/population.py
git commit -m "feat(evolution): population store with persistence and lineage tracking"
```

---

### Task 3: 变异算子（GA 交叉 + DE 差分）

**Files:**
- Create: `src/evolution/mutator.py`

**作用：** 用 LLM 对 Judge prompt 做交叉和变异，生成新候选。

**Step 1: 创建 `src/evolution/mutator.py`**

```python
"""变异算子：用 LLM 生成 Judge prompt 的变异版本。

实现两种策略（参考 EvoPrompt）：
- GA: 选两个 parent，让 LLM 做交叉+变异
- DE: 取 3 个候选做差分变异
"""
import json
import re

from src.bench.backends import DeepSeekBackend


GA_TEMPLATE = """You are an expert prompt engineer. Your task is to create a better judge prompt by combining and improving two existing prompts.

## Parent Prompt 1 (system):
{parent1_system}

## Parent Prompt 1 (user template):
{parent1_user}

## Parent Prompt 2 (system):
{parent2_system}

## Parent Prompt 2 (user template):
{parent2_user}

## Instructions:
1. Crossover: Combine the best aspects of both parent prompts
2. Mutate: Improve the combined prompt to be more effective at judging which memories an AI agent actually used in its response
3. The prompt should help a judge determine which injected memories were truly referenced (not just topically related)

## Output format (strict JSON):
{{"system_prompt": "...", "user_prompt_template": "..."}}

The user_prompt_template MUST contain these placeholders: {{mem_list}}, {{response}}, {{question}}
Output ONLY the JSON, nothing else."""

DE_TEMPLATE = """You are an expert prompt engineer. You will see a base prompt and a "difference" between two other prompts. Apply that difference to improve the base prompt.

## Base prompt (system):
{base_system}

## Base prompt (user template):
{base_user}

## Reference prompt A (system):
{ref_a_system}

## Reference prompt B (system):
{ref_b_system}

## Instructions:
1. Identify the key differences between Reference A and Reference B
2. Apply similar improvements to the Base prompt
3. The result should be a better judge prompt for determining which memories an AI agent actually used

## Output format (strict JSON):
{{"system_prompt": "...", "user_prompt_template": "..."}}

The user_prompt_template MUST contain these placeholders: {{mem_list}}, {{response}}, {{question}}
Output ONLY the JSON, nothing else."""


def _parse_prompt_json(raw: str) -> dict | None:
    """从 LLM 输出中提取 JSON。"""
    text = raw.strip()
    # 去掉 markdown code block
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        d = json.loads(text)
        if "system_prompt" in d and "user_prompt_template" in d:
            # 验证必要占位符
            tpl = d["user_prompt_template"]
            if "{mem_list}" in tpl and "{response}" in tpl:
                return d
    except json.JSONDecodeError:
        # 尝试用正则提取
        m = re.search(r'\{[^{}]*"system_prompt"[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                if "{mem_list}" in d.get("user_prompt_template", ""):
                    return d
            except json.JSONDecodeError:
                pass
    return None


async def mutate_ga(
    parent1_system: str, parent1_user: str,
    parent2_system: str, parent2_user: str,
    backend: DeepSeekBackend,
) -> dict | None:
    """GA 交叉+变异：从两个 parent 生成一个 child。"""
    prompt = GA_TEMPLATE.format(
        parent1_system=parent1_system,
        parent1_user=parent1_user,
        parent2_system=parent2_system,
        parent2_user=parent2_user,
    )
    try:
        raw = await backend.chat(prompt, max_tokens=500)
        return _parse_prompt_json(raw)
    except Exception:
        return None


async def mutate_de(
    base_system: str, base_user: str,
    ref_a_system: str, ref_b_system: str,
    backend: DeepSeekBackend,
) -> dict | None:
    """DE 差分变异：base + F*(a-b)。"""
    prompt = DE_TEMPLATE.format(
        base_system=base_system,
        base_user=base_user,
        ref_a_system=ref_a_system,
        ref_b_system=ref_b_system,
    )
    try:
        raw = await backend.chat(prompt, max_tokens=500)
        return _parse_prompt_json(raw)
    except Exception:
        return None
```

**Step 2: Commit**

```bash
git add src/evolution/mutator.py
git commit -m "feat(evolution): GA crossover and DE differential mutation operators"
```

---

### Task 4: 进化主循环

**Files:**
- Create: `src/evolution/runner.py`

**作用：** 编排进化流程：初始化种群 → 评测 → 选择 → 变异 → 评测 → 循环。

**Step 1: 创建 `src/evolution/runner.py`**

```python
"""进化主循环：GA 策略的 Judge prompt 进化。"""
import asyncio
import random
import time

from src.bench.backends import DeepSeekBackend
from src.evolution.evaluator import evaluate_judge_prompt, build_eval_samples
from src.evolution.population import Population, Individual
from src.evolution.mutator import mutate_ga


# 种子 prompts：当前 baseline + 手工变体
SEED_PROMPTS = [
    {
        "system": (
            "You are a judge. Given injected memories and an agent's response, "
            "determine which memories the response actually referenced or used.\n"
            "Rules:\n"
            "- Only mark a memory as \"used\" if the response clearly draws on information from that memory\n"
            "- Paraphrasing counts as usage\n"
            "- Mentioning the same topic is NOT enough\n"
            "- Output ONLY a JSON array of indices (0-based), e.g. [0, 2]. Output [] if none were used."
        ),
        "user": (
            "Injected memories:\n{mem_list}\n\n"
            "Agent response:\n{response}\n\n"
            "Which memories were used? Output JSON array only."
        ),
    },
    {
        "system": (
            "You are evaluating memory retrieval quality. Given a question, retrieved memory fragments, "
            "and an AI response, determine which fragments contain information that appears in the response.\n"
            "- A fragment is 'used' if any specific fact, name, date, or detail from it appears in the response\n"
            "- General topic overlap does NOT count\n"
            "- Output a JSON array of 0-based indices. Output [] if none match."
        ),
        "user": (
            "Question: {question}\n\n"
            "Retrieved memories:\n{mem_list}\n\n"
            "AI response:\n{response}\n\n"
            "Which memory indices contain facts that appear in the response? JSON array only."
        ),
    },
    {
        "system": (
            "Judge whether each memory fragment was actually used to generate the response.\n"
            "A memory is 'used' ONLY if the response contains specific information (names, dates, events, facts) "
            "that could only come from that memory.\n"
            "Output format: JSON array of indices, e.g. [0] or [1,2] or []."
        ),
        "user": (
            "Memory fragments:\n{mem_list}\n\n"
            "Question asked: {question}\n\n"
            "Generated response:\n{response}\n\n"
            "Which fragments were used? Respond with JSON array only."
        ),
    },
    {
        "system": (
            "你是一个记忆使用判断器。给定注入的记忆片段和 AI 的回答，判断哪些记忆被实际引用了。\n"
            "规则：\n"
            "- 只有回答中明确使用了记忆中的具体信息（人名、日期、事件）才算"引用"\n"
            "- 仅仅话题相关不算\n"
            "- 输出 JSON 数组，如 [0, 2]。如果都没用到输出 []"
        ),
        "user": (
            "记忆片段:\n{mem_list}\n\n"
            "问题: {question}\n\n"
            "AI回答:\n{response}\n\n"
            "哪些记忆被引用了？只输出 JSON 数组。"
        ),
    },
    {
        "system": (
            "For each memory fragment, check if the response contains a specific claim, fact, or detail "
            "that is ONLY found in that fragment and not in the others or in general knowledge.\n"
            "Output a JSON array of fragment indices that were genuinely used. Be strict."
        ),
        "user": (
            "Fragments:\n{mem_list}\n\n"
            "Question: {question}\n"
            "Response: {response}\n\n"
            "Indices of used fragments (JSON array):"
        ),
    },
]


async def run_evolution(
    pop_size: int = 10,
    n_generations: int = 15,
    n_eval_samples: int = 200,
    eval_datasets: list[str] = None,
    api_key: str = "",
    seed: int = 42,
    data_dir: str = "experiments/evolution",
):
    """运行 GA 进化搜索。"""
    if eval_datasets is None:
        eval_datasets = ["LoCoMo-full-all", "LongMemEval-oracle"]

    backend = DeepSeekBackend(api_key=api_key)
    pop = Population(data_dir=data_dir)
    rng = random.Random(seed)

    # 构建评测样本（每个数据集采样 n_eval_samples/2）
    print("Building evaluation samples...")
    all_samples = {}
    per_ds = n_eval_samples // len(eval_datasets)
    for ds_name in eval_datasets:
        all_samples[ds_name] = build_eval_samples(ds_name, n_samples=per_ds, seed=seed)
        print(f"  {ds_name}: {len(all_samples[ds_name])} samples")

    combined_samples = []
    for samples in all_samples.values():
        combined_samples.extend(samples)

    # 评测一个个体
    async def eval_individual(ind: Individual) -> float:
        result = await evaluate_judge_prompt(
            ind.system_prompt, ind.user_prompt_template,
            combined_samples, backend, concurrency=30,
        )
        ind.eval_details = result
        ind.fitness = result["accuracy"]
        pop.add(ind)
        return ind.fitness

    # 断点续跑：如果种群已存在且有足够个体，跳过初始化
    start_gen = pop.generation
    if pop.current and len(pop.current) >= pop_size:
        print(f"Resuming from generation {start_gen}, population size {len(pop.current)}")
        print(f"  Best fitness so far: {pop.get_best().fitness:.1%}")
    else:
        # 初始化种群
        print(f"Initializing population with {len(SEED_PROMPTS)} seeds + {pop_size - len(SEED_PROMPTS)} paraphrases...")
        init_individuals = []

        for i, sp in enumerate(SEED_PROMPTS[:pop_size]):
            ind = Individual(
                id=f"gen000-{i:03d}",
                system_prompt=sp["system"],
                user_prompt_template=sp["user"],
                generation=0,
                mutation_type="seed",
            )
            fitness = await eval_individual(ind)
            init_individuals.append(ind)
            print(f"  Seed {i}: {fitness:.1%}")

        # 如果种子不够 pop_size，用 GA 变异补充
        while len(init_individuals) < pop_size:
            p1, p2 = rng.sample(init_individuals, 2)
            child_data = await mutate_ga(
                p1.system_prompt, p1.user_prompt_template,
                p2.system_prompt, p2.user_prompt_template,
                backend,
            )
            if child_data:
                idx = len(init_individuals)
                ind = Individual(
                    id=f"gen000-{idx:03d}",
                    system_prompt=child_data["system_prompt"],
                    user_prompt_template=child_data["user_prompt_template"],
                    generation=0,
                    parent_ids=[p1.id, p2.id],
                    mutation_type="ga_init",
                )
                fitness = await eval_individual(ind)
                init_individuals.append(ind)
                print(f"  GA init {idx}: {fitness:.1%}")

        pop.current = init_individuals[:pop_size]
        pop.snapshot(0)
        pop.save()
        start_gen = 1

    # 进化循环
    for gen in range(start_gen, n_generations + 1):
        t0 = time.time()
        print(f"\n=== Generation {gen}/{n_generations} ===")

        # 按 fitness 排序当前种群
        pop.current.sort(key=lambda x: x.fitness, reverse=True)
        fitnesses = [ind.fitness for ind in pop.current]
        print(f"  Current best: {fitnesses[0]:.1%}, avg: {sum(fitnesses)/len(fitnesses):.1%}")

        new_children = []
        for j in range(pop_size):
            # 锦标赛选择两个 parent
            group_a = rng.sample(pop.current, min(2, len(pop.current)))
            group_b = rng.sample(pop.current, min(2, len(pop.current)))
            parent1 = max(group_a, key=lambda x: x.fitness)
            parent2 = max(group_b, key=lambda x: x.fitness)

            child_data = await mutate_ga(
                parent1.system_prompt, parent1.user_prompt_template,
                parent2.system_prompt, parent2.user_prompt_template,
                backend,
            )
            if child_data is None:
                # 变异失败，保留较好的 parent
                new_children.append(parent1 if parent1.fitness >= parent2.fitness else parent2)
                continue

            child = Individual(
                id=f"gen{gen:03d}-{j:03d}",
                system_prompt=child_data["system_prompt"],
                user_prompt_template=child_data["user_prompt_template"],
                generation=gen,
                parent_ids=[parent1.id, parent2.id],
                mutation_type="ga_crossover",
            )
            fitness = await eval_individual(child)
            new_children.append(child)
            print(f"  Child {j}: {fitness:.1%} (parents: {parent1.fitness:.1%} + {parent2.fitness:.1%})")

        # topk 选择：合并老种群和新 children，保留 top pop_size
        combined = list(set(
            [(ind.id, ind) for ind in pop.current] +
            [(ind.id, ind) for ind in new_children]
        ))
        all_candidates = [ind for _, ind in combined]
        all_candidates.sort(key=lambda x: x.fitness, reverse=True)
        pop.current = all_candidates[:pop_size]

        elapsed = time.time() - t0
        pop.snapshot(gen)
        pop.save()

        best = pop.get_best()
        print(f"  Gen {gen} done ({elapsed:.0f}s). Best: {best.fitness:.1%} ({best.id})")

    # 最终结果
    print(f"\n{'='*60}")
    print("  Evolution Complete")
    print(f"{'='*60}")
    best = pop.get_best()
    print(f"  Best fitness: {best.fitness:.1%}")
    print(f"  Best ID: {best.id}")
    print(f"  System prompt: {best.system_prompt[:200]}...")
    print(f"  User template: {best.user_prompt_template[:200]}...")
    print(f"  Generations: {pop.generation}")
    print(f"  Total evaluated: {len(pop.all_evaluated)}")

    return best
```

**Step 2: Commit**

```bash
git add src/evolution/runner.py
git commit -m "feat(evolution): GA evolution runner with tournament selection and topk survival"
```

---

### Task 5: CLI 入口

**Files:**
- Create: `scripts/run_evolution.py`

**Step 1: 创建 `scripts/run_evolution.py`**

```python
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
```

**Step 2: Commit**

```bash
git add scripts/run_evolution.py
git commit -m "feat(evolution): CLI entry point for judge prompt evolution"
```

---

### Task 6: 冒烟测试 + 首次运行

**Step 1: 冒烟测试（5 个种子，2 代，50 个样本）**

```bash
MEOMORY_LLM_KEY=${DEEPSEEK_API_KEY} \
.venv/bin/python scripts/run_evolution.py \
  --pop-size 5 --generations 2 --samples 50
```

Expected: 每代 ~2-3 分钟，总共 ~6 分钟。输出应该显示种子评测分数 + 2 代进化。

**Step 2: 验证种群库持久化**

```bash
cat experiments/evolution/best.json | python3 -m json.tool | head -20
cat experiments/evolution/history.json | python3 -m json.tool
```

**Step 3: 正式运行（10 个种群，15 代，200 个样本）**

```bash
MEOMORY_LLM_KEY=${DEEPSEEK_API_KEY} \
.venv/bin/python scripts/run_evolution.py \
  --pop-size 10 --generations 15 --samples 200
```

预估：每代 ~5 分钟（10 个 GA 变异 + 200 样本评测），15 代 ~75 分钟。

**Step 4: 将最优 prompt 回写到 methods.py**

如果进化后的 prompt accuracy > 当前 baseline：
- 更新 `src/bench/methods.py` 中的 `JUDGE_SYSTEM` 和 judge_prompt 模板
- 重跑 `dgd-judge-locomo-full` 和 `dgd-judge-longmemeval` 验证 P@1 提升

**Step 5: Commit**

```bash
git add experiments/evolution/ src/bench/methods.py
git commit -m "feat(evolution): evolved judge prompt with improved feedback accuracy"
```

---

## 预估时间

| Task | 编码 | 测试 | 总计 |
|------|------|------|------|
| 1. 评测函数 | 10min | 5min | 15min |
| 2. 种群库 | 10min | 2min | 12min |
| 3. 变异算子 | 10min | 3min | 13min |
| 4. 进化主循环 | 15min | 5min | 20min |
| 5. CLI 入口 | 5min | 2min | 7min |
| 6. 冒烟+正式运行 | 0 | 80min | 80min |
| **总计** | **50min** | **97min** | **~2.5h** |

编码 50 分钟，正式进化运行 ~75 分钟。
