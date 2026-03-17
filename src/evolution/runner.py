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
            "- 只有回答中明确使用了记忆中的具体信息（人名、日期、事件）才算「引用」\n"
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
