"""进化主循环 v2：直接测 relevance，不模拟 agent。"""
import asyncio
import random
import time

from src.bench.backends import DeepSeekBackend
from src.evolution.evaluator import build_eval_samples
from src.evolution.evaluator_v2 import evaluate_judge_prompt_v2
from src.evolution.population import Population, Individual
from src.evolution.mutator import mutate_ga


# v2 种子：直接问 relevance，不问 usage
SEED_PROMPTS_V2 = [
    {
        "system": (
            "You are a relevance judge. Given a question and memory fragments, "
            "determine which fragments contain information needed to correctly answer the question.\n"
            "Rules:\n"
            "- A fragment is relevant ONLY if it contains specific facts (names, dates, events) that directly answer the question\n"
            "- General topic overlap is NOT enough\n"
            "- Output ONLY a JSON array of indices (0-based). Output [] if none are relevant."
        ),
        "user": (
            "Question: {question}\n\n"
            "Memory fragments:\n{mem_list}\n\n"
            "Which fragments contain information needed to answer the question? JSON array only."
        ),
    },
    {
        "system": (
            "Determine which memory fragments are relevant to answering the given question.\n"
            "A fragment is relevant if it contains the specific answer or key facts needed.\n"
            "Be strict: topical similarity alone does not count.\n"
            "Output a JSON array of 0-based indices."
        ),
        "user": (
            "Question: {question}\n\n"
            "Fragments:\n{mem_list}\n\n"
            "Relevant fragment indices (JSON array):"
        ),
    },
    {
        "system": (
            "You are checking memory retrieval quality. For each fragment, decide: "
            "does this fragment contain the actual answer to the question?\n"
            "Only mark fragments that contain specific, verifiable information matching the question.\n"
            "Output: JSON array of indices, e.g. [0] or [1,2] or []."
        ),
        "user": (
            "Question: {question}\n\n"
            "Retrieved fragments:\n{mem_list}\n\n"
            "Which fragments contain the answer? JSON array:"
        ),
    },
    {
        "system": (
            "判断哪些记忆片段包含回答问题所需的信息。\n"
            "规则：\n"
            "- 只有包含具体事实（人名、日期、事件、数字）能回答问题的片段才算相关\n"
            "- 话题相关但不含答案的不算\n"
            "- 输出 JSON 数组，如 [0, 2]。都不相关输出 []"
        ),
        "user": (
            "问题: {question}\n\n"
            "记忆片段:\n{mem_list}\n\n"
            "哪些片段包含答案？JSON 数组:"
        ),
    },
    {
        "system": (
            "For each memory fragment, answer: Could someone use ONLY this fragment to answer the question?\n"
            "If yes, include its index. If the fragment is merely related but doesn't contain the answer, exclude it.\n"
            "Output format: JSON array of indices."
        ),
        "user": (
            "Question: {question}\n\n"
            "Memory fragments:\n{mem_list}\n\n"
            "Indices of fragments that contain the answer:"
        ),
    },
]


# GA template for v2 (no {response} placeholder)
GA_TEMPLATE_V2 = """You are an expert prompt engineer. Create a better judge prompt by combining two existing prompts.

## Parent 1 (system):
{parent1_system}

## Parent 1 (user template):
{parent1_user}

## Parent 2 (system):
{parent2_system}

## Parent 2 (user template):
{parent2_user}

## Task context:
The judge must determine which memory fragments are RELEVANT to answering a question.
A good judge prompt should make the LLM correctly identify fragments containing the actual answer.

## Output (strict JSON):
{{"system_prompt": "...", "user_prompt_template": "..."}}

The user_prompt_template MUST contain: {{mem_list}}, {{question}}
Output ONLY the JSON."""


async def _mutate_ga_v2(p1, p2, backend):
    prompt = GA_TEMPLATE_V2.format(
        parent1_system=p1.system_prompt, parent1_user=p1.user_prompt_template,
        parent2_system=p2.system_prompt, parent2_user=p2.user_prompt_template,
    )
    try:
        raw = await backend.chat(prompt, max_tokens=500)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        import json
        d = json.loads(text)
        if "system_prompt" in d and "user_prompt_template" in d:
            if "{mem_list}" in d["user_prompt_template"] and "{question}" in d["user_prompt_template"]:
                return d
    except Exception:
        pass
    return None


async def run_evolution_v2(
    pop_size: int = 10,
    n_generations: int = 15,
    n_eval_samples: int = 200,
    eval_datasets: list[str] = None,
    api_key: str = "",
    seed: int = 42,
    data_dir: str = "experiments/evolution-v2",
):
    """v2 进化：直接测 relevance，不模拟 agent。"""
    if eval_datasets is None:
        eval_datasets = ["LoCoMo-full-all", "LongMemEval-oracle"]

    backend = DeepSeekBackend(api_key=api_key)
    pop = Population(data_dir=data_dir)
    rng = random.Random(seed)

    print("Building evaluation samples...")
    all_samples = {}
    per_ds = n_eval_samples // len(eval_datasets)
    for ds_name in eval_datasets:
        all_samples[ds_name] = build_eval_samples(ds_name, n_samples=per_ds, seed=seed)
        print(f"  {ds_name}: {len(all_samples[ds_name])} samples")

    combined_samples = []
    for samples in all_samples.values():
        combined_samples.extend(samples)

    async def eval_individual(ind):
        result = await evaluate_judge_prompt_v2(
            ind.system_prompt, ind.user_prompt_template,
            combined_samples, backend, concurrency=30,
        )
        ind.eval_details = result
        ind.fitness = result["accuracy"]
        pop.add(ind)
        return ind.fitness

    start_gen = pop.generation
    if pop.current and len(pop.current) >= pop_size:
        print(f"Resuming from generation {start_gen}")
    else:
        print(f"Initializing {len(SEED_PROMPTS_V2)} seeds...")
        init_individuals = []

        for i, sp in enumerate(SEED_PROMPTS_V2[:pop_size]):
            ind = Individual(
                id=f"gen000-{i:03d}", system_prompt=sp["system"],
                user_prompt_template=sp["user"], generation=0, mutation_type="seed",
            )
            fitness = await eval_individual(ind)
            init_individuals.append(ind)
            print(f"  Seed {i}: {fitness:.1%}")

        while len(init_individuals) < pop_size:
            p1, p2 = rng.sample(init_individuals, 2)
            child_data = await _mutate_ga_v2(p1, p2, backend)
            if child_data:
                idx = len(init_individuals)
                ind = Individual(
                    id=f"gen000-{idx:03d}", system_prompt=child_data["system_prompt"],
                    user_prompt_template=child_data["user_prompt_template"],
                    generation=0, parent_ids=[p1.id, p2.id], mutation_type="ga_init",
                )
                fitness = await eval_individual(ind)
                init_individuals.append(ind)
                print(f"  GA init {idx}: {fitness:.1%}")

        pop.current = init_individuals[:pop_size]
        pop.snapshot(0)
        pop.save()
        start_gen = 1

    for gen in range(start_gen, n_generations + 1):
        t0 = time.time()
        print(f"\n=== Generation {gen}/{n_generations} ===")

        pop.current.sort(key=lambda x: x.fitness, reverse=True)
        fitnesses = [ind.fitness for ind in pop.current]
        print(f"  Best: {fitnesses[0]:.1%}, avg: {sum(fitnesses)/len(fitnesses):.1%}")

        new_children = []
        for j in range(pop_size):
            group_a = rng.sample(pop.current, min(2, len(pop.current)))
            group_b = rng.sample(pop.current, min(2, len(pop.current)))
            parent1 = max(group_a, key=lambda x: x.fitness)
            parent2 = max(group_b, key=lambda x: x.fitness)

            child_data = await _mutate_ga_v2(parent1, parent2, backend)
            if child_data is None:
                new_children.append(parent1 if parent1.fitness >= parent2.fitness else parent2)
                continue

            child = Individual(
                id=f"gen{gen:03d}-{j:03d}",
                system_prompt=child_data["system_prompt"],
                user_prompt_template=child_data["user_prompt_template"],
                generation=gen, parent_ids=[parent1.id, parent2.id],
                mutation_type="ga_crossover",
            )
            fitness = await eval_individual(child)
            new_children.append(child)
            print(f"  Child {j}: {fitness:.1%}")

        all_candidates = {ind.id: ind for ind in pop.current}
        for ind in new_children:
            all_candidates[ind.id] = ind
        sorted_all = sorted(all_candidates.values(), key=lambda x: x.fitness, reverse=True)
        pop.current = sorted_all[:pop_size]

        elapsed = time.time() - t0
        pop.snapshot(gen)
        pop.save()
        best = pop.get_best()
        print(f"  Gen {gen} done ({elapsed:.0f}s). Best: {best.fitness:.1%} ({best.id})")

    print(f"\n{'='*60}")
    print("  Evolution v2 Complete")
    print(f"{'='*60}")
    best = pop.get_best()
    print(f"  Best fitness: {best.fitness:.1%}")
    print(f"  Best ID: {best.id}")
    print(f"  System: {best.system_prompt[:200]}...")
    print(f"  User: {best.user_prompt_template[:200]}...")
    print(f"  Total evaluated: {len(pop.all_evaluated)}")
    return best
