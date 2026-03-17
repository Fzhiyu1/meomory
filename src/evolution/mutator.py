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
