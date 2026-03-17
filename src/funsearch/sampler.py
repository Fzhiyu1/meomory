"""LLM 采样器：构建 FunSearch 风格的版本化 prompt，调用 DeepSeek 生成代码。

v2: 用 AST 提取函数体（替代 regex），为每个 parent 标注分数，
    prompt 末尾加生成指令。
"""
import ast
import re

from src.bench.backends import DeepSeekBackend
from src.funsearch.specification import PROMPT_TEMPLATE, EVOLVE_FUNCTION_NAME
from src.funsearch.database import Program
from src.funsearch.sandbox import extract_function_body


def _extract_function_body_ast(code: str) -> str | None:
    """用 AST 提取函数体文本（比 regex 更 robust）。

    返回函数体的源码文本（含缩进），或 None。
    """
    code = code.strip()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            lines = code.splitlines()
            # 函数体起始行（跳过 def 行）
            body_start = node.body[0].lineno - 1
            body_end = node.end_lineno
            body_lines = lines[body_start:body_end]
            return "\n".join(body_lines)
    return None


def _rename_function_in_code(code: str, new_name: str) -> str:
    """用 AST 安全地重命名函数定义和递归调用。"""
    code = code.strip()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # fallback 到 regex
        return re.sub(r"def \w+\(", f"def {new_name}(", code, count=1)

    old_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            old_name = node.name
            break

    if not old_name:
        return code

    # 替换 def 行
    result = re.sub(rf"\bdef {re.escape(old_name)}\b", f"def {new_name}", code, count=1)
    # 替换函数体内的递归调用（word boundary 防止误替换）
    result = re.sub(rf"\b{re.escape(old_name)}\s*\(", f"{new_name}(", result)
    return result


def build_prompt(parent_programs: list[Program]) -> str:
    """构建 FunSearch 风格的 prompt。

    给 LLM 看 N 个版本的 update 函数（按分数从低到高排列），
    每个版本标注分数，让它生成改进版 v_next。
    """
    if not parent_programs:
        raise ValueError("Need at least one parent program")

    # 按分数排序（低->高），最好的在最后
    sorted_parents = sorted(parent_programs, key=lambda p: p.score)

    versioned_functions = []
    for i, parent in enumerate(sorted_parents):
        code = parent.code.strip()
        func_name = f"{EVOLVE_FUNCTION_NAME}_v{i}"

        # 用 AST 重命名函数
        renamed = _rename_function_in_code(code, func_name)

        # 为每个版本添加分数标注（docstring 风格）
        body = _extract_function_body_ast(renamed)
        if body is not None:
            # 构建带分数 docstring 的完整函数
            score_pct = f"{parent.score:.1%}"
            docstring = f'    """Version {i}, score: {score_pct}."""\n'
            # 找到 def 行之后插入 docstring
            lines = renamed.splitlines()
            def_line_idx = None
            for li, line in enumerate(lines):
                if line.strip().startswith("def "):
                    def_line_idx = li
                    break
            if def_line_idx is not None:
                lines.insert(def_line_idx + 1, docstring.rstrip())
                renamed = "\n".join(lines)

        versioned_functions.append(renamed)

    versioned_text = "\n\n".join(versioned_functions)
    next_version = len(sorted_parents)
    next_name = f"{EVOLVE_FUNCTION_NAME}_v{next_version}"
    prev_name = f"{EVOLVE_FUNCTION_NAME}_v{next_version - 1}"

    prompt = PROMPT_TEMPLATE.format(
        versioned_functions=versioned_text,
        next_function_name=next_name,
        prev_function_name=prev_name,
    )

    # v2: 在 prompt 末尾加明确的生成指令
    prompt += "\n    # Output ONLY the function body, no explanation, no markdown\n"

    return prompt


async def sample_new_program(
    parent_programs: list[Program],
    backend: DeepSeekBackend,
    island_id: int,
    generation: int,
    index: int,
) -> Program | None:
    """调用 LLM 生成一个新的 update 函数。

    Returns:
        Program 或 None（生成失败）
    """
    prompt = build_prompt(parent_programs)

    try:
        raw = await backend.chat(prompt, max_tokens=1200)
    except Exception:
        return None

    # 提取代码：优先提取完整类定义，fallback 到函数体
    full_code = None
    raw_stripped = raw.strip()

    # 策略 1: 去掉 markdown 代码块
    import re
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", raw_stripped, re.DOTALL)
    text = code_blocks[0].strip() if code_blocks else raw_stripped

    # 策略 2: 直接找 class 定义
    class_match = re.search(r"(class \w+[\s\S]*)", text)
    if class_match:
        cls_code = class_match.group(1).strip()
        # 验证能解析
        import ast
        try:
            ast.parse(cls_code)
            full_code = cls_code
        except SyntaxError:
            # 逐行删末尾直到能解析
            lines = cls_code.splitlines()
            while lines:
                try:
                    ast.parse("\n".join(lines))
                    full_code = "\n".join(lines)
                    break
                except SyntaxError:
                    lines = lines[:-1]

    # 策略 3: fallback 到函数体提取
    if full_code is None:
        body = extract_function_body(raw)
        if body is not None:
            if "def __init__" in body or "def query" in body:
                full_code = f"class AssociativeMemory:\n{body}"
            else:
                full_code = f"def dgd_update(M, key, target, dim, alpha, eta):\n{body}"

    if full_code is None:
        return None

    return Program(
        id=f"gen{generation:03d}-island{island_id}-{index:03d}",
        code=full_code,
        island_id=island_id,
        generation=generation,
        parent_ids=[p.id for p in parent_programs],
    )
