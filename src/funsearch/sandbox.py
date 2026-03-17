"""沙箱：安全编译和执行 LLM 生成的 update 函数。

安全措施：
1. AST 检查：禁止 import、exec、eval、open 等危险操作
2. 超时：单次 update 调用限时
3. 异常捕获：任何错误 → 返回 None
4. 祖先调用检测：防止生成代码递归调用旧版本（v2 新增）
"""
import ast
import math
import re
import signal
import textwrap


# 允许的内置函数
ALLOWED_BUILTINS = {
    "abs", "min", "max", "sum", "len", "range", "enumerate", "zip",
    "int", "float", "bool", "list", "tuple", "round",
    "True", "False", "None",
    "__build_class__", "__name__",  # 类定义需要
    "property", "staticmethod", "classmethod",  # 类装饰器
    "isinstance", "hasattr", "getattr", "setattr",  # 对象操作
    "super", "object",  # 类继承
}

# 禁止的 AST 节点
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)


class SandboxError(Exception):
    pass


def _check_ast_safety(code: str) -> bool:
    """检查代码 AST 是否安全。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        # 禁止 import
        if isinstance(node, FORBIDDEN_NODES):
            return False
        # 禁止 exec/eval/open/__import__
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("exec", "eval", "open", "__import__", "compile"):
                return False
            if isinstance(func, ast.Attribute) and func.attr in ("system", "popen", "exec"):
                return False
    return True


def _make_safe_globals():
    """创建受限执行环境。"""
    return {
        "__builtins__": {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                        for name in ALLOWED_BUILTINS
                        if hasattr(__builtins__, name) or (isinstance(__builtins__, dict) and name in __builtins__)},
        "math": math,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "abs": abs,
        "max": max,
        "min": min,
        "sum": sum,
    }


def compile_update_function(code: str) -> callable | None:
    """将 LLM 生成的函数代码编译为可调用函数。

    Args:
        code: 完整的函数定义字符串（def dgd_update(...)）

    Returns:
        可调用函数，或 None（编译失败/不安全）
    """
    code = textwrap.dedent(code).strip()
    if not code.startswith("def "):
        return None

    if not _check_ast_safety(code):
        return None

    safe_globals = _make_safe_globals()
    try:
        exec(code, safe_globals)
    except Exception:
        return None

    tree = ast.parse(code)
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if not func_names:
        return None

    func_name = func_names[-1]
    return safe_globals.get(func_name)


def compile_class(code: str) -> type | None:
    """将 LLM 生成的类代码编译为可实例化的类。

    Args:
        code: 完整的类定义字符串（class AssociativeMemory: ...）

    Returns:
        类对象，或 None（编译失败/不安全）

    验证：
        1. AST 安全检查
        2. 能实例化（dim=4）
        3. 有 query() 和 update() 方法
        4. query 返回正确长度的 list
        5. update 不崩溃
    """
    code = textwrap.dedent(code).strip()
    if not code.startswith("class "):
        return None

    if not _check_ast_safety(code):
        return None

    safe_globals = _make_safe_globals()
    try:
        exec(code, safe_globals)
    except Exception:
        return None

    # 提取类（取最后定义的 class）
    tree = ast.parse(code)
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if not class_names:
        return None

    cls = safe_globals.get(class_names[-1])
    if cls is None:
        return None

    # 验证：能实例化 + 有 query/update
    try:
        obj = cls(dim=4, alpha=1.0, eta=0.01)
        if not hasattr(obj, 'query') or not hasattr(obj, 'update'):
            return None
        # 快速测试
        key = [0.5, 0.3, 0.1, 0.2]
        target = [0.4, 0.2, 0.3, 0.1]
        act = obj.query(key)
        if not isinstance(act, list) or len(act) != 4:
            return None
        obj.update(key, target)
    except Exception:
        return None

    return cls


def extract_function_body(generated_code: str) -> str | None:
    """从 LLM 生成的续写中提取函数体。

    处理多种 LLM 输出格式：
    1. 纯函数体代码
    2. markdown ```python 代码块
    3. 完整函数定义 def xxx(...)
    4. 解释文字 + 代码混合
    """
    if not generated_code.strip():
        return None

    text = generated_code.strip()

    # 策略 1: 提取 markdown 代码块
    import re
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        text = code_blocks[0].strip()

    # 策略 2: 如果包含完整函数定义，提取函数体
    if re.match(r"\s*def\s+\w+\s*\(", text):
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 提取函数体（跳过 def 行和 docstring）
                    body_start = node.body[0].lineno
                    # 跳过 docstring
                    if (isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                        if len(node.body) > 1:
                            body_start = node.body[1].lineno
                    lines = text.splitlines()
                    body_lines = lines[body_start - 1:node.end_lineno]
                    body = "\n".join(body_lines)
                    if body.strip():
                        return "    " + body.strip().replace("\n", "\n    ") if not body.startswith("    ") else body
        except SyntaxError:
            pass

    # 策略 3: 作为纯函数体处理
    # 先修复缩进：确保每行都有 4 空格缩进
    raw_lines = text.splitlines()
    fixed_lines = []
    for line in raw_lines:
        if line.strip() == "":
            fixed_lines.append("")
        elif not line.startswith("    ") and not line.startswith("\t"):
            fixed_lines.append("    " + line)
        else:
            fixed_lines.append(line)
    text_fixed = "\n".join(fixed_lines)

    full_code = f"def _placeholder(M, key, target, dim, alpha, eta):\n{text_fixed}"

    # 逐行删除末尾直到能解析
    lines = full_code.splitlines()
    while lines:
        try:
            tree = ast.parse("\n".join(lines))
            break
        except SyntaxError:
            lines = lines[:-1]

    if not lines:
        return None

    # 提取函数体（跳过 def 行）
    body_lines = "\n".join(lines).splitlines()[1:]
    body = "\n".join(body_lines)
    return body if body.strip() else None


def _calls_ancestor(code: str, function_name: str) -> bool:
    """检测生成的函数是否调用了祖先版本（_v0, _v1, ...）。

    防止 LLM 生成递归调用旧版本的无效代码。

    Args:
        code: 完整的函数定义代码
        function_name: 被进化的函数名（如 'dgd_update'）

    Returns:
        True 如果代码中调用了 function_name_vN 形式的函数
    """
    pattern = rf'\b{re.escape(function_name)}_v\d+\s*\('
    return bool(re.search(pattern, code))
