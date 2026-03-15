"""检索方法：cosine / BM25 / DGD+GT / DGD+Judge"""
import json
import math
import re

from src.dgd import AssociativeMemory

JUDGE_SYSTEM = """You are a judge. Given injected memories and an agent's response, determine which memories the response actually referenced or used.
Rules:
- Only mark a memory as "used" if the response clearly draws on information from that memory
- Paraphrasing counts as usage
- Mentioning the same topic is NOT enough
- Output ONLY a JSON array of indices (0-based), e.g. [0, 2]. Output [] if none were used."""


def _cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na * nb > 0 else 0


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


class Method:
    name: str

    def setup(self, proj_vecs):
        pass

    def query(self, q_vec) -> list[tuple[float, int]]:
        """返回 [(score, frag_idx)] 降序"""
        raise NotImplementedError

    async def feedback(self, q_vec, top_indices, answer_indices, fragments, proj_vecs, backend, question_text="") -> dict:
        """返回 {updates: int, correct: int}"""
        return {"updates": 0, "correct": 0}


class CosineMethod(Method):
    name = "cosine"

    def setup(self, proj_vecs):
        self.proj_vecs = proj_vecs

    def query(self, q_vec):
        scores = [(_cosine(q_vec, pv), i) for i, pv in enumerate(self.proj_vecs)]
        scores.sort(reverse=True)
        return scores


class BM25Method(Method):
    """简单 BM25 实现"""

    name = "bm25"

    def __init__(self):
        self.k1 = 1.5
        self.b = 0.75

    def setup_bm25(self, fragments):
        """用文档文本建 BM25 索引"""
        self.docs = []
        self.doc_lens = []
        self.df = {}  # document frequency
        self.n = len(fragments)

        for frag in fragments:
            tokens = self._tokenize(frag["body"])
            self.docs.append(tokens)
            self.doc_lens.append(len(tokens))
            seen = set()
            for t in tokens:
                if t not in seen:
                    self.df[t] = self.df.get(t, 0) + 1
                    seen.add(t)

        self.avgdl = sum(self.doc_lens) / self.n if self.n > 0 else 1

    def _tokenize(self, text):
        # 简单分词：英文按空格/标点，中文按字符
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*|[\u4e00-\u9fff]", text.lower())
        return tokens

    def query_bm25(self, query_text):
        q_tokens = self._tokenize(query_text)
        scores = []
        for idx, doc in enumerate(self.docs):
            score = 0
            dl = self.doc_lens[idx]
            doc_tf = {}
            for t in doc:
                doc_tf[t] = doc_tf.get(t, 0) + 1
            for qt in q_tokens:
                if qt not in self.df:
                    continue
                tf = doc_tf.get(qt, 0)
                idf = math.log((self.n - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                score += idf * tf_norm
            scores.append((score, idx))
        scores.sort(reverse=True)
        return scores

    # query 接口需要 q_vec 但 BM25 不用向量，需要特殊处理
    # 在 runner 中会特殊调用 query_bm25
    def setup(self, proj_vecs):
        self.proj_vecs = proj_vecs

    def query(self, q_vec):
        # fallback to cosine if called with vector
        scores = [(_cosine(q_vec, pv), i) for i, pv in enumerate(self.proj_vecs)]
        scores.sort(reverse=True)
        return scores


class DGDMethod(Method):
    """DGD 基类"""

    def __init__(self, dim=256, alpha=1.0, eta=0.01):
        self.dim = dim
        self.alpha = alpha
        self.eta = eta
        self.mem = None

    def setup(self, proj_vecs):
        self.proj_vecs = proj_vecs
        self.mem = AssociativeMemory(dim=self.dim, alpha=self.alpha, eta=self.eta)

    def query(self, q_vec):
        act = self.mem.query(q_vec)
        scores = [(_cosine(act, pv), i) for i, pv in enumerate(self.proj_vecs)]
        scores.sort(reverse=True)
        return scores


class DGDGroundTruth(DGDMethod):
    name = "dgd_gt"

    async def feedback(self, q_vec, top_indices, answer_indices, fragments, proj_vecs, backend, question_text=""):
        target = answer_indices[0]
        self.mem.update(q_vec, proj_vecs[target])
        return {"updates": 1, "correct": 1}


class DGDJudge(DGDMethod):
    name = "dgd_judge"

    async def feedback(self, q_vec, top_indices, answer_indices, fragments, proj_vecs, backend, question_text=""):
        if backend is None:
            return {"updates": 0, "correct": 0}

        top_3 = top_indices[:3]
        top_bodies = [fragments[idx]["body"][:200] for idx in top_3]

        # Agent 回答（传入实际问题）
        injection = "\n".join(f"- {b}" for b in top_bodies)
        prompt = f"[Context]\n{injection}\n\n[Question] {question_text}\n\nAnswer briefly based on context."
        try:
            response = await backend.chat(prompt, max_tokens=200)
        except Exception:
            return {"updates": 0, "correct": 0, "error": True}

        # Judge 判断
        mem_list = "\n".join(f"[{i}] {b}" for i, b in enumerate(top_bodies))
        judge_prompt = (
            f"Injected memories:\n{mem_list}\n\nAgent response:\n{response[:500]}\n\n"
            f"Which memories were used? Output JSON array only."
        )
        try:
            raw = await backend.chat(judge_prompt, system=JUDGE_SYSTEM, max_tokens=50)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            used = json.loads(text)
        except Exception:
            return {"updates": 0, "correct": 0}

        answer_set = set(answer_indices)
        updates = 0
        correct = 0
        for local_idx in used:
            if 0 <= local_idx < len(top_3):
                frag_idx = top_3[local_idx]
                self.mem.update(q_vec, proj_vecs[frag_idx])
                updates += 1
                if frag_idx in answer_set:
                    correct += 1

        return {"updates": updates, "correct": correct}


class DGDCMSJudge(Method):
    """CMS 三层 DGD + LLM Judge：不同遗忘率的三个 M 矩阵并行查询"""
    name = "dgd_cms_judge"

    def __init__(self, dim=256, eta=0.01):
        self.dim = dim
        self.eta = eta
        # 三层：高频（快忘）、中频（慢忘）、低频（几乎不忘）
        self.layers = {
            "high": {"alpha": 0.90, "weight": 0.5, "mem": None},
            "mid":  {"alpha": 0.99, "weight": 0.3, "mem": None},
            "low":  {"alpha": 0.999, "weight": 0.2, "mem": None},
        }

    def setup(self, proj_vecs):
        self.proj_vecs = proj_vecs
        for layer in self.layers.values():
            layer["mem"] = AssociativeMemory(dim=self.dim, alpha=layer["alpha"], eta=self.eta)

    def query(self, q_vec):
        # 三层并行查询，加权合并
        combined = [0.0] * self.dim
        for layer in self.layers.values():
            act = layer["mem"].query(q_vec)
            w = layer["weight"]
            for j in range(self.dim):
                combined[j] += w * act[j]

        scores = [(_cosine(combined, pv), i) for i, pv in enumerate(self.proj_vecs)]
        scores.sort(reverse=True)
        return scores

    async def feedback(self, q_vec, top_indices, answer_indices, fragments, proj_vecs, backend, question_text=""):
        if backend is None:
            return {"updates": 0, "correct": 0}

        top_3 = top_indices[:3]
        top_bodies = [fragments[idx]["body"][:200] for idx in top_3]

        # Agent 回答
        injection = "\n".join(f"- {b}" for b in top_bodies)
        prompt = f"[Context]\n{injection}\n\n[Question] {question_text}\n\nAnswer briefly based on context."
        try:
            response = await backend.chat(prompt, max_tokens=200)
        except Exception:
            return {"updates": 0, "correct": 0, "error": True}

        # Judge 判断
        mem_list = "\n".join(f"[{i}] {b}" for i, b in enumerate(top_bodies))
        judge_prompt = (
            f"Injected memories:\n{mem_list}\n\nAgent response:\n{response[:500]}\n\n"
            f"Which memories were used? Output JSON array only."
        )
        try:
            raw = await backend.chat(judge_prompt, system=JUDGE_SYSTEM, max_tokens=50)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            used = json.loads(text)
        except Exception:
            return {"updates": 0, "correct": 0}

        answer_set = set(answer_indices)
        updates = 0
        correct = 0
        for local_idx in used:
            if 0 <= local_idx < len(top_3):
                frag_idx = top_3[local_idx]
                # 三层都更新（但因为 α 不同，遗忘速度不同）
                for layer in self.layers.values():
                    layer["mem"].update(q_vec, proj_vecs[frag_idx])
                updates += 1
                if frag_idx in answer_set:
                    correct += 1

        return {"updates": updates, "correct": correct}


def create_method(config: dict) -> Method:
    name = config["method"]
    dgd_cfg = config.get("dgd", {})
    if name == "cosine":
        return CosineMethod()
    elif name == "bm25":
        return BM25Method()
    elif name == "dgd_gt":
        return DGDGroundTruth(
            dim=dgd_cfg.get("dim", 256),
            alpha=dgd_cfg.get("alpha", 1.0),
            eta=dgd_cfg.get("eta", 0.01),
        )
    elif name == "dgd_llm_judge":
        return DGDJudge(
            dim=dgd_cfg.get("dim", 256),
            alpha=dgd_cfg.get("alpha", 1.0),
            eta=dgd_cfg.get("eta", 0.01),
        )
    elif name == "dgd_cms_judge":
        return DGDCMSJudge(
            dim=dgd_cfg.get("dim", 256),
            eta=dgd_cfg.get("eta", 0.01),
        )
    else:
        raise ValueError(f"Unknown method: {name}")
