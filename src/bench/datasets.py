"""数据集加载：MemoryBench 各子集"""
import ast
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "memorybench"


class Dataset:
    def __init__(self, name, fragments, questions):
        self.name = name
        self.fragments = fragments  # [{id, body, full}]
        self.questions = questions  # [{question, golden_answer, answer_indices}]

    @classmethod
    def load_dialsim(cls, name: str) -> "Dataset":
        """加载 DialSim 系列。name = 'bigbang' / 'friends' / 'theoffice'"""
        corpus_file = DATA_DIR / f"dialsim-{name}.jsonl"
        test_file = DATA_DIR / f"dialsim-{name}-test.parquet"

        # 加载语料
        with open(corpus_file) as f:
            text = json.loads(f.readline())["text"]
        sessions = text.split("[Date:")
        sessions = ["[Date:" + s.strip() for s in sessions if s.strip()]
        fragments = []
        for i, s in enumerate(sessions):
            body = s[:800].strip()
            if len(body) < 50:
                continue
            fragments.append({"id": i, "body": body, "full": s})

        # 加载测试题
        import pyarrow.parquet as pq

        table = pq.read_table(str(test_file))
        df = table.to_pydict()
        questions = []
        for i in range(len(df["test_idx"])):
            info = df["info"][i]
            if isinstance(info, str):
                info = ast.literal_eval(info)
            prompt = df["input_prompt"][i]
            q_start = prompt.rfind("[Question]")
            a_start = prompt.rfind("[Answer]")
            if q_start >= 0 and a_start >= 0:
                question = prompt[q_start + 10 : a_start].strip()
            else:
                question = prompt[-200:]
            questions.append({
                "question": question[:300],
                "golden_answer": info.get("golden_answer", ""),
            })

        # 找答案所在片段
        for q in questions:
            answer = q["golden_answer"].lower()
            q["answer_indices"] = [i for i, f in enumerate(fragments) if answer in f["full"].lower()]

        # 过滤没找到答案的
        questions = [q for q in questions if q["answer_indices"]]

        return cls(name=f"DialSim-{name}", fragments=fragments, questions=questions)

    def __repr__(self):
        return f"Dataset({self.name}: {len(self.fragments)} fragments, {len(self.questions)} questions)"
