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

    @classmethod
    def load_locomo(cls, index: int) -> "Dataset":
        """加载 LoCoMo 系列。index = 0~9"""
        import pyarrow.parquet as pq

        test_file = DATA_DIR / f"locomo-{index}-test.parquet"
        train_file = DATA_DIR / f"locomo-{index}-train.parquet"

        # 合并 test + train 的题目
        all_questions = []
        corpus_text = None

        for path in [test_file, train_file]:
            if not path.exists():
                continue
            table = pq.read_table(str(path))
            df = table.to_pydict()

            for i in range(len(df["test_idx"])):
                # 提取 corpus（从第一个 dialog 字段获取）
                if corpus_text is None:
                    for key in df:
                        if key.startswith("dialog_") and df[key][i]:
                            d = df[key][i]
                            if isinstance(d, str):
                                try:
                                    d = json.loads(d)
                                except Exception:
                                    d = ast.literal_eval(d)
                            if isinstance(d, list) and d:
                                content = d[0].get("content", "") if isinstance(d[0], dict) else ""
                                if len(content) > 1000:
                                    corpus_text = content
                                    break

                # 提取问题
                oq = df.get("origin_question", [None])[i]
                if not oq:
                    continue

                info = df["info"][i]
                if isinstance(info, str):
                    try:
                        info = json.loads(info)
                    except Exception:
                        try:
                            info = ast.literal_eval(info)
                        except Exception:
                            info = {}

                ga = info.get("golden_answer", "")
                # LoCoMo 的 golden_answer 有时是 dict（category 5 = 没答案）
                if isinstance(ga, dict):
                    continue
                if not ga or not isinstance(ga, str):
                    continue

                all_questions.append({
                    "question": oq[:300],
                    "golden_answer": ga,
                })

        if not corpus_text:
            raise ValueError(f"No corpus found for LoCoMo-{index}")

        # 按对话切分 corpus 为片段
        # 格式：Coversation [date]:\nSpeaker Xsays: ...
        import re
        convs = re.split(r'(?=Coversation \[)', corpus_text)
        convs = [c.strip() for c in convs if c.strip() and len(c.strip()) > 50]

        fragments = []
        for i, c in enumerate(convs):
            body = c[:800].strip()
            fragments.append({"id": i, "body": body, "full": c})

        # 找答案所在片段
        for q in all_questions:
            answer = q["golden_answer"].lower()
            q["answer_indices"] = [i for i, f in enumerate(fragments) if answer in f["full"].lower()]

        # 过滤没找到答案的
        all_questions = [q for q in all_questions if q["answer_indices"]]

        return cls(name=f"LoCoMo-{index}", fragments=fragments, questions=all_questions)

    def __repr__(self):
        return f"Dataset({self.name}: {len(self.fragments)} fragments, {len(self.questions)} questions)"
