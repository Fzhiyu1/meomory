"""数据集加载：MemoryBench 各子集 + LoCoMo 完整版 + LongMemEval"""
import ast
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "memorybench"
LOCOMO_DIR = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "locomo"
LONGMEMEVAL_DIR = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "longmemeval"


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

    @classmethod
    def load_locomo_full(cls, index: int) -> "Dataset":
        """加载 LoCoMo 完整数据集（从 locomo10.json）。index = 0~9

        使用 evidence 字段直接映射到 session 索引，避免文本匹配的局限性。
        包含全部 5 类题目（含对抗题 category 5）。
        """
        data_file = LOCOMO_DIR / "locomo10.json"
        with open(data_file) as f:
            all_data = json.load(f)

        item = all_data[index]
        conv = item["conversation"]

        # 提取所有 session（按编号排序）
        session_keys = sorted(
            [k for k in conv if re.match(r"^session_\d+$", k)],
            key=lambda x: int(x.split("_")[1]),
        )

        # 每个 session 作为一个 fragment
        fragments = []
        session_num_to_frag_idx = {}  # session_num → fragment index
        for sk in session_keys:
            session_num = int(sk.split("_")[1])
            turns = conv[sk]
            date_key = f"{sk}_date_time"
            date = conv.get(date_key, "")

            # 构建 session 文本
            lines = [f"[{date}]"] if date else []
            for turn in turns:
                speaker = turn.get("speaker", "?")
                text = turn.get("text", "")
                lines.append(f"[{speaker}] {text}")
            full_text = "\n".join(lines)
            body = full_text[:800].strip()

            if len(body) < 30:
                continue

            frag_idx = len(fragments)
            session_num_to_frag_idx[session_num] = frag_idx
            fragments.append({"id": frag_idx, "body": body, "full": full_text})

        # 提取 QA，用 evidence 字段映射到 fragment
        questions = []
        for qa in item["qa"]:
            # category 5 用 adversarial_answer，其他用 answer
            if qa["category"] == 5:
                answer = qa.get("adversarial_answer", "")
            else:
                answer = qa.get("answer", "")
            if not answer or not isinstance(answer, str):
                continue

            # 解析 evidence → session numbers
            answer_indices = set()
            for ev in qa.get("evidence", []):
                m = re.match(r"D(\d+):", ev)
                if m:
                    s_num = int(m.group(1))
                    if s_num in session_num_to_frag_idx:
                        answer_indices.add(session_num_to_frag_idx[s_num])

            if not answer_indices:
                continue

            questions.append({
                "question": qa["question"][:300],
                "golden_answer": answer,
                "answer_indices": sorted(answer_indices),
                "category": qa["category"],
            })

        return cls(name=f"LoCoMo-full-{index}", fragments=fragments, questions=questions)

    @classmethod
    def load_locomo_full_all(cls) -> "Dataset":
        """加载全部 10 段 LoCoMo 对话，合并为一个大数据集。"""
        all_fragments = []
        all_questions = []
        offset = 0

        for idx in range(10):
            ds = cls.load_locomo_full(idx)
            # 偏移 fragment id 和 answer_indices
            for f in ds.fragments:
                f["id"] = f["id"] + offset
                all_fragments.append(f)
            for q in ds.questions:
                q["answer_indices"] = [i + offset for i in q["answer_indices"]]
                all_questions.append(q)
            offset += len(ds.fragments)

        return cls(name="LoCoMo-full-all", fragments=all_fragments, questions=all_questions)

    @classmethod
    def load_longmemeval(cls, variant: str = "oracle") -> "Dataset":
        """加载 LongMemEval 数据集。

        variant: 'oracle' (仅相关 session) 或 's' (含干扰 session)
        """
        if variant == "oracle":
            data_file = LONGMEMEVAL_DIR / "longmemeval_oracle.json"
        elif variant == "s":
            data_file = LONGMEMEVAL_DIR / "longmemeval_s.json"
        else:
            raise ValueError(f"Unknown LongMemEval variant: {variant}")

        with open(data_file) as f:
            all_items = json.load(f)

        # 收集所有 unique session 作为 fragments
        session_map = {}  # session_id → fragment_index
        fragments = []
        questions = []

        for item in all_items:
            q_type = item.get("question_type", "unknown")
            # 跳过 abstention 类型（以 _abs 结尾）
            if item.get("question_id", "").endswith("_abs"):
                continue

            # 收集 haystack sessions
            for sid, session in zip(
                item.get("haystack_session_ids", []),
                item.get("haystack_sessions", []),
            ):
                if sid not in session_map:
                    # 将 session 转为文本
                    if isinstance(session, list):
                        lines = []
                        for msg in session:
                            role = msg.get("role", "?")
                            content = msg.get("content", "")
                            lines.append(f"[{role}] {content}")
                        full_text = "\n".join(lines)
                    elif isinstance(session, str):
                        full_text = session
                    else:
                        full_text = str(session)

                    frag_idx = len(fragments)
                    session_map[sid] = frag_idx
                    fragments.append({
                        "id": frag_idx,
                        "body": full_text[:800].strip(),
                        "full": full_text,
                    })

            # 构建 question
            answer_indices = []
            for sid in item.get("answer_session_ids", []):
                if sid in session_map:
                    answer_indices.append(session_map[sid])

            if not answer_indices:
                continue

            questions.append({
                "question": item.get("question", "")[:300],
                "golden_answer": item.get("answer", ""),
                "answer_indices": sorted(set(answer_indices)),
                "question_type": q_type,
            })

        return cls(name=f"LongMemEval-{variant}", fragments=fragments, questions=questions)

    def __repr__(self):
        return f"Dataset({self.name}: {len(self.fragments)} fragments, {len(self.questions)} questions)"
