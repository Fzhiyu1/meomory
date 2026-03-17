"""群岛数据库：FunSearch 风格的岛模型。

v2: 使用 Signature（多测试用例分数向量）分簇，对齐 Google DeepMind 原版设计。
包含 Cluster 机制（按 Signature 分簇）和代码长度偏好。
"""
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---- 类型别名 ----
Signature = tuple[float, ...]
ScoresPerTest = dict[str, float]


@dataclass
class Program:
    """一个进化出的 update 函数。"""
    id: str
    code: str                  # 完整函数定义代码
    score: float = 0.0         # reduced score（overall P@1）
    generation: int = 0
    island_id: int = 0
    parent_ids: list[str] = field(default_factory=list)
    eval_details: dict = field(default_factory=dict)
    scores_per_test: dict[str, float] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def code_length(self) -> int:
        return len(self.code)


# ---- 辅助函数 ----

def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """多测试用例分数向量，作为 Cluster key。"""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """取最后一个 key 的分数作为单一 score（用于排序）。

    约定：scores_per_test 中最后一个 key 是 'overall'。
    """
    keys = list(scores_per_test.keys())
    if not keys:
        return 0.0
    return scores_per_test[keys[-1]]


def _softmax(logits: list[float], temperature: float) -> list[float]:
    """Temperature-scaled softmax（纯 Python 实现，零依赖）。"""
    if not logits:
        return []
    t = max(temperature, 1e-8)
    max_l = max(logits)
    exps = [math.exp((l - max_l) / t) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


class Cluster:
    """Signature 相同的程序簇。同一 Signature 意味着在所有测试用例上行为一致。"""

    def __init__(self, score: float, program: Program):
        self.score = score  # reduced score，用于簇间排序
        self.programs: list[Program] = [program]

    def register(self, program: Program):
        self.programs.append(program)

    def sample_program(self) -> Program:
        """采样一个程序，偏好更短的代码（防止膨胀）。"""
        if len(self.programs) == 1:
            return self.programs[0]

        lengths = [p.code_length for p in self.programs]
        min_len = min(lengths)
        max_len = max(lengths) + 1
        # 归一化长度，取负（越短概率越高）
        normalized = [(l - min_len) / (max_len - min_len) for l in lengths]
        probs = _softmax([-n for n in normalized], temperature=1.0)

        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return self.programs[i]
        return self.programs[-1]


class Island:
    """一个子种群，包含多个 Cluster（按 Signature 分簇）。"""

    def __init__(self, island_id: int, temperature_init: float = 0.1,
                 temperature_period: int = 30_000):
        self.island_id = island_id
        self.clusters: dict[Signature, Cluster] = {}  # signature -> Cluster
        self.best_score: float = -1.0
        self.best_program: Program | None = None
        self.best_scores_per_test: ScoresPerTest | None = None
        self._num_programs = 0
        self._temperature_init = temperature_init
        self._temperature_period = temperature_period

    def register(self, program: Program, scores_per_test: ScoresPerTest):
        """注册程序，用 Signature 分簇。"""
        signature = _get_signature(scores_per_test)
        score = _reduce_score(scores_per_test)

        if signature not in self.clusters:
            self.clusters[signature] = Cluster(score, program)
        else:
            self.clusters[signature].register(program)

        self._num_programs += 1
        if score > self.best_score:
            self.best_score = score
            self.best_program = program
            self.best_scores_per_test = dict(scores_per_test)

    def sample(self, n: int = 2) -> list[Program]:
        """从岛中采样 n 个程序。

        两层采样（和 FunSearch 一致）：
        1. 按簇分数 softmax 选簇（偏好高分簇，但也探索低分簇）
        2. 在簇内偏好短代码
        """
        if not self.clusters:
            return []

        signatures = list(self.clusters.keys())
        cluster_scores = [self.clusters[sig].score for sig in signatures]

        # 温度周期性衰减（对齐 Google：纯线性到 0，仅防除零）
        period = self._temperature_period
        progress = (self._num_programs % period) / period
        temperature = self._temperature_init * (1 - progress)
        temperature = max(temperature, 1e-8)  # 仅防除零

        probs = _softmax(cluster_scores, temperature)

        results = []
        for _ in range(min(n, len(signatures))):
            # 按概率选簇
            r = random.random()
            cumsum = 0
            chosen_idx = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    chosen_idx = i
                    break
            cluster = self.clusters[signatures[chosen_idx]]
            # 在簇内偏好短代码
            results.append(cluster.sample_program())

        return results


class ProgramsDatabase:
    """群岛数据库。"""

    def __init__(
        self,
        num_islands: int = 5,
        reset_interval: int = 50,
        data_dir: str = "experiments/funsearch",
    ):
        self.num_islands = num_islands
        self.reset_interval = reset_interval
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.islands: list[Island] = [Island(i) for i in range(num_islands)]
        self.all_programs: dict[str, Program] = {}
        self.history: list[dict] = []
        self._register_count = 0

        self._load()

    def _load(self):
        """从磁盘恢复。"""
        pop_file = self.data_dir / "population.json"
        if pop_file.exists():
            with open(pop_file) as f:
                data = json.load(f)
            for pd in data.get("all_programs", []):
                prog = Program(**{k: v for k, v in pd.items() if k in Program.__dataclass_fields__})
                self.all_programs[prog.id] = prog
                # 恢复时用 program 中保存的 scores_per_test
                spt = prog.scores_per_test if prog.scores_per_test else {"overall": prog.score}
                if prog.island_id < self.num_islands:
                    self.islands[prog.island_id].register(prog, spt)
            self._register_count = data.get("register_count", 0)

        hist_file = self.data_dir / "history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                self.history = json.load(f)

    def save(self):
        """持久化。"""
        with open(self.data_dir / "population.json", "w") as f:
            json.dump({
                "all_programs": [asdict(p) for p in self.all_programs.values()],
                "register_count": self._register_count,
            }, f, ensure_ascii=False, indent=2)

        with open(self.data_dir / "history.json", "w") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        # Best top-10（综合分数和代码长度排序）
        top10 = sorted(
            self.all_programs.values(),
            key=lambda p: (p.score, -p.code_length),  # 同分时偏好短代码
            reverse=True,
        )[:10]
        with open(self.data_dir / "best.json", "w") as f:
            json.dump([asdict(p) for p in top10], f, ensure_ascii=False, indent=2)

    def get_prompt_programs(self) -> tuple[list[Program], int]:
        """从随机岛中采样 2 个程序。"""
        island_id = random.randint(0, self.num_islands - 1)
        island = self.islands[island_id]
        programs = island.sample(n=2)
        if not programs:
            for other in self.islands:
                programs = other.sample(n=2)
                if programs:
                    break
        return programs, island_id

    def register(self, program: Program, island_id: int | None = None,
                 scores_per_test: ScoresPerTest | None = None) -> None:
        """注册一个已评测的程序。

        Args:
            program: 已评测的程序
            island_id: 目标岛 ID。None 时注册到所有岛（初始种子）。
                       如不传，使用 program.island_id。
            scores_per_test: 多测试用例分数。None 时从 program 属性回退。
        """
        if scores_per_test is None:
            # 兼容旧接口：从 program 属性获取，或用单一 score
            if program.scores_per_test:
                scores_per_test = program.scores_per_test
            else:
                scores_per_test = {"overall": program.score}

        # 保存 scores_per_test 到 program（用于持久化）
        program.scores_per_test = dict(scores_per_test)

        self.all_programs[program.id] = program

        if island_id is None:
            # 注册到所有岛（初始种子）
            target_ids = list(range(self.num_islands))
        else:
            target_ids = [island_id]

        for iid in target_ids:
            if iid < self.num_islands:
                self.islands[iid].register(program, scores_per_test)

        self._register_count += 1
        if self._register_count % self.reset_interval == 0:
            self._maybe_reset_islands()

    def _maybe_reset_islands(self):
        """淘汰最弱的一半岛。"""
        scores = [(i, island.best_score) for i, island in enumerate(self.islands)]
        # 加微量噪声打破平局
        scores_with_noise = [(i, s + random.gauss(0, 1e-6)) for i, s in scores]
        scores_with_noise.sort(key=lambda x: x[1])

        num_reset = self.num_islands // 2
        weak_ids = [s[0] for s in scores_with_noise[:num_reset]]
        strong_ids = [s[0] for s in scores_with_noise[num_reset:]]

        for wid in weak_ids:
            self.islands[wid] = Island(wid)
            donor_id = random.choice(strong_ids)
            donor = self.islands[donor_id]
            if donor.best_program and donor.best_scores_per_test:
                seed = Program(
                    id=f"reset-{self._register_count}-island{wid}",
                    code=donor.best_program.code,
                    score=donor.best_program.score,
                    island_id=wid,
                    parent_ids=[donor.best_program.id],
                    generation=donor.best_program.generation,
                    scores_per_test=dict(donor.best_scores_per_test),
                )
                self.islands[wid].register(seed, donor.best_scores_per_test)

    def snapshot(self):
        """记录快照。"""
        best_scores = [island.best_score for island in self.islands]
        all_scores = [p.score for p in self.all_programs.values()]
        all_lengths = [p.code_length for p in self.all_programs.values()]
        cluster_counts = [len(island.clusters) for island in self.islands]
        self.history.append({
            "register_count": self._register_count,
            "total_programs": len(self.all_programs),
            "best_per_island": best_scores,
            "clusters_per_island": cluster_counts,
            "global_best": max(all_scores) if all_scores else 0,
            "global_avg": sum(all_scores) / len(all_scores) if all_scores else 0,
            "avg_code_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def get_best(self) -> Program | None:
        if not self.all_programs:
            return None
        return max(self.all_programs.values(), key=lambda p: p.score)
