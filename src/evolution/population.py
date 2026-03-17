"""种群库：持久化存储、谱系追踪、选择操作。"""
import json
import time
from pathlib import Path


class Individual:
    """一个候选 Judge prompt。"""

    def __init__(
        self,
        id: str,
        system_prompt: str,
        user_prompt_template: str,
        fitness: float = 0.0,
        generation: int = 0,
        parent_ids: list[str] = None,
        mutation_type: str = "seed",
        eval_details: dict = None,
    ):
        self.id = id
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.fitness = fitness
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.mutation_type = mutation_type
        self.eval_details = eval_details or {}
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {
            "id": self.id,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_type": self.mutation_type,
            "eval_details": self.eval_details,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        ind = cls(
            id=d["id"],
            system_prompt=d["system_prompt"],
            user_prompt_template=d["user_prompt_template"],
            fitness=d.get("fitness", 0),
            generation=d.get("generation", 0),
            parent_ids=d.get("parent_ids", []),
            mutation_type=d.get("mutation_type", "unknown"),
            eval_details=d.get("eval_details", {}),
        )
        ind.created_at = d.get("created_at", "")
        return ind


class Population:
    """种群库，支持持久化和谱系追踪。"""

    def __init__(self, data_dir: str = "experiments/evolution"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.population_file = self.data_dir / "population.json"
        self.history_file = self.data_dir / "history.json"
        self.best_file = self.data_dir / "best.json"

        self.current: list[Individual] = []
        self.history: list[dict] = []  # 每代快照
        self.all_evaluated: dict[str, Individual] = {}  # id → Individual
        self._load()

    def _load(self):
        """从磁盘恢复状态。"""
        if self.population_file.exists():
            with open(self.population_file) as f:
                data = json.load(f)
            self.current = [Individual.from_dict(d) for d in data.get("current", [])]
            self.all_evaluated = {
                d["id"]: Individual.from_dict(d)
                for d in data.get("all_evaluated", [])
            }
        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)

    def save(self):
        """持久化到磁盘。"""
        with open(self.population_file, "w") as f:
            json.dump({
                "current": [ind.to_dict() for ind in self.current],
                "all_evaluated": [ind.to_dict() for ind in self.all_evaluated.values()],
            }, f, ensure_ascii=False, indent=2)

        with open(self.history_file, "w") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        # 更新 best top-10
        top10 = sorted(self.all_evaluated.values(), key=lambda x: x.fitness, reverse=True)[:10]
        with open(self.best_file, "w") as f:
            json.dump([ind.to_dict() for ind in top10], f, ensure_ascii=False, indent=2)

    def add(self, ind: Individual):
        """添加已评测的个体。"""
        self.all_evaluated[ind.id] = ind

    def snapshot(self, generation: int):
        """记录当代快照。"""
        fitnesses = [ind.fitness for ind in self.current]
        self.history.append({
            "generation": generation,
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "population_size": len(self.current),
            "best_id": max(self.current, key=lambda x: x.fitness).id if self.current else "",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def get_best(self) -> Individual | None:
        if not self.all_evaluated:
            return None
        return max(self.all_evaluated.values(), key=lambda x: x.fitness)

    @property
    def generation(self) -> int:
        return len(self.history)
