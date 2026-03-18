"""v6 seed 集合：5 种不同架构，分配到 5 个岛。

岛 0: 原始 DGD（最简单的关联记忆）
岛 1: CMS 三层架构（多时间尺度，天然抗干扰）
岛 2: gen059 双通道（当前精度最强，从文件加载）
岛 3: gen057 单通道（抗噪王，从文件加载）
岛 4: 空白初始实现（单位矩阵，自由探索）
"""
import json
from pathlib import Path


SEED_ORIGINAL_DGD = '''
class AssociativeMemory:
    def __init__(self, dim, alpha=1.0, eta=0.01):
        self.dim = dim
        self.alpha = alpha
        self.eta = eta
        self.M = [[0.0]*dim for _ in range(dim)]

    def query(self, key):
        dim = self.dim
        out = [0.0]*dim
        for i in range(dim):
            s = 0.0
            for j in range(dim):
                s += self.M[i][j] * key[j]
            out[i] = s
        return out

    def update(self, key, target):
        dim = self.dim
        act = self.query(key)
        for i in range(dim):
            err = target[i] - act[i]
            for j in range(dim):
                self.M[i][j] = self.alpha * self.M[i][j] + self.eta * err * key[j]
'''

SEED_CMS = '''
class AssociativeMemory:
    def __init__(self, dim, alpha=1.0, eta=0.01):
        self.dim = dim
        self.eta = eta
        self.M_high = [[0.0]*dim for _ in range(dim)]
        self.M_mid = [[0.0]*dim for _ in range(dim)]
        self.M_low = [[0.0]*dim for _ in range(dim)]
        self.step = 0

    def _matvec(self, M, v):
        dim = self.dim
        out = [0.0]*dim
        for i in range(dim):
            s = 0.0
            for j in range(dim):
                s += M[i][j] * v[j]
            out[i] = s
        return out

    def query(self, key):
        dim = self.dim
        act = key[:]
        for M, w in [(self.M_high, 0.5), (self.M_mid, 0.3), (self.M_low, 0.2)]:
            layer_act = self._matvec(M, act)
            act = [w * la + (1-w) * a for la, a in zip(layer_act, act)]
        return act

    def update(self, key, target):
        dim = self.dim
        self.step += 1
        act = self.query(key)
        def _update_layer(M, key, target, act):
            for i in range(dim):
                err = target[i] - act[i]
                for j in range(dim):
                    M[i][j] += self.eta * err * key[j]
        _update_layer(self.M_high, key, target, act)
        if self.step % 5 == 0:
            _update_layer(self.M_mid, key, target, act)
        if self.step % 20 == 0:
            _update_layer(self.M_low, key, target, act)
'''

SEED_BLANK = '''
class AssociativeMemory:
    def __init__(self, dim, alpha=1.0, eta=0.01):
        self.dim = dim
        self.eta = eta
        self.M = [[1.0 if i==j else 0.0 for j in range(dim)] for i in range(dim)]

    def query(self, key):
        dim = self.dim
        out = [0.0]*dim
        for i in range(dim):
            s = 0.0
            for j in range(dim):
                s += self.M[i][j] * key[j]
            out[i] = s
        return out

    def update(self, key, target):
        dim = self.dim
        act = self.query(key)
        for i in range(dim):
            err = target[i] - act[i]
            for j in range(dim):
                self.M[i][j] += self.eta * err * key[j]
'''


def load_all_seeds(population_path: str = "experiments/funsearch-v4/old_population.json"):
    """加载 5 种 seed，返回 [(name, code), ...]。"""
    seeds = [
        ("original_dgd", SEED_ORIGINAL_DGD.strip()),
        ("cms_3layer", SEED_CMS.strip()),
    ]

    # 从 population 加载 gen059 和 gen057
    pop_file = Path(population_path)
    if pop_file.exists():
        pop = json.loads(pop_file.read_text())
        progs = {p["id"]: p["code"] for p in pop["all_programs"]}
        for target_name, target_id in [("gen059", "gen059-island3-001"), ("gen057", "gen057-island3-001")]:
            if target_id in progs:
                seeds.append((target_name, progs[target_id].strip()))
            else:
                # fallback: 用空白 seed
                seeds.append((target_name, SEED_BLANK.strip()))
    else:
        seeds.append(("gen059_fallback", SEED_BLANK.strip()))
        seeds.append(("gen057_fallback", SEED_BLANK.strip()))

    seeds.append(("blank_identity", SEED_BLANK.strip()))
    return seeds
