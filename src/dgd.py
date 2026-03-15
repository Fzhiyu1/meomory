"""Delta Gradient Descent 关联记忆模块

M(key) → value 的在线学习关联记忆。
- 初始化：用 key-value 对训练 M
- 查询：activation = M @ query → 在 value 集合中找最近邻
- 更新：DGD 规则在线修正 M
"""
import json
import math
from pathlib import Path


class AssociativeMemory:
    """DGD 关联记忆。M 是 [dim × dim] 的权重矩阵。"""

    def __init__(self, dim: int = 256, alpha: float = 0.99, eta: float = 0.01):
        """
        Args:
            dim: 向量维度（投影后）
            alpha: 遗忘率（1.0 = 不遗忘，越小遗忘越快）
            eta: 学习率
        """
        self.dim = dim
        self.alpha = alpha
        self.eta = eta
        # 初始化 M 为单位矩阵（初始状态 = 直通，query 直接作为 activation）
        self.M = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]

    def query(self, key: list[float]) -> list[float]:
        """activation = M @ key"""
        return _matvec(self.M, key)

    def update(self, key: list[float], target: list[float]):
        """DGD 更新规则。

        M_new = M_old @ (α*I - η * k @ k^T) - η * error @ k^T
        其中 error = M_old @ key - target

        展开: M_new[i][j] = α * M_old[i][j] - η * activation[i] * key[j] - η * error[i] * key[j]
        """
        activation = self.query(key)
        error = [a - t for a, t in zip(activation, target)]
        dim = self.dim

        for i in range(dim):
            for j in range(dim):
                self.M[i][j] = (self.alpha * self.M[i][j]
                                - self.eta * activation[i] * key[j]
                                - self.eta * error[i] * key[j])

    def train(self, keys: list[list[float]], values: list[list[float]], epochs: int = 10):
        """用 key-value 对批量训练 M。"""
        for epoch in range(epochs):
            total_error = 0.0
            for key, value in zip(keys, values):
                activation = self.query(key)
                error_norm = sum((a - v) ** 2 for a, v in zip(activation, value)) ** 0.5
                total_error += error_norm
                self.update(key, value)
            avg_error = total_error / len(keys) if keys else 0
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: avg error = {avg_error:.4f}")

    def save(self, path: Path):
        """保存 M 矩阵和参数。"""
        data = {
            "dim": self.dim,
            "alpha": self.alpha,
            "eta": self.eta,
            "M": self.M,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "AssociativeMemory":
        """从文件加载。"""
        with open(path) as f:
            data = json.load(f)
        mem = cls(dim=data["dim"], alpha=data["alpha"], eta=data["eta"])
        mem.M = data["M"]
        return mem


def _matvec(M: list[list[float]], v: list[float]) -> list[float]:
    """矩阵向量乘法：M @ v"""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]
