"""进化目标定义：整个 AssociativeMemory 类。

v5: 从单函数扩展到整个类——LLM 可以自由设计 init/query/update + 额外状态。
"""

EVOLVE_FUNCTION_NAME = "AssociativeMemory"

INITIAL_IMPLEMENTATION = '''
class AssociativeMemory:
    """Associative memory for DGD retrieval."""

    def __init__(self, dim, alpha=1.0, eta=0.01):
        """Initialize memory.

        Args:
            dim: vector dimension
            alpha: forgetting rate (1.0 = no time decay)
            eta: learning rate
        """
        self.dim = dim
        self.alpha = alpha
        self.eta = eta
        # Identity matrix as initial state
        self.M = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]

    def query(self, key):
        """Return activation = M @ key.

        Args:
            key: query vector (list of floats, length dim)
        Returns:
            activation vector (list of floats, length dim)
        """
        dim = self.dim
        return [sum(self.M[i][j] * key[j] for j in range(dim)) for i in range(dim)]

    def update(self, key, target):
        """Update M given a query key and correct target.

        Args:
            key: query vector
            target: correct target vector
        """
        dim = self.dim
        alpha = self.alpha
        eta = self.eta
        M = self.M
        activation = self.query(key)
        error = [activation[i] - target[i] for i in range(dim)]
        for i in range(dim):
            for j in range(dim):
                identity_term = alpha if i == j else 0.0
                forget = M[i][j] * (identity_term - eta * key[i] * key[j])
                learn = -eta * error[i] * key[j]
                M[i][j] = forget + learn
'''

PROMPT_TEMPLATE = '''"""Associative memory class for DGD retrieval system.

The class maintains internal state and provides query() and update() methods.
query(key) returns an activation vector used to find nearest neighbors.
update(key, target) adjusts internal state based on feedback.

Goal: design a class that, after repeated update() calls, makes query() return
activations that better match the correct targets.

You may:
- Add any extra state in __init__ (momentum, history, running averages, etc.)
- Change the query logic (not just M @ key)
- Change the update rule (not just standard Hebbian)
- Use math functions: sqrt, exp, log, abs, max, min, sum
- Use list comprehensions

You may NOT:
- Import any modules
- Use numpy or any external library
- Make query() or update() take different arguments
- Output anything outside the class definition

Output ONLY the class definition, no explanation, no markdown.
"""

{versioned_functions}

class {next_function_name}:
    """Improved version of {prev_function_name}."""
'''
