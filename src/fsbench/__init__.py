"""fsbench: budget-aware binary feature-selection searchers (PREREG_STAGE1 sections 4, 5, 7, 9).

Imports work through the `src` directory: pytest adds it via pyproject.toml
([tool.pytest.ini_options] pythonpath = ["src"]); scripts add it to sys.path themselves.
No global random state is used anywhere: every random draw comes from a numpy Generator
passed in by the caller.
"""

from .evaluator import CHECKPOINTS, BudgetExhausted, Checkpoint, Evaluator

__all__ = ["CHECKPOINTS", "BudgetExhausted", "Checkpoint", "Evaluator"]
