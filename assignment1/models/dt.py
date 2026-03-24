# Simple Decision Tree with a fixed random seed for reproducibility.
# No pruning applied — paper doesn't mention any max depth constraint.

from sklearn.tree import DecisionTreeClassifier

NAME = "DT"
USES_SCALED = False   # tree-based → no scaling needed


def build(random_state: int = 42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=random_state)
