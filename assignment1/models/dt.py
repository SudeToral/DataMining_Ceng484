from sklearn.tree import DecisionTreeClassifier

NAME = "DT"
USES_SCALED = False   # tree-based → no scaling needed


def build(random_state: int = 42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=random_state)
