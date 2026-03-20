from sklearn.ensemble import RandomForestClassifier

NAME = "RF"
USES_SCALED = False   # tree-based → no scaling needed


def build(n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
