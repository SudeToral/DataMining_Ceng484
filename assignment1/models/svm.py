from sklearn.svm import SVC

NAME = "SVM"
USES_SCALED = True   # SVM is margin-based → needs StandardScaler


def build(kernel: str = "rbf", random_state: int = 42) -> SVC:
    return SVC(kernel=kernel, random_state=random_state)
