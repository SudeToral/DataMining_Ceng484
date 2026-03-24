# SVM with RBF kernel. Needs scaled input since it's margin-based.
# The paper didn't report C or gamma, so we go with sklearn defaults.

from sklearn.svm import SVC

NAME = "SVM"
USES_SCALED = True   # SVM is margin-based → needs StandardScaler


def build(kernel: str = "rbf", random_state: int = 42) -> SVC:
    return SVC(kernel=kernel, random_state=random_state)
