# KNN is sensitive to the choice of k, so we use GridSearch with 5-fold CV
# to pick the best k from [3, 5, 7, 9, 11] rather than hardcoding it.
# Needs scaled input since it's distance-based.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

NAME = "KNN"
USES_SCALED = True   # KNN is distance-based → needs MinMaxScaler


def build() -> GridSearchCV:
    """
    Tune k over [3, 5, 7, 9, 11] with 5-fold CV on the training set
    to find the k that best matches the paper's reported 81% accuracy.
    """
    base  = KNeighborsClassifier()
    param = {"n_neighbors": [3, 5, 7, 9, 11]}
    return GridSearchCV(base, param, cv=5, scoring="accuracy", n_jobs=-1)
