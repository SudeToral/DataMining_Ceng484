from sklearn.neural_network import MLPClassifier

NAME = "ANN"
USES_SCALED = True   # gradient-based → needs MinMaxScaler


def build(random_state: int = 42) -> MLPClassifier:
    """
    Paper describes a 3-layer ANN (input → hidden → output).
    Single hidden layer of 100 neurons with ReLU, Adam optimiser,
    early stopping to prevent overfitting, max 1000 iterations.
    """
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),   # input → 100 → 50 → output
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=1000,
        random_state=random_state,
    )
