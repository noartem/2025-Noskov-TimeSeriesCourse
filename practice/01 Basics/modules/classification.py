import numpy as np
from typing import Self
from modules.utils import z_normalize
from modules.metrics import ED_distance, DTW_distance, norm_ED_distance

# from modules.metrics import *


default_metrics_params = {
    "euclidean": {"normalize": True},
    "dtw": {"normalize": True, "r": 0.05},
}


class TimeSeriesKNN:
    """
    KNN Time Series Classifier

    Parameters
    ----------
    n_neighbors: number of neighbors
    metric: distance measure between time series
             Options: {euclidean, dtw}
    metric_params: dictionary containing parameters for the distance metric being used
    """

    def __init__(
        self,
        n_neighbors: int = 3,
        metric: str = "euclidean",
        metric_params: dict | None = None,
    ) -> None:
        self.n_neighbors: int = n_neighbors
        self.metric: str = metric
        self.metric_params: dict | None = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> Self:
        """
        Fit the model using X_train as training data and Y_train as labels

        Parameters
        ----------
        X_train: train set with shape (ts_number, ts_length)
        Y_train: labels of the train set

        Returns
        -------
        self: the fitted model
        """
        self.X_train = X_train
        self.Y_train = Y_train

        return self

    def _distance(self, x_train: np.ndarray, x_test: np.ndarray) -> float:
        """
        Compute distance between the train and test samples

        Parameters
        ----------
        x_train: sample of the train set
        x_test: sample of the test set

        Returns
        -------
        dist: distance between the train and test samples
        """

        dist = 0

        if self.metric == "euclidean":
            if self.metric_params.get("normalize", False):
                dist = norm_ED_distance(x_train, x_test)
            else:
                dist = ED_distance(x_train, x_test)
        elif self.metric == "dtw":
            if self.metric_params.get("normalize", False):
                x_train = z_normalize(x_train)
                x_test = z_normalize(x_test)

            dist = DTW_distance(x_train, x_test, r=self.metric_params["r"])
        else:
            raise ValueError(f"Metric '{self.metric}' is not supported.")

        return dist

    def _find_neighbors(self, x_test: np.ndarray) -> list[tuple[float, int]]:
        """
        Find the k nearest neighbors of the test sample

        Parameters
        ----------
        x_test: sample of the test set

        Returns
        -------
        neighbors: k nearest neighbors (distance between neighbor and test sample, neighbor label) for test sample
        """

        neighbors = []

        all_distances = []
        for i in range(len(self.X_train)):
            dist = self._distance(self.X_train[i], x_test)
            all_distances.append((dist, self.Y_train[i]))

        all_distances.sort(key=lambda x: x[0])

        neighbors = all_distances[: self.n_neighbors]

        return neighbors

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for samples of the test set

        Parameters
        ----------
        X_test: test set with shape (ts_number, ts_length))

        Returns
        -------
        y_pred: class labels for each data sample from test set
        """

        y_pred = []

        for x_test in X_test:
            neighbors = self._find_neighbors(x_test)
            labels = [label for _, label in neighbors]
            y_pred.append(max(set(labels), key=labels.count))

        return np.array(y_pred)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy classification score

    Parameters
    ----------
    y_true: ground truth (correct) labels
    y_pred: predicted labels returned by a classifier

    Returns
    -------
    score: accuracy classification score
    """

    score = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            score = score + 1
    score = score / len(y_true)

    return score
