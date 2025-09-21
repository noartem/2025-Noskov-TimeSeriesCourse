import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize
import numpy as np

# We need to import all the distance functions and the normalizer
from .metrics import ED_distance, DTW_distance, norm_ED_distance
from .utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {"euclidean", "dtw"}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = "euclidean", is_normalize: bool = False) -> None:
        self.metric: str = metric
        self.is_normalize: bool = is_normalize

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """
        norm_str = "normalized " if self.is_normalize else "non-normalized "
        return norm_str + self.metric + " distance"

    def _choose_distance(self):
        """Choose distance function for calculation of matrix

        Returns
        -------
        dict_func: function reference
        """
        match self.metric:
            case "euclidean":
                return norm_ED_distance if self.is_normalize else ED_distance
            case "dtw":
                return DTW_distance
            case _:
                return None

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate distance matrix

        Parameters
        ----------
        input_data: time series set

        Returns
        -------
        matrix_values: distance matrix
        """
        num_ts = input_data.shape[0]
        matrix_values = np.zeros(shape=(num_ts, num_ts))

        dist_func = self._choose_distance()
        if dist_func is None:
            raise ValueError(f"Metric '{self.metric}' is not supported.")

        processed_data = input_data.copy()

        if self.metric == "dtw" and self.is_normalize:
            processed_data = np.apply_along_axis(z_normalize, 1, processed_data)

        for i in range(num_ts):
            for j in range(i + 1, num_ts):
                dist = dist_func(processed_data[i], processed_data[j])
                matrix_values[i, j] = dist
                matrix_values[j, i] = dist

        return matrix_values
