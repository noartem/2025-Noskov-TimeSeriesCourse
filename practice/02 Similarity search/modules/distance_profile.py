import numpy as np

from modules.metrics import ED_distance, norm_ED_distance


def brute_force(
    ts: np.ndarray, query: np.ndarray, is_normalize: bool = True
) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)

    N = n - m + 1
    assert N > 0

    dist_profile = np.zeros(shape=(N,))

    if is_normalize:
        for i in range(N):
            subseq = ts[i : i + m]
            d = norm_ED_distance(query, subseq)
            if d is None or np.isnan(d):
                d = np.inf
            dist_profile[i] = d
    else:
        for i in range(N):
            subseq = ts[i : i + m]
            d = ED_distance(query, subseq)
            if d is None or np.isnan(d):
                d = np.inf
            dist_profile[i] = d

    return dist_profile
