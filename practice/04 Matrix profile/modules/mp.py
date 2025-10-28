import numpy as np
import stumpy


def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile

    Parameters
    ----------
    ts1: the first time series
    m: the subsequence length
    exclusion_zone: exclusion zone
    ts2: the second time series

    Returns
    -------
    output: the matrix profile structure
            (matrix profile, matrix profile index, subsequence length, exclusion zone, the first and second time series)
    """

    if ts2 is None:
        mp = stumpy.stump(ts1, m)
    else:
        mp = stumpy.stump(ts1, m, T_B=ts2)

    return {
        'mp': mp[:, 0],
        'mpi': mp[:, 1].astype(int),
        'm' : m,
        'excl_zone': exclusion_zone,
        'data': {'ts1': ts1, 'ts2': ts2}
    }
