import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp = matrix_profile["mp"]
    pi = matrix_profile["mpi"]
    excl_zone = matrix_profile["excl_zone"]

    mp_work = np.copy(mp)
    mp_work[np.isnan(mp_work)] = -np.inf

    while len(discords_idx) < top_k:
        i = int(np.argmax(mp_work))
        if not np.isfinite(mp_work[i]):
            break

        if pi is None:
            j = -1
        else:
            j = int(pi[i])
            if j >= len(mp_work):
                j = -1

        discords_idx.append(i)
        discords_dist.append(float(mp[i]))
        discords_nn_idx.append(j)

        apply_exclusion_zone(mp_work, i, excl_zone, -np.inf)

        if j != -1:
            apply_exclusion_zone(mp_work, j, excl_zone, -np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
