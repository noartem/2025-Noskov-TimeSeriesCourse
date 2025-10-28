import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []
    
    mp = matrix_profile["mp"]
    pi = matrix_profile["mpi"]
    excl_zone = matrix_profile["excl_zone"]

    mp_work = np.copy(mp)
    mp_work[np.isnan(mp_work)] = np.inf

    while len(motifs_idx) < top_k:
        i = int(np.argmin(mp_work))
        if not np.isfinite(mp_work[i]):
            break

        j = int(pi[i])
        if j < 0 or j >= len(mp_work) or i == j or not np.isfinite(mp[i]):
            mp_work[i] = np.inf
            continue

        a, b = min(i, j), max(i, j)
        if any(a == x and b == y for x, y in motifs_idx):
            mp_work[i] = np.inf
            continue

        motifs_idx.append((a, b))
        motifs_dist.append(float(mp[i]))

        apply_exclusion_zone(mp_work, i, excl_zone, np.inf)
        apply_exclusion_zone(mp_work, j, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
