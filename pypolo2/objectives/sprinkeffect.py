from typing import List
import numpy as np



def sprink_effect(candidates: np.ndarray, allstate: np.ndarray,mean: np.ndarray,extent: List[float]) -> np.ndarray:
    """
    Compute the effect of sprinker.

    Parameters
    ----------
    candidates : np.ndarray
        candidate point.
    allstate : np.ndarray
        all point
    mean : np.ndarray
        related values
    Returns
    -------
    sprink_effect: np.ndarray
        简化的洒水效果,考虑洒水区域降低30%,周围一圈降低10%

    """
    sprink_effect_list = []
    for i in range(candidates.shape[0]):
        effect = 0
        candidate_point = candidates[i]
        for a in range(3):
            for b in range(3):
                c1 = candidate_point[0] - 1 + a
                c2 = candidate_point[1] - 1 + b
                c3 = candidate_point[2]
                if c1 < extent[0] or c1 > extent[1] or c2 < extent[2] or c2 > extent[3]:
                    continue
                else:
                    row_index = np.where(np.all(allstate == np.array([c1, c2, c3]), axis=1))[0]
                    if a == 1 and b == 1:
                        if row_index.size > 0:
                            effect = effect + mean[row_index[0]]*0.3
                        else:
                            raise ValueError
                    else:
                        if row_index.size > 0:
                            effect = effect + mean[row_index[0]]*0.2
                        else:
                            raise ValueError
        sprink_effect_list.append(effect)      
    sprink_effect = np.array(sprink_effect_list)
    return sprink_effect
