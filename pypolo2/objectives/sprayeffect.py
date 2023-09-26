from typing import List
import numpy as np

def spray_effect(candidates: np.ndarray, allstate: np.ndarray,mean: np.ndarray,extent: List[float]) -> np.ndarray:
    """
    Compute the effect of sprayer.

    Parameters
    ----------
    candidates : np.ndarray
        candidate point.洒水点
    allstate : np.ndarray
        all point。全部坐标
    mean : np.ndarray
        related values,全部坐标对应的污染物分布
    Returns
    -------
    spray_effect: np.ndarray
       启发式的洒水效果函数

    """
    spray_effect_list = []
    for i in range(candidates.shape[0]):
        effect = 0
        candidate_point = candidates[i]
        for a in range(3):
            for b in range(3):
                c1 = candidate_point[0] - 1 + a
                c2 = candidate_point[1] - 1 + b
                c3 = candidate_point[2]
                if c1 < extent[0] or c1 >= extent[1] or c2 < extent[2] or c2 >= extent[3]:
                    continue
                else:
                    row_index = np.where(np.all(allstate == np.array([c1, c2, c3]), axis=1))[0]
                    if a == 1 and b == 1:
                        if row_index.size > 0:
                            effect = effect + 0.2*mean[row_index[0]]
                        else:
                            raise ValueError
                    else:
                        if row_index.size > 0:
                            effect = effect + 0.15*mean[row_index[0]]
                        else:
                            raise ValueError
        spray_effect_list.append(effect)      
    spray_effect = np.array(spray_effect_list)
    return spray_effect


