import copy
import random
import sys
import pickle as pkl
from time import time

sys.path.append(".")
from GridContext.MultiStepMovingContext import MultiStepMovingContext
import numpy as np
from numba import njit

@njit
def accessible(agent_position_list, map_shape):
  t, _ = agent_position_list.shape
  for j in range(t):
    pos = agent_position_list[j]
    if(not map_shape[pos[0], pos[1]]):
      return False
  return True

class GridAccessContext(MultiStepMovingContext):
    def __init__(self, map_shape, time, agent_init_position, **kwargs) -> None:
      """
      Required kwargs:
      target_dist: the map_shaped target distributions
      access_mask: indicating whether a location is accessible
      name: the name for the target dist
      """
      super().__init__(map_shape, time, agent_init_position, **kwargs)
      self.target_dist = kwargs['target_dist'] / np.sum(kwargs['target_dist'])
      self.access_mask = kwargs['access_mask']
      self.name = kwargs['name']

    def CheckValid(self, agent_position_list):
      if(np.max(agent_position_list <= -1) == True):
        return False
      if(np.max(agent_position_list[:, 0] >= self.map_shape[0]) == True):
        return False
      if(np.max(agent_position_list[:, 1] >= self.map_shape[1]) == True):
        return False
      if(not accessible(agent_position_list, self.access_mask)):
        return False
      return True