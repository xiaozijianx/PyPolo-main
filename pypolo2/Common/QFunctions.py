import numpy as np
import math
from numba import njit

@njit
def calculate_kldiv(ground_matrix, target_dist, eps=1e-12):
  ground_matrix += eps
  target_dist += eps
  x =  ground_matrix / np.sum(ground_matrix)
  kldiv =  np.sum( x * np.log(x / target_dist))
  return kldiv

@njit
def calculate_fairness(ground_matrix, eps=1e-12): # Fairness version, considering about time decay
  ground_matrix += eps
  x =  ground_matrix / np.sum(ground_matrix)
  entropy = - np.sum(x * np.log(x))
  return -entropy

@njit
def time_decay_aggregation(trace_set, map_size, decay_rate =1.0):
  agent_number, time, _ = trace_set.shape
  ground_matrix = np.zeros(map_size)
  for i in range(agent_number):
    for j in range(time):
      ground_matrix[trace_set[i, j, 0], trace_set[i, j, 1]] += math.pow(decay_rate, time)
  return ground_matrix

# @njit
# def calculte_sq_by_matrix():
#   ground_matrix = time_decay_aggregation(self.curr_trace_set, self.map_shape)
#   eps =  min(1.0 / self.map_shape[0] / self.map_shape[1] / self.time, 1e-12)
#   kldiv = calculate_kldiv(ground_matrix, self.target_dist, eps)