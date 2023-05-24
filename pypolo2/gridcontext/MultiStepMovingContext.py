import copy
import sys
import numpy as np
sys.path.append(".")

from GridContext.GridMovingContext import GridMovingContext

class MultiStepMovingContext(GridMovingContext):
  def __init__(self, map_shape, time, agent_init_position, **kwargs) -> None:
    self.previous_trace = np.zeros(map_shape)
    super().__init__(map_shape, time, agent_init_position, **kwargs)

  def ReInitialize(self):
    self.agent_number = len(self.agent_init_position)
    self.policy_matrix = np.ones((self.agent_number, self.time)).astype(np.int16) * (len(self.move_matrix) // 2)
    self.curr_trace_set = self.calculate_trace_set()
    self.curr_matrix = self.calculate_matrix()
    self.possible_actions = len(self.move_matrix)

  def SetPreviousTrace(self, previous_trace):
    self.previous_trace = previous_trace

  def calculate_matrix(self):
    return super().calculate_matrix() + self.previous_trace