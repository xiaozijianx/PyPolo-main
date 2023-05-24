import copy
import random
import sys
import pickle as pkl
from time import time

import itertools
import numpy as np
# from ..Common.QFunctions import time_decay_aggregation, calculate_kldiv
# from ..Common.common import even_dist, gaussian_dist, dg_corner_dist, noise_dist
from ..models import IModel
from ..objectives.entropy import gaussian_entropy
MoveMatrix2D = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
MoveMatrix2DWithSprayControl = list(itertools.product([-1, 0, 1], [-1, 0, 1], [0, 1]))

class GridMovingContext():
  def __init__(self, map_shape, time, agent_init_position, model:IModel, sprinkeffect_all, allstate, move_matrix = MoveMatrix2DWithSprayControl, alpha = 0.5) -> None:
    """
    Accepted keywords:
    target_dist: A mapshape sized map, indicating Object data value distributions.
    NOTE: the length of trace set is 1 more than time
    """
    #作业区域
    self.map_shape = map_shape
    #规划时长
    self.time = time
    #初始智能体位置
    self.agent_init_position = agent_init_position
    #智能体的移动模型
    self.move_matrix = move_matrix
    #预测模型及所需参数
    self.model = model
    self.alpha = alpha
    self.sprinkeffect_all = sprinkeffect_all
    self.allstate = allstate
    self.agent_number = len(self.agent_init_position)
    self.time_co = 0.0001
    
    #智能体策略矩阵，包含每个智能体每个时间的动作，初始均设为中间值，存储的是动作序号
    self.policy_matrix = np.ones((self.agent_number, time)).astype(np.int16) * (len(self.move_matrix) // 2)
    
    #智能体轨迹，存放了包括初始轨迹在内，轨迹是一系列二维坐标
    self.curr_trace_set = self.calculate_trace_set()
    
    #当前轨迹所覆盖矩阵
    self.curr_matrixA, self.curr_matrixB, self.curr_matrixC = self.calculate_matrix()
    
    #可选动作长度
    self.possible_actions = len(self.move_matrix)

  def CalculateSQ(self):
      return self.calculate_objective_scores()

  def calculate_objective_scores(self, method = 1):
    #calcullate spray effcet
    spray_effect = 0
    curr_trace_set = self.curr_trace_set.copy()
    # sprinkeffect_all = self.sprinkeffect_all.copy()
    normed_effect = (self.sprinkeffect_all - self.sprinkeffect_all.min()) / self.sprinkeffect_all.ptp()
    sprinkeffect = np.zeros((self.task_extent[1]+1-self.task_extent[0],self.task_extent[3]+1-self.task_extent[2]))
    for i in range (self.task_extent[0],self.task_extent[1]+1):
      for j in range (self.task_extent[2],self.task_extent[3]+1):
        sprinkeffect[i,j] = normed_effect[i*(self.task_extent[3]+1-self.task_extent[2])+j]
        
    for j in range(self.time + 1):
      for i in range(self.agent_number):
        if self.curr_trace_set[i, j, 2] == 1:
          r0 = curr_trace_set[i, j, 0]
          c0 = curr_trace_set[i, j, 1]
          for a in range(3):
            for b in range(3):
              r = int(r0 - 1 + a)
              c = int(c0 - 1 + b)
              if r >= 0 and r < self.map_shape[0] and c >= 0 and c < self.map_shape[1]:
                if a == 1 and b == 1:
                  spray_effect = spray_effect + 0.3 * (1-0.9)**j * sprinkeffect[r,c]
                  sprinkeffect[r,c] = (1 - 0.3 ) * sprinkeffect[r,c]
                else:
                  spray_effect = spray_effect + 0.2 * (1-0.9)**j * sprinkeffect[r,c]
                  sprinkeffect[r,c] = (1 - 0.2 ) * sprinkeffect[r,c]

    #calculate mi about select point for all points
    if method == 1:
      #calculate mi at  one time point
      curr_matrixB = self.curr_matrixB
      allstate = self.allstate
      self.model.add_data_x(curr_matrixB)
      prior_diag_std, poste_diag_std, poste_cov, poste_cov = self.model.prior_poste(allstate)
      poste_cov
      # hprior = gaussian_entropy(prior_diag_std.ravel())
      # hposterior = gaussian_entropy(poste_diag_std.ravel())
      # mi_all = hprior - hposterior
      # if np.any(mi_all < 0.0):
      #     print(mi_all.ravel())
      #     raise ValueError("Predictive MI < 0.0!")
    
    
    return 0

  def calculate_matrix(self):
    #calculate three matrix, one for spray effect, onr for mi and one as before
    #first as before
    matrixA =  np.zeros(self.map_shape)
    for i in range(self.agent_number):
      for j in range(self.time + 1):
        if j == 0:
          continue
        x = self.curr_trace_set[i, j, 0:2][0].astype(np.int16)
        y = self.curr_trace_set[i, j, 0:2][1].astype(np.int16)
        matrixA[x, y] += 1
          
    #third matrix for mi calculate at one times
    num = 0
    matrixB = np.zeros(((self.time+1)*self.agent_number,3))
    for x in range(self.map_shape[0]):
      for y in range(self.map_shape[1]):
        if matrixA[x, y] > 0.5:
          matrixB[num,0] = x
          matrixB[num,1] = y
          matrixB[num,2] = self.model.time_stamp
          num = num + 1
    matrixB = matrixB[0:num]
    
    #fourth matrix for mi calculate at different times
    num = 0
    matrixC = np.zeros(((self.time+1)*self.agent_number,3))
    mid = np.zeros((1,3))
    for j in range(self.time + 1):
      for i in range(self.agent_number):
        x = self.curr_trace_set[i, j, 0:2][0].astype(np.int16)
        y = self.curr_trace_set[i, j, 0:2][1].astype(np.int16)
        z = self.curr_trace_set[i, j, 3]
        can = np.array([[x,y,z]])
        if np.all(np.any(can == mid, axis=1)):
          continue
        mid = np.vstack((mid,can))
        matrixC[num,0] = x
        matrixC[num,1] = y
        matrixC[num,2] = z
        num = num + 1
      mid = np.zeros((1,3))
    matrixC = matrixC[0:num]
    
    return matrixA, matrixB, matrixC
  
  def calculate_trace_set(self):
    curr_trace_set =  np.zeros((self.agent_number, self.time + 1, 4))
    for i in range(self.agent_number):
      for j in range(self.time + 1):
        if(j == 0):
          curr_trace_set[i, j, 0:2] = np.array(self.agent_init_position[i])
          curr_trace_set[i, j, 2:3] = 0
          curr_trace_set[i, j, 3:4] = self.model.time_stamp
        else:
          curr_trace_set[i, j, 0:2] = np.array(self.move_matrix[self.policy_matrix[i, j - 1, 0:1]]) + \
                                             curr_trace_set[i, j - 1, 0:2]
          curr_trace_set[i, j, 2:3] = np.array(self.move_matrix[self.policy_matrix[i, j - 1, 2:3]])
          curr_trace_set[i, j, 3:4] = self.model.time_stamp + self.time_co * j
          
    return curr_trace_set

  def GetAgentNumber(self) -> int:
    return self.agent_number

  def GetMaxTime(self) -> int:
    return self.time

  def GetPossibleActions(self) -> int:
    return self.possible_actions
  
  def GetAgentInitialPosition(self):
    return self.agent_init_position

  def GetMoveMatrices(self):
    return self.move_matrix

  def GetMapShape(self):
    return self.map_shape

  
  def CheckValid(self, agent_position_list):
    if(np.max(agent_position_list <= -1) == True):
      return False
    if(np.max(agent_position_list[:, 0] >= self.map_shape[0]) == True):
      return False
    if(np.max(agent_position_list[:, 1] >= self.map_shape[1]) == True):
      return False
    return True