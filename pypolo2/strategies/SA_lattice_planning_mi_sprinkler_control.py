import copy
import random
import sys
import pickle as pkl
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List

# from sklearn.utils import shuffle
# from Common.utils import PrintExecutionTime

from ..gridcontext.GridMovingContext import GridMovingContext, MoveMatrix2D
from ..objectives.entropy import gaussian_entropy
from ..objectives.sprinkeffect import sprink_effect
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot

# import networkx as nx

# return 0 if success, -1 if invalid
# a action is a tuple specifying time, move
#尝试动作，输入规划单元，智能体序号，时间序号，动作序号。
def try_move(context, agent, time, move):
  #获得智能体的动作列表
  MoveMatrix = context.GetMoveMatrices()
  #获得所选的智能体轨迹
  agent_position_list = context.curr_trace_set[agent, :, :].copy()
  #获得所选智能体所选时刻的当前动作
  previous_action = context.policy_matrix[agent, time]
  #计算动作之间的差值
  move_diff = np.array(MoveMatrix[move]) - np.array(MoveMatrix[previous_action])
  #计算更新动作后的智能体轨迹，所选时刻后均会更新
  agent_position_list[time + 1:, :] += move_diff
  #检查轨迹是否有效
  if(context.CheckValid(agent_position_list)):
    return agent_position_list[time + 1:, :]
  else:
    return None

def do_move(context, agent, time, move, agent_position_list) -> GridMovingContext:
  context.policy_matrix[agent, time] = move
  context.curr_trace_set[agent, time + 1:, :] = agent_position_list

def SimulatedAnnealing(origin_mc_context: GridMovingContext, *, n_playout=10000, initial_temp=1, k=0.95, bound=100, min_temp= 0.001, mini_step=1):
  sq_list = []
  curr_turns = 0
  curr_k = 1
  # try:
  Temp = initial_temp
  curr_context = copy.deepcopy(origin_mc_context)
  while(curr_turns < bound):
    iters = 0
    curr_turns += 1
    if(curr_k >= min_temp):
      curr_k = k * curr_k
    print(curr_turns)
    while(iters < n_playout):
      iters += 1
      rand_agent = random.randint(0, curr_context.GetAgentNumber() - 1)
      rand_time = random.randint(0, curr_context.GetMaxTime() - 1)
      rand_action = random.randint(0, curr_context.GetPossibleActions() - 1)
      try_result = try_move(curr_context, rand_agent, rand_time, rand_action)
      if(try_result is None):
        continue
      else:
        #根据随机选择的动作操作智能体轨迹
        new_mc_context = copy.deepcopy(curr_context)
        do_move(new_mc_context, rand_agent, rand_time, rand_action, try_result)
        #计算前后得分
        sq1 = curr_context.CalculateSQ()
        sq2 = new_mc_context.CalculateSQ()
        delta_e = sq1 - sq2
        # better, always accept
        if(delta_e >= 0):
          curr_context = new_mc_context
        else:
          # accept by chance
          accept_prob = np.exp(delta_e / (curr_k * Temp))
          # accept.append(accept_prob)
          if(random.random() < accept_prob):
            curr_context = new_mc_context
          continue
    sq_list.append(curr_context.CalculateSQ())
  return curr_context, sq_list

# @PrintExecutionTime
def SimulatedAnnealingFixed(origin_context: GridMovingContext, bound):
  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Temp = 1
  sq_list_total = []
  sq_list_total.append(origin_context.CalculateSQ())
  k = math.pow(0.0001, 1 / bound)
  context, sq_list = SimulatedAnnealing(origin_context, Temp, single_playout, k, bound)
  return context, sq_list_total + sq_list

def SAWrapper(params):
  return SimulatedAnnealing(*params)

# @PrintExecutionTime
def TemperatureParallelSA(origin_context: GridMovingContext, bound, thread_num=4, mini_step=1, init_temp=1):
  assert(bound % mini_step == 0)
  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Temp = np.array([1.0 * pow(2, i) for i in range(thread_num)])
  k = math.pow(0.0001, 1 / bound)

  SA_contexts = [(copy.deepcopy(origin_context), temp, single_playout, k, mini_step) for temp in Temp]
  sq_list_total = []
  sq_list_total.append(origin_context.CalculateSQ())

  with ProcessPoolExecutor(len(Temp)) as executor:
    for i in range(bound // mini_step):
      contexts_and_sq_lists = list(executor.map(SAWrapper, SA_contexts))

      new_contexts = [cq[0] for cq in contexts_and_sq_lists]
      lists = [cq[1] for cq in contexts_and_sq_lists]
      sq_list_total += np.min(np.array(lists), axis=0).tolist()

      # random.shuffle(new_contexts)
      Temp *= math.pow(k, mini_step)
      for i in range(0, len(new_contexts), 2):
        sq_1 = new_contexts[i].CalculateSQ()
        sq_2 = new_contexts[i + 1].CalculateSQ()
        if(sq_1 > sq_2):
          new_contexts[i], new_contexts[i+1] = new_contexts[i+1], new_contexts[i]
        else:
          accept_prob = np.exp(sq_1 - sq_2 / Temp[i])
          if(random.random() < accept_prob):
            new_contexts[i], new_contexts[i+1] = new_contexts[i+1], new_contexts[i]
      SA_contexts = [(context, temp, single_playout, k, mini_step) for context, temp in zip(new_contexts, Temp)]
    
  
  SA_contexts.sort(key=lambda x : x[0].CalculateSQ())
  return SA_contexts[0][0], sq_list_total

#定义SA算法包装
class SALatticePlanningMISprinklerControl(IStrategy):
    """Informative planning based on Mutual informaiton and sprinkler effect on latttice map use SA algorithms."""

    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        vehicle_team: dict,
    ) -> None:
        """
        Parameters
        ----------
        task_extent: List[float], [xmin, xmax, ymin, ymax]
            Bounding box of the sampling task workspace.
        rng: np.random.RandomState
            Random number generator if `get` has random operations.
        vehicle_team: dict
            team of vehicle.

        """
        super().__init__(task_extent, rng)
        self.vehicle_team = vehicle_team
        self.moving_context = None
        
    def get(self, model: IModel, alpha = 0.5, step_number = 4) -> np.ndarray:
        """Get goal states for sampling.

        Parameters
        ----------
        model: IModel, optional
            A probabilistic model that provides `mean` and `std` via `forward`.
        alpha: float,
            weight between mi and effect.

        Returns
        -------
        result: dict, id:(goal_states,spray_flag)
            Sampling goal states and spray_flag

        """
        #calculate sprinkeffect for all point
        allstate_list = []
        for i in range (self.task_extent[0],self.task_extent[1]+1):
            for j in range (self.task_extent[2],self.task_extent[3]+1):
                allstate_list.append([i, j, model.time_stamp])
        allstate = np.array(allstate_list)
        
        #compute predict mean and sprink_effect of all point
        mean, _ = model(allstate)
        sprinkeffect_all = sprink_effect(allstate,allstate,mean,self.task_extent).ravel()
        
        map_shape = (self.task_extent[1] + 1,self.task_extent[3] + 1)
        time = step_number
        agent_init_position = []
        for id, vehicle in self.vehicle_team.items():
          agent_init_position.append(vehicle.state[0:2])
        
        # policy_matrix = 
        self.moving_context = GridMovingContext(map_shape, time, agent_init_position, model, sprinkeffect_all, allstate)
        
        SimulatedAnnealingFixed(self.moving_context, 50)
        
        #compute mi of all points
        prior_diag_std, poste_diag_std, poste_cov, poste_cov = model.prior_poste(allstate)
        hprior = gaussian_entropy(prior_diag_std.ravel())
        hposterior = gaussian_entropy(poste_diag_std.ravel())
        mi_all = hprior - hposterior
        if np.any(mi_all < 0.0):
            print(mi_all.ravel())
            raise ValueError("Predictive MI < 0.0!")
        
        result_mean = mean.copy()
        result_mi_all = mi_all.copy()
        result_sprinkeffect_all = sprinkeffect_all.copy()
        result = dict()
        
        for id, vehicle in self.vehicle_team.items():
            # Normalized scores
            # normed_mi = (mi_all - mi_all.min()) / mi_all.ptp()
            normed_mi = (mi_all.max() - mi_all) / mi_all.ptp()
            normed_effect = (sprinkeffect_all - sprinkeffect_all.min()) / sprinkeffect_all.ptp()
            
            # normed_mi = mi_all / np.sum(mi_all)
            # normed_effect = sprinkeffect_all / np.sum(sprinkeffect_all)
            
            #trans to matrix form
            mi = np.zeros((self.task_extent[1]+1-self.task_extent[0],self.task_extent[3]+1-self.task_extent[2]))
            sprinkeffect = np.zeros((self.task_extent[1]+1-self.task_extent[0],self.task_extent[3]+1-self.task_extent[2]))
            
            #set threshold that sprinkeffect under this threshold means don't spray
            threshold = 25
            for i in range (self.task_extent[0],self.task_extent[1]+1):
                for j in range (self.task_extent[2],self.task_extent[3]+1):
                    mi[i,j] = normed_mi[i*(self.task_extent[3]+1-self.task_extent[2])+j]
                    if sprinkeffect_all[i*(self.task_extent[3]+1-self.task_extent[2])+j] > threshold:
                        sprinkeffect[i,j] = normed_effect[i*(self.task_extent[3]+1-self.task_extent[2])+j]
                        
            scores = alpha*mi + (1-alpha)*sprinkeffect

            path = self.greedy_search_multi_step(8,scores,id)[1:]

            goal_states = np.zeros((len(path),2))
            spray_flag = np.ones((len(path),1), dtype=bool)

            # Append waypoint
            for index, location in enumerate(path):
                goal_states[index,0] = location[0]
                goal_states[index,1] = location[1]
                #find point under threshold
                if sprinkeffect[location[0],location[1]] == 0:
                    spray_flag[index,0] = False
            
            result[id] = (goal_states,spray_flag)
            
            #reduce effect
            for i in range(len(path)):
                if self.vehicle_team[id].water_volume_now > i:
                    for m in range(3):
                        for n in range(3):
                            r = goal_states[i,0] -1 + m
                            c = goal_states[i,1] -1 + n
                            if r < self.task_extent[0] or r > self.task_extent[1] or c < self.task_extent[2] or c > self.task_extent[3]:
                                continue
                            if m == 1 and n == 1:
                                mi_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]=(0.9**i*2)*mi_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]
                                if spray_flag[i,0] == True:
                                    sprinkeffect_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]=(1-(0.9**i*0.3))*sprinkeffect_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]
                            if spray_flag[i,0] == True:
                                sprinkeffect_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]=(1-(0.9**i*0.2))*sprinkeffect_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]
                            mi_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]=(0.9**i*1.5)*mi_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]
                
        return result, result_mi_all, result_mean, result_sprinkeffect_all
    