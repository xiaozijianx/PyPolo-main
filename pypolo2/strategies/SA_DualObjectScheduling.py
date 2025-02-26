import copy
import random
import sys
import pickle as pkl
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List
import time as tm
import matplotlib.pyplot as plt

# from sklearn.utils import shuffle
# from Common.utils import PrintExecutionTime

from ..gridcontext.GridMovingContext import GridMovingContext as GridMovingContext
# from ..gridcontext.GridMovingContext_MIdependSprayweight import GridMovingContext_MIDependSprayweight as GridMovingContext
from ..objectives.entropy import gaussian_entropy
from ..objectives.sprayeffect import spray_effect, calculate_effect
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot

# return 0 if success, -1 if invalid
# a action is a tuple specifying time, move
#尝试移动位置，仅考虑位置移动
def try_move(context, agent, time, move):
  #获得智能体的动作列表
  MoveMatrix = context.GetMoveMatrices()
  #获得所选的智能体轨迹list=[time,action], time=[0:num],action=[x,y,volume,timestamp]
  agent_position_list = context.curr_trace_set[agent, :, :].copy()
  #获得所选智能体所选时刻的当前动作
  previous_policy = context.policy_matrix[agent].copy()
  #判断当前动作是否为补水状态，补水状态不可移动
  if previous_policy[time,2] == -1:
    return None, None
  #计算动作之间的差值(移动)
  move_diff = np.array(MoveMatrix[move][0:2]) - previous_policy[time, 0:2]
  #计算更新移动动作后的智能体轨迹，所选时刻后均会更新
  agent_position_list[(time + 1):, 0:2] += move_diff
  #检查轨迹是否有效
  if(context.CheckValid(agent_position_list)):
    new_policy = previous_policy.copy()
    new_policy[time, 0:2] = np.array(MoveMatrix[move][0:2])
    return agent_position_list[time + 1:, :], new_policy
  else:
    return None, None
  
# Spraying delete operator
def try_spray0(rng,context, agent, selecttime):
  previous_policy = context.policy_matrix[agent].copy()
  num = 0
  time = 0
  for i in range(context.GetMaxTime()):
    if previous_policy[i,2] == 1:
      num = num + 1
    if num == selecttime + 1:
      time = i
      break
  if previous_policy[time,2] != 1:
    return None
  else:
    replenish_time = 0
    for i in range(context.GetMaxTime()-time-1):
      action = previous_policy[time+i+1]
      if action[2] == -1:
        replenish_time = time + i + 1
        break
    if replenish_time == 0:
      return None
    rand_replenish_time = rng.randint(time + 1, replenish_time+1)
    
    new_policy = previous_policy.copy()
    rep = 0
    for i in range(context.GetMaxTime()- time - 1):
      if new_policy[time + i + 1, 2] == -1:
        rep = rep + 1
        continue
      if rep >= 1:
        # EXCHANGE
        new_policy[time + i + 1 - rep, 0] = new_policy[time + i + 1, 0]
        new_policy[time + i + 1 - rep, 1] = new_policy[time + i + 1, 1]
        new_policy[time + i + 1, 0] = 0
        new_policy[time + i + 1, 1] = 0
        rep = 0
    
    new_policy[time,2] = 0
    for i in range(context.GetMaxTime() - rand_replenish_time - 1):
      m = context.GetMaxTime() - rand_replenish_time - 2 - i
      new_policy[rand_replenish_time + m + 1,2] = new_policy[rand_replenish_time + m,2]
    new_policy[rand_replenish_time,2] = 1
    return new_policy

# Spraying insert operator
def try_spray1(rng,context, agent, selecttime):
  agent_position_list = context.curr_trace_set[agent, :, :].copy()
  previous_policy = context.policy_matrix[agent].copy()
  num = 0
  time = 0
  for i in range(context.GetMaxTime()):
    if previous_policy[i,2] == 0:
      num = num + 1
    if num == selecttime + 1:
      time = i
      break
  if previous_policy[time,2] == 0:
    # 计算变更后的policy
    new_policy = previous_policy.copy()
    # 将选择的不洒水动作后的补水动作处的移动动作前移(交换)
    for i in range(context.GetMaxTime()- time - 1):
      if new_policy[time + i + 1, 2] == -1:
        # 与前面的移动动作交换(把前面的动作拿过来，把前面变成不移动)
        new_policy[time + i + 1, 0] = new_policy[time + i, 0]
        new_policy[time + i + 1, 1] = new_policy[time + i, 1]
        new_policy[time + i, 0] = 0
        new_policy[time + i, 1] = 0
    
    # 将选中时间后的洒水动作依次顺延
    for i in range(context.GetMaxTime()- time - 1):
      new_policy[time + i, 2] = new_policy[time + i + 1, 2]
      
    # 在最后插入合适的动作
    if agent_position_list[context.GetMaxTime(), 2] >= 1 and previous_policy[context.GetMaxTime() - 1, 2] != -1:
      new_policy[context.GetMaxTime() - 1, 2] = 1
    elif agent_position_list[context.GetMaxTime(), 2] < 1:
      new_policy[context.GetMaxTime() - 1, 2] = -1
      new_policy[context.GetMaxTime() - 1, 0] = 0
      new_policy[context.GetMaxTime() - 1, 1] = 0
    elif agent_position_list[context.GetMaxTime(), 2] >= context.Setting.water_volume:
      new_policy[context.GetMaxTime() - 1, 2] = 1
    elif agent_position_list[context.GetMaxTime(), 2] < context.Setting.water_volume and previous_policy[context.GetMaxTime() - 1, 2] == -1:
      new_policy[context.GetMaxTime() - 1, 2] = -1
      new_policy[context.GetMaxTime() - 1, 0] = 0
      new_policy[context.GetMaxTime() - 1, 1] = 0
    return new_policy
  else:
    return None
  
# Spraying exchange operator
def try_spray2(rng,context, agent, selecttime):
  previous_policy = context.policy_matrix[agent].copy()
  num = 0
  time = 0
  for i in range(context.GetMaxTime()):
    if previous_policy[i,2] == 0:
      num = num + 1
    if num == selecttime + 1:
      time = i
      break
    
  if previous_policy[time,2] == 0:
    # 寻找前后两个补水时刻
    replenish_time_1 = 0#前一个补水时刻
    for i in range(time):
      action = previous_policy[time-i-1]
      if action[2] == -1:
        replenish_time_1 = time-i-1
        break
      if i == time - 1:
        replenish_time_1 = -1
    
    replenish_time_2 = 0
    for i in range(context.GetMaxTime()-time-1):
      action = previous_policy[time+i+1]
      if action[2] == -1:
        replenish_time_2 = time + i + 1
        break
      if i == context.GetMaxTime()-time-2:
        replenish_time_2 = context.GetMaxTime()
    if replenish_time_2 - replenish_time_1 <= 1:
      return None
    
    # 统计该洒水阶段内的所有洒水次数
    num = 0
    for i in range(replenish_time_2 - replenish_time_1 - 1):
      if previous_policy[replenish_time_1 + 1 + i,2] == 1:
        num = num + 1
    
    if num == 0:
      return None
    
    # 寻找准备交换的洒水时段
    rand_exchange_time = rng.randint(0, num)
    exchange_time = 0
    for i in range(context.GetMaxTime()):
      if previous_policy[replenish_time_1 + 1 + i,2] == 1:
        num = num - 1
      if num == rand_exchange_time:
        exchange_time = replenish_time_1 + 1 + i
        break
    if previous_policy[exchange_time,2] != 1:
      return None
    # 计算变更后的policy
    new_policy = previous_policy.copy()
    new_policy[time,2] = 1
    new_policy[exchange_time,2] = 0
    return new_policy
  else:
    return None
    
def do_move(context, agent, time, New_policy, agent_position_list) -> GridMovingContext:
  # MoveMatrix = context.GetMoveMatrices()
  context.policy_matrix[agent] = New_policy
  context.curr_trace_set[agent, time + 1:, :] = agent_position_list
  
def do_spray(context, agent, New_policy) -> GridMovingContext:
  context.policy_matrix[agent] = New_policy
  for i in range(context.GetAgentNumber()):
    for j in range(context.GetMaxTime() + 1):
      if(j == 0):
        continue
      else:# 注意，洒水时，由于调整了移动，因此位置循序也要变化，这种变化可以在设计变化逻辑时考虑，也可以在实施变化时统一考虑
        #这里选择在这里统一考虑
        # 洒水
        # print(context.policy_matrix)
        context.curr_trace_set[i, j, 0:2] = context.curr_trace_set[i, j - 1, 0:2] + np.array(context.policy_matrix[i, j - 1, 0:2])
        if context.policy_matrix[i, j - 1, 2] == 1:
          context.curr_trace_set[i, j, 2] = context.curr_trace_set[i, j - 1, 2] - context.policy_matrix[i, j - 1, 2]
        # 补水
        elif context.policy_matrix[i, j - 1, 2] == -1:
          context.curr_trace_set[i, j, 2] = context.curr_trace_set[i, j - 1, 2] - context.policy_matrix[i, j - 1, 2] * context.Setting.replenish_speed
          if context.curr_trace_set[i, j, 2] >= context.Setting.water_volume:
            context.curr_trace_set[i, j, 2] = context.Setting.water_volume
        # 其他
        else:
          context.curr_trace_set[i, j, 2] = context.curr_trace_set[i, j - 1, 2]

# def SimulatedAnnealing(origin_mc_context: GridMovingContext, *, n_playout=10000, initial_temp=1, k=0.95, bound=100, min_temp= 0.001, mini_step=1):
def SimulatedAnnealing(rng, origin_mc_context: GridMovingContext, *,enough_info = None, n_playout=10000, initial_temp=1, k=0.95, bound=100, min_temp= 0.001, 
                        object = 1, object_mi = 50):
  #注意，这里的n_playout与singleplayou
  sq_list = []
  curr_turns = 0
  curr_k = 1
  # try:
  Temp = initial_temp
  curr_context = copy.deepcopy(origin_mc_context)
  curr_context.Setting.accept_rate = []

  while(curr_turns < bound):
    iters = 0
    curr_turns += 1
    # print(curr_turns)
    if(curr_k >= min_temp):
      curr_k = k * curr_k
    while(iters < n_playout):
      iters += 1
      # 分类  #############################################
      if object == 1: # 仅信息目标
        rand_category = 0
      if object == 2: # 仅洒水目标，仅位置
        rand_category = 0
      elif object == 3: # 仅洒水目标，仅顺序
        rand_category = 1
      elif object == 4: # 仅洒水目标，同时
        rand_category = rng.randint(0, 2)

      # 生成操作子需要的随机数 ################################
      rand_spray_category = rng.randint(0, 3)
      rand_agent = rng.randint(0, curr_context.GetAgentNumber())
      rand_time = rng.randint(0, curr_context.GetMaxTime())
      SprayTime = curr_context.GetSprayTime(rand_agent)
      if SprayTime == 0:
        rand_time1 = 0
      else:
        rand_time1 = rng.randint(0, SprayTime)
      DontSprayTime = curr_context.GetDontSprayTime(rand_agent)
      if DontSprayTime == 0:
        rand_time2 = 0
      else:
        rand_time2 = rng.randint(0, DontSprayTime)
      rand_action = rng.randint(0, curr_context.GetPossibleActions())

      # try action
      agent_position_list = None
      New_policy = None
      if rand_category == 0:
        # 调整位置
        agent_position_list, New_policy = try_move(curr_context, rand_agent, rand_time, rand_action)
      else:
        # 调整洒水动作
        if rand_spray_category == 0:
          New_policy = try_spray0(rng,curr_context, rand_agent, rand_time1)
        elif rand_spray_category == 1:
          New_policy = try_spray1(rng,curr_context, rand_agent, rand_time2) 
        elif rand_spray_category == 2:
          New_policy = try_spray2(rng,curr_context, rand_agent, rand_time2)
      
      # 根据随机选择的动作操作智能体轨迹
      # 分类执行
      new_mc_context = copy.deepcopy(curr_context)
      # print(rand_category,rand_spray_category)
      if rand_category == 0:
        # 调整位置
        if(agent_position_list is None):
          continue
        do_move(new_mc_context, rand_agent, rand_time, New_policy, agent_position_list)
      elif rand_category == 1:
        # 调整洒水
        if rand_spray_category == 0:
          if(New_policy is None):
            continue
          do_spray(new_mc_context, rand_agent, New_policy)  
        elif rand_spray_category == 1:
          if(New_policy is None):
            continue
          do_spray(new_mc_context, rand_agent, New_policy)  
        elif rand_spray_category == 2:
          if(New_policy is None):
            continue
          do_spray(new_mc_context, rand_agent, New_policy)  

      # 结果接收
      # 仅使用信息目标作为接收标准
      if object == 1:
        MI_before = curr_context.CalculateMISQ()
        MI_after = new_mc_context.CalculateMISQ()
        delta_MI = MI_after - MI_before
        if(delta_MI >= 0):
          curr_context = new_mc_context
        else:
          # accept by chance
          accept_prob = np.exp(delta_MI / (curr_k * Temp[0]))
          if(rng.random() < accept_prob):
            curr_context = new_mc_context

      # 仅以洒水目标
      # 仅位置，仅在此时需要考虑计算信息目标
      elif object == 2:
        MI_after = new_mc_context.CalculateMISQ()
        if MI_after < object_mi[rand_agent]:
          MI_before = curr_context.CalculateMISQ()
          delta_MI = MI_after - MI_before  
          if delta_MI >= 0:
            curr_context = new_mc_context
          elif delta_MI < 0:
            accept_prob = np.exp(delta_MI / (curr_k * Temp[0]))
            if(rng.random() < accept_prob):
              curr_context = new_mc_context
        else:
          sprayeffect_before = curr_context.CalculateSpraySQ(method = 2)
          sprayeffect_after = new_mc_context.CalculateSpraySQ(method = 2)
          delta_sprayeffect = sprayeffect_after - sprayeffect_before
          if delta_sprayeffect >= 0:
            curr_context = new_mc_context
            curr_context.Setting.accept_rate.append(1)
          elif delta_sprayeffect < 0:
            accept_prob = np.exp(delta_sprayeffect / (curr_k * Temp[1]))
            if(rng.random() < accept_prob):
              curr_context = new_mc_context
            curr_context.Setting.accept_rate.append(accept_prob)

      # 3仅顺序 
      elif object == 3:
        sprayeffect_before = curr_context.CalculateSpraySQ(method = 2)
        sprayeffect_after = new_mc_context.CalculateSpraySQ(method = 2)
        # if sprayeffect_after > 300:
        #   print(sprayeffect_after)
        delta_sprayeffect = sprayeffect_after - sprayeffect_before
        if delta_sprayeffect >= 0:
          curr_context = new_mc_context
          curr_context.Setting.accept_rate.append(1)
        elif delta_sprayeffect < 0:
          accept_prob = np.exp(delta_sprayeffect / (curr_k * Temp[1]))
          if(rng.random() < accept_prob):
            curr_context = new_mc_context
          curr_context.Setting.accept_rate.append(accept_prob)
      # 4同时,在改变位置时需要考虑信息目标
      elif object == 4:
        if rand_category == 0:
          MI_after = new_mc_context.CalculateMISQ()
          if MI_after < object_mi[rand_agent]:
            MI_before = curr_context.CalculateMISQ()
            delta_MI = MI_after - MI_before  
            if delta_MI >= 0:
              curr_context = new_mc_context
            elif delta_MI < 0:
              accept_prob = np.exp(delta_MI / (curr_k * Temp[0]))
              if(rng.random() < accept_prob):
                curr_context = new_mc_context
          else:
            sprayeffect_before = curr_context.CalculateSpraySQ(method = 2)
            sprayeffect_after = new_mc_context.CalculateSpraySQ(method = 2)
            delta_sprayeffect = sprayeffect_after - sprayeffect_before
            if delta_sprayeffect >= 0:
              curr_context = new_mc_context
              curr_context.Setting.accept_rate.append(1)
            elif delta_sprayeffect < 0:
              accept_prob = np.exp(delta_sprayeffect / (curr_k * Temp[1]))
              if(rng.random() < accept_prob):
                curr_context = new_mc_context
              curr_context.Setting.accept_rate.append(accept_prob)
        else:
          sprayeffect_before = curr_context.CalculateSpraySQ(method = 2)
          sprayeffect_after = new_mc_context.CalculateSpraySQ(method = 2)
          delta_sprayeffect = sprayeffect_after - sprayeffect_before
          if delta_sprayeffect >= 0:
            curr_context = new_mc_context
            curr_context.Setting.accept_rate.append(1)
          elif delta_sprayeffect < 0:
            accept_prob = np.exp(delta_sprayeffect / (curr_k * Temp[1]))
            if(rng.random() < accept_prob):
              curr_context = new_mc_context
            curr_context.Setting.accept_rate.append(accept_prob)
    MI_after = curr_context.CalculateMISQ()
    sq_list.append(MI_after)
  return curr_context, sq_list

# @PrintExecutionTime
def SimulatedAnnealingInitual(rng, origin_context: GridMovingContext,bound0,bound1,bound2,bound3,alpha,currentstep,agent_scores):
  # 洒水车规划算法，双目标
  # 计算当前的分数并储存
  MI_before = origin_context.CalculateMISQ()
  sq_list_total = []
  sq_list_total.append(MI_before)

  # 第0阶段：先进行纯信息目标探索
  # 无信息目标要求
  time1 = tm.time()
  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  # single_playout = 50
  Info_Temp = 30
  Spray_Temp = 50
  Temp = [Info_Temp, Spray_Temp]
  k = math.pow(0.0002, 1 / bound0)
  context, _ = SimulatedAnnealing(rng,origin_context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound0, object = 1)
  mi_high = context.CalculateMISQ()
  time2 = tm.time()
  print("信息搜索耗时1")
  print(time2-time1)
  
  # 然后进行综合规划
  # 首先计算信息目标，并且对不同车辆划分角色
  object_mi = np.zeros(context.GetAgentNumber())
  # 对agent_scores排序
  sorted_indices = sorted(
    range(len(agent_scores)),
    key=lambda i: (agent_scores[i], i)  # 先按分数升序，再按原索引升序
  )
  rank = [0] * len(agent_scores)
  for pos, idx in enumerate(sorted_indices):
    rank[idx] = pos
  for i in range(context.GetAgentNumber()):
    object_mi[i] = mi_high * alpha * 1.4 *(0.95**currentstep)*(0.95**rank[i])
  print("当前车辆分数")
  print(agent_scores)
  print("车辆得分排名")
  print(rank)
  print(object_mi)
  # import sys
  # sys.exit()

  # # 然后分步规划
  # single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  # Info_Temp = 30
  # Spray_Temp = 50
  # Temp = [Info_Temp, Spray_Temp]
  # k = math.pow(0.8, 1 / bound1)
  # context, _ = SimulatedAnnealing(rng,origin_context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound1, object = 2, object_mi = object_mi)
  # time3 = tm.time()
  # print("Stage1 耗时")
  # print(time3-time2)

  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Info_Temp = 10
  Spray_Temp = 20
  Temp = [Info_Temp, Spray_Temp]
  k = math.pow(0.0002, 1 / bound3)
  context, sq_list = SimulatedAnnealing(rng, context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound3, object = 2, object_mi = object_mi)
  time4 = tm.time()
  print("Stage1 耗时")
  print(time4-time2)

  # 对一阶段的信息指标进行画图：
  # fig, ax = plt.subplots(1, 1, figsize=(8, 5))  # 5行4列的子图布局，可以根据需要调整大小
  # ax.plot(sq_list_total+sq_list)
  # ax.set_ylim([-10, 40])
  # ax.set_title(f"sq_list_total")
  # plt.tight_layout()
  # plt.show()

  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Info_Temp = 10
  Spray_Temp = 100
  Temp = [Info_Temp, Spray_Temp]
  k = math.pow(0.0002, 1 / bound2)
  context, _ = SimulatedAnnealing(rng, context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound2, object = 3, object_mi = object_mi)
  time5 = tm.time()
  print("Stage2 耗时")
  print(time5-time4)
  
  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Info_Temp = 10
  Spray_Temp = 20
  # Spray_Temp = np.max((20 - origin_context.Setting.current_step * 3,5))
  Temp = [Info_Temp, Spray_Temp]
  k = math.pow(0.0002, 1 / bound2)
  context, _ = SimulatedAnnealing(rng, context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound2, object = 4, object_mi = object_mi)
  time6 = tm.time()
  print("Stage3 耗时")
  print(time6-time5)

  return context, sq_list_total + sq_list

def SimulatedAnnealingProcess(rng, origin_context: GridMovingContext, bound0, bound2, bound3, alpha, currentstep,agent_scores):
  # 洒水车规划算法，假设环境已知，以洒水收益微单目标进行长周期多动作规划
  # 计算当前的分数并储存
  MI_before = origin_context.CalculateMISQ()
  sq_list_total = []
  sq_list_total.append(MI_before)

  # 第0阶段：先进行纯信息目标探索
  time1 = tm.time()
  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Info_Temp = 20
  Spray_Temp = 50
  Temp = [Info_Temp, Spray_Temp]
  k = math.pow(0.0002, 1 / bound3)
  context, _ = SimulatedAnnealing(rng,origin_context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound3, object = 1)
  mi_high = context.CalculateMISQ()
  time2 = tm.time()
  print("信息搜索耗时1")
  print(time2-time1)
  
  # 然后进行综合规划
  # 首先计算信息目标
  object_mi = np.zeros(context.GetAgentNumber())
  # 对agent_scores排序
  sorted_indices = sorted(
    range(len(agent_scores)),
    key=lambda i: (agent_scores[i], i)  # 先按分数升序，再按原索引降序
  )
  rank = [0] * len(agent_scores)
  for pos, idx in enumerate(sorted_indices):
    rank[idx] = pos
  for i in range(context.GetAgentNumber()):
    object_mi[i] = mi_high * alpha *(0.985**currentstep)*(0.93**rank[i])
  print("当前车辆分数")
  print(agent_scores)
  print("车辆得分排名")
  print(rank)
  print(object_mi)
  
  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Info_Temp = 10
  Spray_Temp = 20
  Temp = [Info_Temp, Spray_Temp]
  k = math.pow(0.0002, 1 / bound3)
  context, sq_list = SimulatedAnnealing(rng, context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound3, object = 2, object_mi = object_mi)
  time3 = tm.time()
  print("Stage1 耗时")
  print(time3-time2)

  # 对一阶段的信息指标进行画图：
  # print("currentstep")
  # print(currentstep)
  # fig, ax = plt.subplots(1, 1, figsize=(8, 5))  # 5行4列的子图布局，可以根据需要调整大小
  # ax.plot(sq_list_total+sq_list)
  # ax.set_ylim([-10, 70])
  # ax.set_title(f"sq_list_total")
  # plt.tight_layout()
  # plt.show()

  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Info_Temp = 10
  Spray_Temp = 200
  Temp = [Info_Temp, Spray_Temp]
  k = math.pow(0.0002, 1 / bound2)
  context, _ = SimulatedAnnealing(rng, context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound2, object = 3, object_mi = object_mi)
  time4 = tm.time()
  print("Stage2 耗时")
  print(time4-time3)
  
  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Info_Temp = 10
  Spray_Temp = 20
  # Spray_Temp = np.max((20 - origin_context.Setting.current_step * 3,5))
  Temp = [Info_Temp, Spray_Temp]
  k = math.pow(0.0002, 1 / bound2)
  context, _ = SimulatedAnnealing(rng, context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound2, object = 4, object_mi = object_mi)
  time5 = tm.time()
  print("Stage3 耗时")
  print(time5-time4)

  return context, sq_list_total + sq_list

#定义SA算法包装
class SADualObjectScheduling(IStrategy):
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
            Bounding box of the sampling task workspace.shou
        rng: np.random.RandomState
            Random number generator if `get` has random operations.
        vehicle_team: dict
            team of vehicle.

        """
        super().__init__(task_extent, rng)
        self.vehicle_team = vehicle_team
        self.moving_context = None
        self.confidence = 0.5

        
    def get(self, model: IModel, Setting, pred) -> np.ndarray:
        """Get goal states for sampling.

        Parameters
        ----------
        model: IModel, optional
            A probabilistic model that provides `mean` and `std` via `forward`.
        Setting: Congif Class

        Returns
        -------
        result: dict, id:(goal_states,spray_flag)
            Sampling goal states and spray_flag

        """
        print('current_turn')
        print((Setting.current_step + Setting.adaptive_step)/Setting.adaptive_step)
        # 计算每辆车周围一圈的不确定性和污染浓度 ############################################################
        allstate_list_forinfor = []
        allstate_list_forpred = []
        for i in range (self.task_extent[0],self.task_extent[1]):
            for j in range (self.task_extent[2],self.task_extent[3]):
                allstate_list_forpred.append([i, j, model.time_stamp])
                allstate_list_forinfor.append([i, j, model.time_stamp])
        allstate_forinfor = np.array(allstate_list_forinfor)
        allstate_forpred = np.array(allstate_list_forpred)
        
        #compute predict mean and spray_effect of all point
        mean, _ = model(allstate_forpred)
        sprayeffect_all = spray_effect(allstate_forpred,allstate_forpred,mean,self.task_extent).ravel()
        
        #compute mi of all points
        prior_diag_std, poste_diag_std, poste_cov, poste_cov = model.prior_poste(allstate_forinfor)
        hprior = gaussian_entropy(prior_diag_std.ravel())
        hposterior = gaussian_entropy(poste_diag_std.ravel())
        mi_all = hprior - hposterior
        if np.any(mi_all < 0.0):
            print(mi_all.ravel())
            raise ValueError("Predictive MI < 0.0!")
        
        #标准化
        if np.all(mi_all == 0.0):
            normed_mi = np.ones_like(mi_all)
        else:
            # normed_mi = (mi_all.max() - mi_all) / mi_all.ptp()
            normed_mi = (mi_all - mi_all.min()) / mi_all.ptp()
        normed_effect = (sprayeffect_all - sprayeffect_all.min()) / sprayeffect_all.ptp()
        # trans to matrix form
        mi = np.zeros((self.task_extent[1]-self.task_extent[0],self.task_extent[3]-self.task_extent[2]))
        sprayeffect = np.zeros((self.task_extent[1]-self.task_extent[0],self.task_extent[3]-self.task_extent[2]))

        for i in range (self.task_extent[0],self.task_extent[1]):
            for j in range (self.task_extent[2],self.task_extent[3]):
                mi[i,j] = normed_mi[i*(self.task_extent[3]-self.task_extent[2])+j]
                sprayeffect[i,j] = normed_effect[i*(self.task_extent[3]-self.task_extent[2])+j]
        scores = self.confidence*sprayeffect + (1-self.confidence)*mi
        # 计算每个车辆的得分
        agent_init_position = []
        agent_scores=[]
        for id, vehicle in self.vehicle_team.items():
          agent_init_position.append(vehicle.state[0:2])
          score = 0
          for a in range(3):
            for b in range(3):
              c1 = vehicle.state[0] - 1 + a
              c2 = vehicle.state[1] - 1 + b
              if c1 < self.task_extent[0] or c1 >= self.task_extent[1] or c2 < self.task_extent[2] or c2 >= self.task_extent[3]:
                continue
              else:
                score = score + scores[int(c1),int(c2)]
          agent_scores.append(score)

        # 正式洒水规划
        # 计算当前需要规划的步数
        sche_step = 0
        if Setting.current_step > Setting.max_num_samples - Setting.sche_step:
          if Setting.max_num_samples - Setting.current_step > 7:
            sche_step = Setting.max_num_samples - Setting.current_step
          else:
            sche_step = 8
        else:
          sche_step = Setting.sche_step
        # 计算用于规划的目标集合 阶梯式的非均匀##############################################################
        allpoint_list = []
        layer = 0
        while Setting.adaptive_step*layer < Setting.sche_step and layer <= 2:
          if layer == 0:
              xyinterval = (self.task_extent[1]-self.task_extent[0])/(Setting.layer_xy[0])
              tinterval = Setting.adaptive_step/(Setting.layer_t[0])
              for num in np.arange(Setting.adaptive_step*layer+0.5*tinterval, Setting.adaptive_step*(layer+1), tinterval):
                  for i in np.arange (self.task_extent[0]+0.5*xyinterval,self.task_extent[1],xyinterval):
                      for j in np.arange (self.task_extent[0]+0.5*xyinterval,self.task_extent[1],xyinterval):
                          allpoint_list.append([i, j, model.time_stamp + num * Setting.time_co])
          elif layer > 0 and layer <= 1:
              xyinterval = (self.task_extent[1]-self.task_extent[0])/(Setting.layer_xy[1])
              tinterval = Setting.adaptive_step/(Setting.layer_t[1])
              for num in np.arange(Setting.adaptive_step*layer+0.5*tinterval, Setting.adaptive_step*(layer+1), tinterval):
                  for i in np.arange (self.task_extent[0]+0.5*xyinterval,self.task_extent[1],xyinterval):
                      for j in np.arange (self.task_extent[0]+0.5*xyinterval,self.task_extent[1],xyinterval):
                          allpoint_list.append([i, j, model.time_stamp + num * Setting.time_co])
          elif layer > 1:
              xyinterval = (self.task_extent[1]-self.task_extent[0])/Setting.layer_xy[2]
              tinterval = (Setting.sche_step - Setting.adaptive_step*layer)/(Setting.layer_t[2])
              for num in np.arange(Setting.adaptive_step*layer+0.5*tinterval, Setting.sche_step, tinterval):
                  for i in np.arange (self.task_extent[0]+0.5*xyinterval,self.task_extent[1],xyinterval):
                      for j in np.arange (self.task_extent[0]+0.5*xyinterval,self.task_extent[1],xyinterval):
                          allpoint_list.append([i, j, model.time_stamp + num * Setting.time_co])
          layer = layer + 1
        allpoint = np.array(allpoint_list)
        # print("分层情况")
        # print(allpoint)
        # import sys
        # sys.exit()

        if self.moving_context is None:
          agent_init_position = []
          for id, vehicle in self.vehicle_team.items():
            agent_init_position.append(vehicle.state[0:2])
          agent_init_position = np.array(agent_init_position)
          self.moving_context = GridMovingContext(agent_init_position, model, pred, allpoint, Setting)
          self.alpha = Setting.alpha
          self.moving_context, sq_list_total = SimulatedAnnealingInitual(self.rng, self.moving_context, Setting.bound0, Setting.bound1, Setting.bound2, Setting.bound3, self.alpha, Setting.current_step, agent_scores)
        else:
          self.moving_context.adaptive_update(model, pred, allpoint, Setting)
          self.moving_context, sq_list_total = SimulatedAnnealingProcess(self.rng, self.moving_context, Setting.bound0, Setting.bound2, Setting.bound3, self.alpha, Setting.current_step, agent_scores)
        
        #context中包含最后的结果
        policy_now = self.moving_context.policy_matrix.copy()
        agent_position_list = self.moving_context.curr_trace_set.copy()
                
        result = dict()

        for id, vehicle in self.vehicle_team.items():
          goal_states = np.zeros((sche_step,2))
          spray_states = np.ones((sche_step,1))

          # Append waypoint
          for index in range(sche_step):
            goal_states[index,0] = agent_position_list[id-1,index+1,0]
            goal_states[index,1] = agent_position_list[id-1,index+1,1]
            spray_states[index,0] = policy_now[id-1,index,2]
            
          result[id] = (goal_states,spray_states)
                
        return result
    