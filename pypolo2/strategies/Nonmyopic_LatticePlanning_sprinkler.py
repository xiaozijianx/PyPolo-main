from typing import List

import numpy as np
import sys

from ..objectives.entropy import gaussian_entropy
from ..objectives.sprayeffect import spray_effect
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot
import scipy.ndimage as sn
import scipy.stats as ss

import networkx as nx


class NonMyopicLatticePlanningSprinkler(IStrategy):
    """
    Myopic planning on latttice map.
    only planning for spray-effect, environment is already known.
    baseline for PINNS spray model
    """

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
        num_candidates: int
            Number of candidate locations to evaluate.
        vehicle_team: dict
            team of vehicle.

        """
        super().__init__(task_extent, rng)
        self.vehicle_team = vehicle_team
        self.confidence = 0.5
        
    def greedy_search_multi_step(self, length, weights, id):
        graph = nx.DiGraph()
        position = (int(self.vehicle_team[id].state[0]),int(self.vehicle_team[id].state[1]))
        nodes = [(position, length)]

        # for each node, find other nodes that can be moved to with the remaining amount of path length
        while nodes:
            current_node, current_length = nodes.pop(0)
            if current_length == 0:
                continue

            for (dr, dc) in self.vehicle_team[id].movements:
                if (dr, dc) == (0,0):
                    continue
                
                neighbor_node = (current_node[0] + dr, current_node[1] + dc)

                neighbor = (neighbor_node, int(current_length-1))
                edge = ((current_node, current_length), neighbor)
                if graph.has_edge(edge[0], edge[1]):
                    continue
                # if graph.has_edge(edge_1[0], edge_1[1]):
                #     continue

                if self.task_extent[0] <= neighbor_node[0] < self.task_extent[1] and self.task_extent[2] <= neighbor_node[1] < self.task_extent[3]:
                    nodes.append(neighbor)
                    # graph.add_edge(edge[0], edge[1], weight=1e-4+(0.8**(length-current_length))*weights[neighbor_node[0], neighbor_node[1]])
                    graph.add_edge(edge[0], edge[1], weight=1e-4 + weights[neighbor_node[0], neighbor_node[1]])

        if len(graph.edges()) == 1:
            raise ValueError

        path = nx.algorithms.dag_longest_path(graph)
        path_weight = sum([graph.get_edge_data(path[i], path[i+1])['weight'] for i in range(len(path)-1)])
        path = [element[0] for element in path]

        return path
        
    def get(self, model: IModel, Setting, pred, agent_scores) -> np.ndarray:
        # predict model
        # 获取在当前时刻污染分布下，在每个点洒水时的效果。
        # 这种算法将每个点独立考虑，而没有考虑其收益之间的影响。
        print('current_turn')
        print((Setting.current_step + Setting.adaptive_step)/Setting.adaptive_step)

        allstate_list_forinfor = []
        allstate_list_forpred = []
        for i in range (self.task_extent[0],self.task_extent[1]):
            for j in range (self.task_extent[2],self.task_extent[3]):
                allstate_list_forpred.append([i, j, model.time_stamp])
                allstate_list_forinfor.append([i, j, model.time_stamp])
        allstate_forinfor = np.array(allstate_list_forinfor)
        allstate_forpred = np.array(allstate_list_forpred)
        # print(allstate_forpred.shape)
        # print(pred.shape)
        # compute predict mean and spray_effect of all point
        mean, _ = model(allstate_forpred)
        sprayeffect_all = spray_effect(allstate_forpred,allstate_forpred,mean,self.task_extent).ravel()

        #compute mi of all points
        prior_diag_std, poste_diag_std, _, _ = model.prior_poste(allstate_forinfor)
        hprior = gaussian_entropy(prior_diag_std.ravel())
        hposterior = gaussian_entropy(poste_diag_std.ravel())
        mi_all = hprior - hposterior
        if np.any(mi_all < 0.0):
            print(mi_all.ravel())
            raise ValueError("Predictive MI < 0.0!")
        
        result = dict()
        for id, vehicle in self.vehicle_team.items():
            # processing
            if np.all(mi_all == 0.0):
                normed_mi = np.ones_like(mi_all)
            else:
                normed_mi = (mi_all.max() - mi_all) / mi_all.ptp()
                # normed_mi = (mi_all - mi_all.min()) / mi_all.ptp()
            normed_effect = (sprayeffect_all - sprayeffect_all.min()) / sprayeffect_all.ptp()
            # trans to matrix form
            mi = np.zeros((self.task_extent[1]-self.task_extent[0],self.task_extent[3]-self.task_extent[2]))
            sprayeffect = np.zeros((self.task_extent[1]-self.task_extent[0],self.task_extent[3]-self.task_extent[2]))
            for i in range (self.task_extent[0],self.task_extent[1]):
                for j in range (self.task_extent[2],self.task_extent[3]):
                    mi[i,j] = normed_mi[i*(self.task_extent[3]-self.task_extent[2])+j]
                    sprayeffect[i,j] = normed_effect[i*(self.task_extent[3]-self.task_extent[2])+j]
            scores = self.confidence*sprayeffect + (1-self.confidence)*mi
           
            path = self.greedy_search_multi_step(6, scores, id)[1:]

            goal_states = np.zeros((len(path),2))
            spray_flag = np.ones((len(path),1)) 
            
            # Append waypoint
            replenishment = 0
            pathlen = 0
            movestep = 0
            pathaccept = 0
            initual_state = vehicle.state
            initual_replenish = True
            replenish_flag = False
            numreplenish = 0
            water_volume_now = vehicle.water_volume_now
            # print(path)
            for index, location in enumerate(path):
                # print(index)
                if pathlen > len(path) - 1:
                    break
                if index < Setting.water_volume//Setting.replenish_speed:
                    if len(vehicle.goal_spray_flag) != 0 and initual_replenish == True and vehicle.goal_spray_flag[0] == -1 and water_volume_now < Setting.water_volume:
                        goal_states[index,0] = initual_state[0]
                        goal_states[index,1] = initual_state[1]
                        pathlen = pathlen + 1 
                        spray_flag[index,0] = -1
                        water_volume_now = water_volume_now + Setting.replenish_speed
                        continue
                    else:
                        initual_replenish = False
                if index >= Setting.water_volume//Setting.replenish_speed or initual_replenish == False:
                    if water_volume_now - movestep <= 0 or replenish_flag == True:
                        if pathaccept == 0:
                            goal_states[index,0] = initual_state[0]
                            goal_states[index,1] = initual_state[1]
                        else:
                            goal_states[index,0] = path[pathaccept-1][0]
                            goal_states[index,1] = path[pathaccept-1][1]
                        pathlen = pathlen + 1 
                        spray_flag[index,0] = -1
                        water_volume_now = water_volume_now + Setting.replenish_speed
                        replenish_flag = True
                        numreplenish = numreplenish + 1
                        if numreplenish >= Setting.water_volume//Setting.replenish_speed:
                            replenish_flag = False
                    else:
                        goal_states[index,0] = path[pathaccept][0]
                        goal_states[index,1] = path[pathaccept][1]
                        spray_flag[index,0] = 1
                        pathlen = pathlen + 1 
                        movestep = movestep + 1
                        pathaccept = pathaccept + 1
            
            result[id] = (goal_states,spray_flag)
            
            #reduce effect
            MIMAX = mi_all.max()
            for i in range(len(path)):
                for m in range(5):
                    for n in range(5):
                        r = goal_states[i,0] -2 + m
                        c = goal_states[i,1] -2 + n
                        if r < self.task_extent[0] or r >= self.task_extent[1] or c < self.task_extent[2] or c >= self.task_extent[3]:
                            continue
                        if m == 2 and n == 2:
                            if spray_flag[i,0] == True:
                                sprayeffect_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]=(1-(0.6))*sprayeffect_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]
                            mi_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)] = 0.9*MIMAX
                        else:
                            if spray_flag[i,0] == True:
                                sprayeffect_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]=(1-(0.4))*sprayeffect_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]
        return result