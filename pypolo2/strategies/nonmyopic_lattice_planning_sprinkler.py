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
                    graph.add_edge(edge[0], edge[1], weight=1e-4+(0.8**(length-current_length))*weights[neighbor_node[0], neighbor_node[1]])
                    # graph.add_edge(edge[0], edge[1], weight=1e-4 + weights[neighbor_node[0], neighbor_node[1]])

        if len(graph.edges()) == 1:
            raise ValueError

        path = nx.algorithms.dag_longest_path(graph)
        path_weight = sum([graph.get_edge_data(path[i], path[i+1])['weight'] for i in range(len(path)-1)])
        path = [element[0] for element in path]

        return path
        
    def get(self, model: IModel, Setting, pred) -> np.ndarray:
        # predict model
        allstate_list_forpred = []
        for i in range (self.task_extent[0],self.task_extent[1]):
            for j in range (self.task_extent[2],self.task_extent[3]):
                allstate_list_forpred.append([i, j, model.time_stamp])
        allstate_forpred = np.array(allstate_list_forpred)
        sprayeffect_all = spray_effect(allstate_forpred,allstate_forpred,pred,self.task_extent,method=2).ravel()

        # Processing
        sprayeffect = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        for i in range (Setting.task_extent[0],Setting.task_extent[1]):
            for j in range (Setting.task_extent[2],Setting.task_extent[3]):
                sprayeffect[i,j] = sprayeffect_all[i*(Setting.task_extent[3]-Setting.task_extent[2])+j]
        
        result = dict()
        for id, vehicle in self.vehicle_team.items():
            #change the normaliz method
            normed_effect = sprayeffect_all / 100.0
            # normed_effect = (sprayeffect_all - sprayeffect_all.min()) / sprayeffect_all.ptp()
            # trans to matrix form
            sprayeffect = np.zeros((self.task_extent[1]-self.task_extent[0],self.task_extent[3]-self.task_extent[2]))
            for i in range (self.task_extent[0],self.task_extent[1]):
                for j in range (self.task_extent[2],self.task_extent[3]):
                    sprayeffect[i,j] = normed_effect[i*(self.task_extent[3]-self.task_extent[2])+j]
            scores = sprayeffect
           
            path = self.greedy_search_multi_step(6, scores, id)[1:]

            goal_states = np.zeros((len(path),2))
            spray_flag = np.ones((len(path),1), dtype=bool) 
            
            # Append waypoint
            replenishment = 0
            pathlen = 0
            movestep = 0
            pathaccept = 0
            initual_state = vehicle.state
            initual_replenish = True
            replenish_flag = False
            numreplenish = 0
            for index, location in enumerate(path):
                print(index)
                if pathlen >= len(path) - 1:
                    break
                # 先判断水量
                if index < Setting.water_volume//Setting.replenish_speed:
                    if vehicle.goal_spray_flag[index] == -1 and initual_replenish == True:
                        goal_states[index,0] = initual_state[0]
                        goal_states[index,1] = initual_state[1]
                        pathlen = pathlen + 1 
                        spray_flag[index,0] = False
                        continue
                    else:
                        initual_replenish = False
                if index >= Setting.water_volume//Setting.replenish_speed or initual_replenish == False:
                    if vehicle.water_volume_now - movestep < 0 or replenish_flag == True:
                        if pathaccept == 0:
                            goal_states[index,0] = initual_state[0]
                            goal_states[index,1] = initual_state[1]
                        else:
                            goal_states[index,0] = path[pathaccept-1][0]
                            goal_states[index,1] = path[pathaccept-1][0]
                        pathlen = pathlen + 1 
                        spray_flag[index,0] = False
                        replenish_flag = True
                        numreplenish = numreplenish + 1
                        if numreplenish >= Setting.water_volume//Setting.replenish_speed:
                            replenish_flag = False
                    else:
                        goal_states[index,0] = path[pathaccept][0]
                        goal_states[index,1] = path[pathaccept][1]
                        spray_flag[index,0] = True
                        pathlen = pathlen + 1 
                        movestep = movestep + 1
                        pathaccept = pathaccept + 1
            
            result[id] = (goal_states,spray_flag)
            
            #reduce effect
            for i in range(len(path)):
                if self.vehicle_team[id].water_volume_now > i:
                    for m in range(3):
                        for n in range(3):
                            r = goal_states[i,0] -1 + m
                            c = goal_states[i,1] -1 + n
                            if r < self.task_extent[0] or r >= self.task_extent[1] or c < self.task_extent[2] or c >= self.task_extent[3]:
                                continue
                            if m == 1 and n == 1:
                                mi_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]=(0.9**i*2)*mi_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]
                                if spray_flag[i,0] == True:
                                    sprayeffect_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]=(1-(0.9**i*0.2))*sprayeffect_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]
                            if spray_flag[i,0] == True:
                                sprayeffect_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]=(1-(0.9**i*0.15))*sprayeffect_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]
                            mi_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]=(0.9**i*1.5)*mi_all[int(r*(self.task_extent[3]-self.task_extent[2])+c)]
                       
        return result