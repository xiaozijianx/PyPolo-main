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


class NonMyopicLatticePlanningMISprinkler(IStrategy):
    """Myopic informative planning based on Mutual informaiton on latttice map.
        CASE STUDY FOR forestfire
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
        self.confidence = np.zeros((self.task_extent[1]-self.task_extent[0],self.task_extent[3]-self.task_extent[2]))
        self.omega = [[[] for _ in range(self.task_extent[1]-self.task_extent[0])] for _ in range(self.task_extent[3]-self.task_extent[2])]
        self.sigma_omega = 1
        
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
        
    def get(self, model: IModel, Setting) -> np.ndarray:
        # predict model
        # use gpr to predict，
        allstate_list_forinfor = []
        allstate_list_forpred = []
        for i in range (self.task_extent[0],self.task_extent[1]):
            for j in range (self.task_extent[2],self.task_extent[3]):
                # allstate_list_forpred.append([i, j, model.time_stamp + Setting.adaptive_step * Setting.time_co])
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
        
        # Processing
        mi = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        pred = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        sprayeffect = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        for i in range (Setting.task_extent[0],Setting.task_extent[1]):
            for j in range (Setting.task_extent[2],Setting.task_extent[3]):
                mi[i,j] = mi_all[i*(Setting.task_extent[3]-Setting.task_extent[2])+j]
                pred[i,j] = mean[i*(Setting.task_extent[3]-Setting.task_extent[2])+j,0]
                sprayeffect[i,j] = sprayeffect_all[i*(Setting.task_extent[3]-Setting.task_extent[2])+j]
        
        # record the time series results
        result_mean = pred.copy()
        result_mi_all = mi.copy()
        result_sprayeffect_all = sprayeffect.copy()
        
        self.update_confidence()
        result = dict()
        for id, vehicle in self.vehicle_team.items():
            # Normalized scores
            # if mi_all = 0,normed_mi = 1
            if np.all(mi_all == 0.0):
                normed_mi = np.ones_like(mi_all)
            else:
                normed_mi = (mi_all.max() - mi_all) / mi_all.ptp()
            #change the normaliz method
            normed_effect = sprayeffect_all / 100.0
            # normed_effect = (sprayeffect_all - sprayeffect_all.min()) / sprayeffect_all.ptp()
            
            # trans to matrix form
            mi = np.zeros((self.task_extent[1]-self.task_extent[0],self.task_extent[3]-self.task_extent[2]))
            sprayeffect = np.zeros((self.task_extent[1]-self.task_extent[0],self.task_extent[3]-self.task_extent[2]))
            
            #set threshold that sprayeffect under this threshold means don't spray
            threshold = Setting.effect_threshold
            for i in range (self.task_extent[0],self.task_extent[1]):
                for j in range (self.task_extent[2],self.task_extent[3]):
                    mi[i,j] = normed_mi[i*(self.task_extent[3]-self.task_extent[2])+j]
                    sprayeffect[i,j] = normed_effect[i*(self.task_extent[3]-self.task_extent[2])+j]
            if Setting.strategy_name == "forestfire_Nonmyopic_Adaptive_Operation":
                scores = sprayeffect
            else:
                scores = self.confidence*sprayeffect + (1-self.confidence)*mi
            path = self.greedy_search_multi_step(Setting.sche_step, scores, id)[1:]

            goal_states = np.zeros((len(path),2))
            spray_flag = np.ones((len(path),1), dtype=bool)
            
            # Append waypoint
            for index, location in enumerate(path):
                goal_states[index,0] = location[0]
                goal_states[index,1] = location[1]
                #find point under threshold
                if sprayeffect[location[0],location[1]] < threshold:
                    spray_flag[index,0] = False
            
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
                       
        return result, result_mi_all, result_mean, result_sprayeffect_all
    
    def update_confidence(self):
        for i in range(self.confidence.shape[0]):
            for j in range(self.confidence.shape[1]):
                for _, vehicle in self.vehicle_team.items():
                    dist2robot = self.get_chebyshev_dist((i, j), vehicle.state[0:2])
                    self.omega[i][j].append(ss.norm.pdf(dist2robot, loc=0, scale=1))

                while len(self.omega[i][j]) > 50:
                    self.omega[i][j].pop(0)
                    
                self.confidence[i][j] = 1 - np.exp(-np.sum(self.omega[i][j]) / self.sigma_omega**2)
        self.confidence = np.clip(self.confidence, 0, 1)
        
    def get_chebyshev_dist(self, position1, position2):
        return max(abs(position1[0] - position2[0]), abs(position1[1] - position2[1]))