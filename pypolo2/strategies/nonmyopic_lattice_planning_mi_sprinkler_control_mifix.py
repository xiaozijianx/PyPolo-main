from typing import List

import numpy as np

from ..objectives.entropy import gaussian_entropy
from ..objectives.sprinkeffect import sprink_effect
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot

import networkx as nx
import sys


class NonMyopicLatticePlanningMISprinklerControlFix(IStrategy):
    """Myopic informative planning based on Mutual informaiton on latttice map."""

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
        
    def greedy_search_multi_step(self, length, alpha, sprinkeffect_all, model, allstate):
        """Get goal states for sampling.

        Parameters
        ----------
        length: int
            Number of goal states.
        alpha: float
            weight between mi and effect
        Returns
        -------
        path: list
            goal states.

        """
        result = dict()
        for id, vehicle in self.vehicle_team.items():
            # Normalized scores
            normed_effect = (sprinkeffect_all - sprinkeffect_all.min()) / sprinkeffect_all.ptp()
            #trans to matrix form
            sprinkeffect = np.zeros((self.task_extent[1]+1-self.task_extent[0],self.task_extent[3]+1-self.task_extent[2]))
            #set threshold that sprinkeffect under this threshold means don't spray
            threshold = 25
            for i in range (self.task_extent[0],self.task_extent[1]+1):
                for j in range (self.task_extent[2],self.task_extent[3]+1):
                    if sprinkeffect_all[i*(self.task_extent[3]+1-self.task_extent[2])+j] > threshold:
                        sprinkeffect[i,j] = normed_effect[i*(self.task_extent[3]+1-self.task_extent[2])+j]

            # result = self.greedy_search_multi_step(8)[1:]
            graph = nx.DiGraph()
            position = (int(vehicle.state[0]),int(vehicle.state[1]))
            nodes = [(position, length)]

            # for each node, find other nodes that can be moved to with the remaining amount of path length
            return_flag = False
            last_length = length + 1
            while nodes:
                current_node, current_length = nodes.pop(-1)
                print(current_length)
                if current_length == 0:
                    continue
                #calculate new mi
                if current_length != length:
                    if current_length >= last_length:
                        num_return = 4 * (current_length - last_length + 1)
                        model.reduce_data_x(int(num_return))
                        # x_train = x_train[0:int(num_return), :]
                        return_flag = False
                    new_x = np.array([current_node[0],current_node[1],(model.time_stamp+0.0001*(length-current_length))]).reshape(1,-1)
                    x_station = np.zeros((3,2))
                    x_station[0,0] = 14.0
                    x_station[0,1] = 6.0
                    x_station[1,0] = 8.0
                    x_station[1,1] = 16.0
                    x_station[2,0] = 4.0
                    x_station[2,1] = 6.0
                    x_station_time = np.zeros((3,3))
                    x_station_time[:,0:2] = x_station
                    x_station_time[:,2] = model.time_stamp + 0.0001*(length-current_length)
                    new_x = np.vstack((new_x, x_station_time))
                    model.add_data_x(new_x)
                    # x_train = np.vstack((x_train, np.array([current_node[0],current_node[1],(model.time_stamp+0.0001*(length-current_length))]).reshape(1,-1)))
                    # x_train = np.vstack((x_train, x_station_time))
                
                candidatestate_list = []
                for i in range (9):
                    for j in range (9):
                        r = current_node[0] - 4 + i
                        c = current_node[1] - 4 + j
                        if r < self.task_extent[0] or r > self.task_extent[1] or c < self.task_extent[2] or r > self.task_extent[3]:
                            continue
                        candidatestate_list.append([r, c, model.time_stamp + 0.0001*(length-current_length)])
                allstate = np.array(candidatestate_list).copy()
                prior_diag_std, poste_diag_std, poste_cov, poste_cov = model.prior_poste(allstate)

                hprior = gaussian_entropy(prior_diag_std.ravel())
                hposterior = gaussian_entropy(poste_diag_std.ravel())
                mi_all = hprior - hposterior
                if np.any(mi_all < 0.0):
                    print(mi_all.ravel())
                    raise ValueError("Predictive MI < 0.0!")
                
                normed_mi = (mi_all.max() - mi_all) / mi_all.ptp()
                #trans to matrix form
                mi = np.zeros((self.task_extent[1]+1-self.task_extent[0],self.task_extent[3]+1-self.task_extent[2]))
                # for i in range (self.task_extent[0],self.task_extent[1]+1):
                #     for j in range (self.task_extent[2],self.task_extent[3]+1):
                #         mi[i,j] = normed_mi[i*(self.task_extent[3]+1-self.task_extent[2])+j]
                        
                for i in range (9):
                    for j in range (9):
                        r = current_node[0] - 4 + i
                        c = current_node[1] - 4 + j
                        if r < self.task_extent[0] or r > self.task_extent[1] or c < self.task_extent[2] or r > self.task_extent[3]:
                            continue
                        if normed_mi.shape[0] > 0:
                            mi[r,c] = normed_mi[0]  
                            normed_mi = normed_mi[1:]
                        else:
                            mi[r,c] = normed_mi[0] 
                
                scores = alpha*mi + (1-alpha)*sprinkeffect
                for (dr, dc) in self.vehicle_team[id].movements:
                    if (dr, dc) == (0,0):
                        continue
                    
                    neighbor_node = (current_node[0] + dr, current_node[1] + dc)
                    neighbor = (neighbor_node, int(current_length-1))
                    edge = ((current_node, current_length), neighbor)
                    # if graph.has_edge(edge[0], edge[1]):
                    #     continue  

                    if self.task_extent[0] <= neighbor_node[0] <= self.task_extent[1] and self.task_extent[2] <= neighbor_node[1] <= self.task_extent[3]:
                        nodes.append(neighbor)
                        graph.add_edge(edge[0], edge[1], weight=1e-4+(0.9**(length-current_length))*scores[neighbor_node[0], neighbor_node[1]])
                
                last_length = current_length

            if len(graph.edges()) == 1:
                raise ValueError

            #reduce added data
            model.reduce_data_x(4*int(length-1))

            path = nx.algorithms.dag_longest_path(graph)
            
            path_weight = sum([graph.get_edge_data(path[i], path[i+1])['weight'] for i in range(len(path)-1)])
            path = [element[0] for element in path][1:]

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
            num_spray = 0
            for i in range(len(path)):
                if spray_flag[i,0] == True:
                    num_spray = num_spray + 1
                if self.vehicle_team[id].water_volume_now > num_spray - 1:
                    for m in range(3):
                        for n in range(3):
                            r = goal_states[i,0] -1 + m
                            c = goal_states[i,1] -1 + n
                            if r < self.task_extent[0] or r > self.task_extent[1] or c < self.task_extent[2] or c > self.task_extent[3]:
                                continue
                            if m == 1 and n == 1:
                                if spray_flag[i,0] == True:
                                    sprinkeffect_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]=(1-(0.9**i*0.3))*sprinkeffect_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]
                            if spray_flag[i,0] == True:
                                sprinkeffect_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]=(1-(0.9**i*0.2))*sprinkeffect_all[int(r*(self.task_extent[3]+1-self.task_extent[2])+c)]
        return result
        
    def get(self, model: IModel, alpha = 1, step_number = 4) -> np.ndarray:
        """Get goal states for sampling.

        Parameters
        ----------
        model: IModel, optional
            A probabilistic model that provides `mean` and `std` via `forward`.
        num_states: int
            Number of goal states.

        Returns
        -------
        goal_states: np.ndarray, shape=(num_states, dim_states)
            Sampling goal states.

        """
        # Propose candidate locations
        allstate_list = []
        for i in range (self.task_extent[0],self.task_extent[1]+1):
            for j in range (self.task_extent[2],self.task_extent[3]+1):
                allstate_list.append([i, j, model.time_stamp])
        allstate = np.array(allstate_list)
        
        #compute predict mean and sprink_effect of all point
        mean, _ = model(allstate)
        sprinkeffect_all = sprink_effect(allstate,allstate,mean,self.task_extent).ravel()
        
        #compute mi of all points
        prior_diag_std, poste_diag_std, poste_cov, poste_cov = model.prior_poste(allstate)
        hprior = gaussian_entropy(prior_diag_std.ravel())
        hposterior = gaussian_entropy(poste_diag_std.ravel())
        mi_all = hprior - hposterior
        # print(mi_all)
        # sys.exit()
        if np.any(mi_all < 0.0):
            print(mi_all.ravel())
            raise ValueError("Predictive MI < 0.0!")
        
        result_mean = mean.copy()
        result_mi_all = mi_all.copy()
        result_sprinkeffect_all = sprinkeffect_all.copy()
        
        result = self.greedy_search_multi_step(step_number, alpha, sprinkeffect_all, model, allstate)
   
        return result, result_mi_all, result_mean, result_sprinkeffect_all
    
   
