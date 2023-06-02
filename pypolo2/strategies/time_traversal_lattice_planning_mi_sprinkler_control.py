from typing import List

import numpy as np

from ..objectives.entropy import gaussian_entropy, gaussian_entropy_multivariate
from ..objectives.sprinkeffect import sprink_effect
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot

import networkx as nx
import sys


class TimeTraversalLatticePlanningMISprinklerControl(IStrategy):
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
        
    def greedy_search_multi_step(self, length, alpha, sprinkeffect_all, model):
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
        num_added = 0
        x_station = np.zeros((3,2))
        x_station[0,0] = 14.0
        x_station[0,1] = 6.0
        x_station[1,0] = 8.0
        x_station[1,1] = 16.0
        x_station[2,0] = 4.0
        x_station[2,1] = 6.0
        x_station_time = np.zeros((3*length,3))
        for time in range(length):
            x_station_time[3*time:3*(time+1),0:2] = x_station
            x_station_time[3*time:3*(time+1),2] = model.time_stamp + 0.0001*(time+1)
        model.add_data_x(x_station_time)
        
        #trans to matrix form
        sprayeffect = np.zeros((self.task_extent[1]+1-self.task_extent[0],self.task_extent[3]+1-self.task_extent[2]))
        #set threshold that sprinkeffect under this threshold means don't spray
        threshold = 15
        for i in range (self.task_extent[0],self.task_extent[1]+1):
            for j in range (self.task_extent[2],self.task_extent[3]+1):
                if sprinkeffect_all[i*(self.task_extent[3]+1-self.task_extent[2])+j] > threshold:
                    sprayeffect[i,j] = sprinkeffect_all[i*(self.task_extent[3]+1-self.task_extent[2])+j]
        
        for id, vehicle in self.vehicle_team.items():
            position = (int(vehicle.state[0]),int(vehicle.state[1]))
            nodes = [(position, length)]

            # for each node, find other nodes that can be moved to with the remaining amount of path length
            last_length = length + 1
            
            candidates = []
            select_points = np.zeros((1,3))
            select_points[0,0] = vehicle.state[0]
            select_points[0,1] = vehicle.state[1]
            select_points[0,2] = model.time_stamp
            
            while nodes:
                current_node, current_length = nodes.pop(-1)
                if current_length == 0:
                    new_x = np.array([current_node[0],current_node[1],model.time_stamp+0.0001*(length-current_length)]).reshape(1,-1)
                    select_points = np.vstack((select_points, new_x))
                    #calculate spray effect
                    sprayeffect_copy = sprayeffect.copy()
                    spray_effect = 0
                    for i in range(select_points.shape[0]):
                        if i == 0:
                            continue
                        r0 = int(select_points[i, 0])
                        c0 = int(select_points[i, 1])
                        if sprayeffect_copy[r0,c0] > 0:
                            for a in range(3):
                                for b in range(3):
                                    r = int(r0 - 1 + a)
                                    c = int(c0 - 1 + b)
                                    if r >= 0 and r <= self.task_extent[1] and c >= 0 and c <= self.task_extent[3]:
                                        if a == 1 and b == 1:
                                            spray_effect = spray_effect + (1-0.1)**i * sprayeffect_copy[r,c]
                                            sprayeffect_copy[r,c] = (1 - (1-0.7)**(i+1) ) * sprayeffect_copy[r,c]
                                            if sprayeffect_copy[r,c] < threshold:
                                                sprayeffect_copy[r,c] = 0
                                        else:
                                            sprayeffect_copy[r,c] = (1 - (1-0.8)**(i+1) ) * sprayeffect_copy[r,c]
                                            if sprayeffect_copy[r,c] < threshold:
                                                sprayeffect_copy[r,c] = 0
                    
                    #calculate mi
                    #Removing duplicate points
                    processed_points = np.unique(select_points[1:], axis=0)
                    train_data = model.get_data_x()
                    
                    nrows, ncols = train_data.shape
                    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                        'formats':ncols * [train_data.dtype]}
                    mid_points = np.intersect1d(train_data.view(dtype), processed_points.view(dtype))
                    processed_points2 = np.setdiff1d(processed_points.view(dtype), mid_points)
                    processed_points2 = processed_points2.view(train_data.dtype).reshape(-1, ncols)
                    _, _, prior_cov, poste_cov = model.prior_poste(processed_points2)
                    prior_entropy = gaussian_entropy_multivariate(prior_cov)
                    poste_entropy = gaussian_entropy_multivariate(poste_cov)
                    # print(prior_entropy, poste_entropy)
                    mi = prior_entropy - poste_entropy
                    num = processed_points2.shape[0]
                    if num <= 0:
                        unit_mi = 1000
                    else:
                        unit_mi = mi/num
                    
                    scores = alpha*(-1)*unit_mi + (1-alpha)*spray_effect
                    candidates.append([select_points[1:], processed_points2, scores])
                    # print("points", select_points ,"mi", mi,"  effect", spray_effect)
                    select_points = select_points[0:-1]
                    continue
                
                #calculate new mi
                if current_length != length:
                    if current_length >= last_length:
                        num_return = current_length - last_length + 1
                        select_points = select_points[0:-1*num_return]
                        
                    new_x = np.array([current_node[0],current_node[1],model.time_stamp+0.0001*(length-current_length)]).reshape(1,-1)
                    select_points = np.vstack((select_points, new_x))
                
                for (dr, dc) in self.vehicle_team[id].movements:
                    if (dr, dc) == (0,0):
                        continue
                    
                    neighbor_node = (current_node[0] + dr, current_node[1] + dc)
                    neighbor = (neighbor_node, int(current_length-1))
                    
                    if self.task_extent[0] <= neighbor_node[0] <= self.task_extent[1] and self.task_extent[2] <= neighbor_node[1] <= self.task_extent[3]:
                        nodes.append(neighbor)
                
                last_length = current_length

            #calculate finish ,select best points
            max_selected_path = max(candidates, key=lambda x: float('-inf') if np.isnan(x[2]) else x[2])
            path = max_selected_path[0]
            process_path = max_selected_path[1]
            print(path, process_path)

            goal_states = np.zeros((path.shape[0],2))
            # spray_flag = np.ones((path.shape[0],1), dtype=bool)
            spray_flag = np.zeros((path.shape[0],1), dtype=bool)
            
            # Append waypoint
            for index in range(path.shape[0]):
                goal_states[index,0] = path[index,0]
                goal_states[index,1] = path[index,1]
                #find point under threshold
                for i in range(path.shape[0]):
                    r0 = int(path[i, 0])
                    c0 = int(path[i, 1])
                    if sprayeffect[r0,c0] == 0:
                        spray_flag[index,0] = False 
                    else:
                        for a in range(3):
                            for b in range(3):
                                r = int(r0 - 1 + a)
                                c = int(c0 - 1 + b)
                                if r >= 0 and r <= self.task_extent[1] and c >= 0 and c <= self.task_extent[3]:
                                    if a == 1 and b == 1:
                                        sprayeffect[r,c] = (1 - (1-0.7)**(i+1)) * sprayeffect[r,c]
                                        if sprayeffect[r,c] < threshold:
                                            sprayeffect[r,c] = 0
                                    else:
                                        sprayeffect[r,c] = (1 - (1-0.8)**(i+1)) * sprayeffect[r,c]
                                        if sprayeffect[r,c] < threshold:
                                            sprayeffect[r,c] = 0
                                                
            result[id] = (goal_states,spray_flag)
            model.add_data_x(process_path)
            num_added = num_added + process_path.shape[0]
        
        model.reduce_data_x(num_added + x_station_time.shape[0])
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
        .

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
        
        result = self.greedy_search_multi_step(step_number, alpha, sprinkeffect_all, model)
   
        return result, result_mi_all, result_mean, result_sprinkeffect_all
    
   
