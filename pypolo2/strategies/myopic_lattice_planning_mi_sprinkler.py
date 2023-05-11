from typing import List

import numpy as np

from ..objectives.entropy import gaussian_entropy
from ..objectives.sprinkeffect import sprink_effect
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot


class MyopicLatticePlanningMISprinkler(IStrategy):
    """Myopic informative planning based on Mutual informaiton on latttice map."""

    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        num_candidates: int,
        robot: IRobot,
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
        robot: IRobot
            A robot model.

        """
        super().__init__(task_extent, rng)
        self.num_candidates = num_candidates
        self.robot = robot

    def get(self, model: IModel, Strategy = "MI_Effect_myopic", alpha = 1, num_states: int = 1) -> np.ndarray:
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
        candidate_states_list = []
        for i in range(5):
            for j in range(5):
                c1 = self.robot.state[0] -2 + i
                c2 = self.robot.state[1] -2 + j
                if c1 < self.task_extent[0] or c1 > self.task_extent[1] or c2 < self.task_extent[2] or c2 > self.task_extent[3]:
                    continue
                # if i == 1 and j == 1:
                #     continue
                candidate_states_list.append([c1, c2])
        candidate_states = np.array(candidate_states_list)
        #add demision
        x, _ = model.get_data()
        if x.shape[1] != candidate_states.shape[1]:
            model_input = np.zeros((candidate_states.shape[0],x.shape[1]))
            model_input[:,0:candidate_states.shape[1]] = candidate_states
            model_input[:,candidate_states.shape[1]:x.shape[1]] = model.time_stamp
        else:
            model_input = candidate_states
        
        # first information object
        # Evaluate candidates
        prior_diag_std, poste_diag_std, poste_cov, poste_cov = model.prior_poste(model_input)
        hprior = gaussian_entropy(prior_diag_std.ravel())
        hposterior = gaussian_entropy(poste_diag_std.ravel())
        mi = hprior - hposterior
        if np.any(mi < 0.0):
            print(mi.ravel())
            raise ValueError("Predictive MI < 0.0!")
        
        #second sprinkler effect
        allstate_list = []
        for i in range (self.task_extent[0],self.task_extent[1]+1):
            for j in range (self.task_extent[2],self.task_extent[3]+1):
                allstate_list.append([i, j, model.time_stamp])
        allstate = np.array(allstate_list)
        mean, _ = model(allstate)
        sprinkeffect = sprink_effect(model_input,allstate,mean,self.task_extent).ravel()
        
        #third compute mi of all points
        prior_diag_std, poste_diag_std, poste_cov, poste_cov = model.prior_poste(allstate)
        hprior_all = gaussian_entropy(prior_diag_std.ravel())
        hposterior_all = gaussian_entropy(poste_diag_std.ravel())
        mi_all = hprior_all - hposterior_all
        if np.any(mi_all < 0.0):
            print(mi_all.ravel())
            raise ValueError("Predictive MI < 0.0!")
        sprinkeffect_all = sprink_effect(allstate,allstate,mean,self.task_extent).ravel()
        
        # Normalized scores
        normed_mi = (mi - mi.min()) / mi.ptp()
        normed_effect = (sprinkeffect - sprinkeffect.min()) / sprinkeffect.ptp()
        if Strategy == "MI_myopic":
            scores = normed_mi
        elif Strategy == "Effect_myopic":
            scores = - normed_effect
        else:
            scores = normed_mi - alpha*normed_effect
        # scores = - normed_effect
        # Append waypoint
        sorted_indices = np.argsort(scores)
        goal_states = candidate_states[sorted_indices[0:num_states]]
        spray_flag = True
        
        
        return goal_states,spray_flag,mi_all,mean,sprinkeffect_all
