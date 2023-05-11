from typing import List

import numpy as np

from ..objectives.entropy import gaussian_entropy
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot


class MyopicLatticePlanningMI(IStrategy):
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

    def get(self, model: IModel,last_state: np.ndarray, extent:List[float], num_states: int = 1) -> np.ndarray:
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
        for i in range(3):
            for j in range(3):
                c1 = last_state[0] -1 + i
                c2 = last_state[1] -1 + j
                if c1 < extent[0] or c1 > extent[1] or c2 < extent[2] or c2 > extent[3]:
                    continue
                # if i == 1 and j == 1:
                #     continue
                candidate_states_list.append([c1, c2])
        candidate_states = np.array(candidate_states_list)
        #add demision
        x_train, _ = model.get_data()
        if x_train.shape[1] != candidate_states.shape[1]:
            model_input = np.zeros((candidate_states.shape[0],x_train.shape[1]))
            model_input[:,0:candidate_states.shape[1]] = candidate_states
            model_input[:,candidate_states.shape[1]:x_train.shape[1]] = model.time_stamp
        else:
            model_input = candidate_states
            
        # Evaluate candidates
        print(model.time_stamp)
        prior_std, poste_std = model.prior_poste(model_input)

        hprior = gaussian_entropy(prior_std.ravel())
        hposterior = gaussian_entropy(poste_std.ravel())
        mi = hprior - hposterior
        if np.any(mi < 0.0):
            print(mi.ravel())
            raise ValueError("Predictive MI < 0.0!")
        # Normalized scores
        normed_entropy = (mi - mi.min()) / mi.ptp()
        scores = normed_entropy
        # Append waypoint
        sorted_indices = np.argsort(scores)
        goal_states = candidate_states[sorted_indices[0:num_states]]
        # self.robot.goal_states.append(goal_states.ravel())
        # Controling and sampling
        #     while self.robot.has_goal:
        #         self.robot.update(*self.robot.control())
        # x_new = self.robot.commit_data()
        return goal_states,mi[sorted_indices],candidate_states[sorted_indices],hprior,hposterior
