from typing import List

import numpy as np

from ..objectives.entropy import gaussian_entropy
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot


class MyopicPlanningMI(IStrategy):
    """Myopic informative planning based on Mutual informaiton."""

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

    def get(self, model: IModel, last_state: np.ndarray, num_states: int = 1) -> np.ndarray:
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
        xs = self.rng.uniform(
            low=self.task_extent[0],
            high=self.task_extent[1],
            size=self.num_candidates,
        )
        ys = self.rng.uniform(
            low=self.task_extent[2],
            high=self.task_extent[3],
            size=self.num_candidates,
        )
        candidate_states = np.column_stack((xs, ys))
        #add demision
        x_train, _ = model.get_data()
        if x_train.shape[1] != candidate_states.shape[1]:
            model_input = np.zeros((candidate_states.shape[0],x_train.shape[1]))
            model_input[:,0:candidate_states.shape[1]] = candidate_states
            model_input[:,candidate_states.shape[1]:x_train.shape[1]] = model.time_stamp
        else:
            model_input = candidate_states
            
        # Evaluate candidates
        prior_std, poste_std = model.prior_poste(model_input)

        hprior = gaussian_entropy(prior_std.ravel())
        hposterior = gaussian_entropy(poste_std.ravel())
        mi = hprior - hposterior
        if np.any(mi < 0.0):
            print(mi.ravel())
            raise ValueError("Predictive MI < 0.0!")
 
        diffs = candidate_states - last_state
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        # Normalized scores
        normed_entropy = (mi - mi.min()) / mi.ptp()
        normed_dists = (dists - dists.min()) / dists.ptp()
        scores = normed_entropy + normed_dists
        # Append waypoint
        sorted_indices = np.argsort(scores)
        goal_states = candidate_states[sorted_indices[0:num_states]]
        # self.robot.goal_states.append(goal_states.ravel())
        # Controling and sampling
        #     while self.robot.has_goal:
        #         self.robot.update(*self.robot.control())
        # x_new = self.robot.commit_data()
        return goal_states
