from typing import List

import numpy as np

from ..objectives.entropy import gaussian_entropy
from ..models import IModel
from .strategy import IStrategy


class ActiveSampling(IStrategy):
    """Active sampling."""
    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        num_candidates: int,
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

        """
        super().__init__(task_extent, rng)
        self.num_candidates = num_candidates

    def get(self, model: IModel, num_states: int = 1) -> np.ndarray:
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
        _, std = model(model_input)
        entropy = gaussian_entropy(std.ravel())
        sorted_indices = np.argsort(entropy)
        goal_states = candidate_states[sorted_indices[-num_states:]]
        return goal_states
