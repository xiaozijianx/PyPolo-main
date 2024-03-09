from typing import List

import numpy as np
import redis
import sys

from ..models import IModel
from .strategy import IStrategy
from ..gridcontext.Dec_MCTSTeam import DecMCTS_Team

import threading
import random

class DecMCTSSprinkle(IStrategy):
    """
    一种分层的调度策略，上层调度针对团队，下层调度针对个体
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
        self.teams = []
        
        
    def get(self, model: IModel, Setting, obs_distribution) -> np.ndarray:
        r = redis.Redis(host='localhost', port=6379)
        comms_aware = True
        out_of_date_timeout = None
        # 实例化团队
        if len(self.teams) == 0:
            for id, vehicle in self.vehicle_team.items():
                self.teams.append(DecMCTS_Team(team_id=id, start_loc=vehicle.state,model=model,obs_distribution=obs_distribution,
                                                    comms_drop="distance", comms_drop_rate=0.9,
                                                    comms_aware_planning=comms_aware,
                                                    out_of_date_timeout=out_of_date_timeout))
        # 实例化个体

        i = -1  
        complete = False
        while not complete:
            is_execute_iteration = ((i % 2) == 0)
            i += 1
            threads = []
            for r in self.teams:
                thread = threading.Thread(target=r.update, args=(is_execute_iteration,))
                threads.append(thread)
            random.shuffle(threads)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            complete = True
            for r in self.teams:
                complete = complete and r.complete
        
        # getresult
        result = dict()
        # for id, vehicle in self.vehicle_team.items():

        print(result)               
        return result