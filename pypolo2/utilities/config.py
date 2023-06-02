from typing import List

import numpy as np


class Config:
    """Configuring some parameters."""
    def __init__(self, strategy = None, 
                 diffusivity_K = 5, grid_x = 20, grid_y = 20, time_co = 0.0001, delta_t = 0.01,
                 sensing_rate = 1.0, noise_scale = 1.0, num_init_samples = 1, seed = 0,
                 init_amplitude = 1.0, init_lengthscale = 0.5, init_noise = 1.0,
                 lr_hyper = 0.01, lr_nn = 0.001,
                 team_size = 5, water_volume=10,
                 eval_grid = [50, 50],
                 max_num_samples = 50,
                 Strategy = "Nonmyopic_MI_Effect", step = 8, Env = "Dynamic", alpha = 0.0, adaptive = "Adaptive",
                 spray = "Control",  effect_threshold = 25, With_water = "WithWater") -> None:
        #气体扩散相关参数
        self.diffusivity_K = diffusivity_K # diffusivity
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.env = 10 * np.ones((self.grid_x, self.grid_y)) #initial_field_test
        
        #source
        self.R = np.zeros((self.grid_x, self.grid_y))+0.5# initialize pollution resource map matrix
        self.R[16][16] = 1000 # a pollution resource in [7][2]
        self.R[5][13] = 500
        self.R[19][1] = 500

        #time parameter
        # self.time_co = 0.0001 #高斯过程回归，时间步长
        self.time_co = 1 #高斯过程回归，时间步长
        self.delta_t = 0.01 #环境演化时间步长
        
        #range
        self.env_extent = [0, self.grid_x, 0, self.grid_y]
        self.task_extent = [0, self.grid_x-1, 0, self.grid_y-1]
        
        #sensing parameter
        self.sensing_rate = sensing_rate
        self.noise_scale = noise_scale
        
        self.num_init_samples = num_init_samples
        self.seed = seed
        
        self.x_init = np.zeros((self.num_init_samples,2))
        self.x_init[0,0] = 5.0
        self.x_init[0,1] = 5.0

        #固定监测站和补水位置
        self.x_station = np.zeros((3,2))
        self.x_station[0,0] = 14.0
        self.x_station[0,1] = 6.0
        self.x_station[1,0] = 8.0
        self.x_station[1,1] = 16.0
        self.x_station[2,0] = 4.0
        self.x_station[2,1] = 6.0

        self.water_station = np.zeros((1,2))
        self.water_station[0,0] = 5.0
        self.water_station[0,1] = 5.0
        
        self.amplitude = init_amplitude
        self.lengthscale = init_lengthscale
        self.init_noise = init_noise
        self.time_stamp = 0
        
        self.lr_hyper = lr_hyper
        self.lr_nn = lr_nn
    
        #vehicle team
        self.team_size = team_size
        self.water_volume = water_volume
        
        #evaluator
        self.eval_grid = eval_grid
        
        #experiment
        self.max_num_samples = max_num_samples
        self.strategy = strategy
        
        #recording
        self.env_list = []
        self.mi_list = []
        self.pred_list = []
        self.sprinkeffect_list = []
        self.effect_list = []
        
        #experiment parameters
        self.Strategy = Strategy
        self.step = step
        self.Env = Env
        self.alpha = alpha
        self.adaptive = adaptive
        self.With_water = With_water
        self.spray = spray
        self.threshold = effect_threshold

