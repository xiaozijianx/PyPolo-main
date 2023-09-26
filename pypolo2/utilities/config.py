from typing import List

import numpy as np


class Config:
    """Configuring some parameters."""
    def __init__(self, save_dir = "./outputs", save_name = "text", strategy = None, 
                 diffusivity_K =1.2, grid_x = 20, grid_y = 20, time_co = 0.0001, delta_t = 0.01,
                 sensing_rate = 1.0, noise_scale = 1.0, num_init_samples = 1, seed = 0,
                 time_before_sche = 5, station_size = 1,
                 init_amplitude = 1.0, init_lengthscale = 0.5, init_noise = 1.0,
                 lr_hyper = 0.01, lr_nn = 0.001,
                 team_size = 5, water_volume=6, replenish_speed = 2,
                 max_num_samples = 40, current_step = 0 ,bound = 30, alpha = 1.5,
                 Strategy_Name = "SA_OnlyonetimeMI_simpleeffect",
                 sche_step = 20, adaptive_step = 2, Env = "Dynamic",
                 effect_threshold = 0.0) -> None:
        
        # 实验数据选择,污染源数目选择,森林灭火拓展试验专用
        self.starttime = '2018-11-23 08:00:00'
        
        # 文件存放目录及名称
        self.save_dir = save_dir
        self.save_name = save_name
        
        # 气体扩散相关参数
        self.diffusivity_K = diffusivity_K # diffusivity
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.env = 10 * np.ones((self.grid_x, self.grid_y)) #initial_field_test
        
        #source
        self.randomsource = False
        self.R = np.zeros((self.grid_x, self.grid_y)) + 4# initialize pollution resource map matrix
        self.R[2][2] = 1000
        self.R[17][17] = 1000
        self.R[17][2] = 1000
        self.R[2][17] = 1000
        self.R_change_interval = 15

        #time parameter
        self.time_co = 0.1 #高斯过程回归，时间步长
        self.delta_t = 0.3 #环境演化时间步长
        
        #range
        self.env_extent = [0, self.grid_x, 0, self.grid_y]
        self.task_extent = [0, self.grid_x, 0, self.grid_y]
        
        #sensing parameter
        self.sensing_rate = sensing_rate
        self.noise_scale = noise_scale
        
        # experiment parameter
        self.num_init_samples = num_init_samples
        self.seed = seed
        self.max_num_samples = max_num_samples
        self.current_step = current_step
        self.bound = bound
        self.alpha = alpha
        self.strategy = strategy #class
        self.strategy_name = Strategy_Name
        self.sche_step = sche_step
        self.adaptive_step = adaptive_step
        if self.adaptive_step > self.sche_step:
            raise ValueError("adaptive_step must smaller than sche_step")
        self.Env = Env
        self.effect_threshold = effect_threshold
        
        # 初始车辆位置
        self.x_init = np.zeros((self.num_init_samples,2))
        self.x_init[0,0] = 10.0
        self.x_init[0,1] = 10.0

        # 固定监测站和补水位置
        self.station_size = station_size
        self.x_station = np.zeros((self.station_size,2))
        self.x_station[0,0] = 16.0
        self.x_station[0,1] = 16.0
        self.water_station = np.zeros((1,2))
        self.water_station[0,0] = 5.0
        self.water_station[0,1] = 5.0
        
        #调度前数据时长
        self.time_before_sche = time_before_sche
        
        # 核参数
        self.amplitude = init_amplitude
        self.lengthscale = init_lengthscale
        self.init_noise = init_noise
        self.time_stamp = 0
        
        self.lr_hyper = lr_hyper
        self.lr_nn = lr_nn
    
        # vehicle team
        self.team_size = team_size
        self.replenish_speed = replenish_speed
        self.water_volume = water_volume
        

    
        

