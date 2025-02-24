from typing import List

import numpy as np


class Config:
    """Configuring some parameters."""
    def __init__(self, root_dir = "../outputs", save_name = "text", strategy = None, 
                 diffusivity_K =1.2, grid_x = 20, grid_y = 20, time_co = 0.0001, delta_t = 0.01,
                 sensing_rate = 1.0, noise_scale = 1.0, num_init_samples = 1, seed = 11,
                 time_before_sche = 5, station_size = 1, sourcenum = 4, R_change_interval = 15,
                 init_amplitude = 1.0, init_lengthscale = 0.5, init_noise = 1.0,
                 lr_hyper = 0.01, lr_nn = 0.001,
                 team_size = 4, water_volume=4, replenish_speed = 1,
                 max_num_samples = 18, current_step = 0 , bound0=50, bound1 = 100, bound2 = 15, bound3 = 100,
                #  alpha = [0.75,0.9,0.99,1.05,1.5],
                 alpha = 0.2,
                 Strategy_Name = "SA_Dualobject",
                 sche_step = 10, adaptive_step = 3, Env = "Dynamic",
                 effect_threshold = 0.0) -> None:
        
        # 实验数据选择,污染源数目选择,森林灭火拓展试验专用
        self.starttime = '2018-11-23 08:00:00'
        
        # 文件存放目录及名称
        self.root_dir = root_dir
        self.save_dir = root_dir
        self.save_name = save_name
        
        # 气体扩散相关参数
        self.diffusivity_K = diffusivity_K # diffusivity，以前的仿真环境中使用的扩散系数，现在info.json中配置
        self.grid_x = grid_x #一格代表250m
        self.grid_y = grid_y
        # self.env = 80 * np.ones((grid_x, grid_y))\
        #             + 0 * np.random.random((grid_x, grid_y))# randomly initialize "initial_field" map matrix around 250
        self.env = 25 * np.ones((grid_x, grid_y))\
                    + 0 * np.random.random((grid_x, grid_y))#初始污染物分布，每个网格为100m x 100m，污染源单位PM2.5
        
        #source
        self.randomsource = True
        self.sourcenum = sourcenum
        # self.sourcenum = team_size
        self.R =  -3 * np.ones((grid_x, grid_y)) + 6 * np.random.random((grid_x, grid_y)) # initialize pollution resource map matrix
        self.R_change_interval = R_change_interval
        self.data_sprayer_train = [] 
        # self.RR = np.zeros((self.sourcenum, 3)).astype(int)
        # R记录污染分布，RR仅记录污染源分布及位置
        self.RR = np.zeros((6, 3)).astype(int)
        self.RR[0,0] = 17
        self.RR[0,1] = 3
        self.RR[0,2] = 60
        self.RR[1,0] = 17
        self.RR[1,1] = 17
        self.RR[1,2] = 60
        self.RR[2,0] = 3
        self.RR[2,1] = 17
        self.RR[2,2] = 60
        self.RR[3,0] = 3
        self.RR[3,1] = 3
        self.RR[3,2] = 60
        self.RR[4,0] = 3
        self.RR[4,1] = 10
        self.RR[4,2] = 60
        self.RR[5,0] = 17
        self.RR[5,1] = 10
        self.RR[5,2] = 60
        self.RR = self.RR.astype(int)
        for a in range(6):
            self.R[self.RR[a,0],self.RR[a,1]] = self.RR[a,2]
        
        self.Traffic_jam_number = 40
        self.Traffic_jam = np.zeros((self.Traffic_jam_number, 3)).astype(int)
        self.Traffic = 5 * np.random.random((grid_x, grid_y))
        self.jam_time = np.zeros(team_size).astype(int)

        self.sources = []# 疑似污染源

        #time parameter
        self.time_co = 0.1 #高斯过程回归，时间步长
        self.delta_t = 10 # 仿真环境推进时间步长，min
        
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
        # 模拟退火的搜索轮数
        self.bound0 = bound0 # 信息搜索轮数
        self.bound1 = bound1 # 稀疏搜索轮数
        self.bound2 = bound2 # 后续优化轮数
        self.bound3 = bound3 # 次轮搜索轮数

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
        
        # 实验接受率
        self.accept_rate = []

        # 目标位置分层密度
        self.layer_xyinterval = [4,5,6]
        self.layer_tinterval = [2,3,5]

    
        

