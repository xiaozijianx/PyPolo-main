from pathlib import Path
import numpy as np
import pandas as pd
import os
import time as tm
import Sprayer_PDE as SP
import pypolo2

def get_multi_robots(Setting):
    vehicle_team = dict()
    for i in range(Setting.team_size):
        robot = pypolo2.robots.SPRINKLER_REPLENISHANYWHERE(
            init_state = np.array([Setting.x_init[-1, 0], Setting.x_init[-1, 1]]),
            Setting = Setting
        )
        vehicle_team[i+1] = robot #因此team的id从1开始
    return vehicle_team


def get_strategy(rng, Setting, vehicle_team):
    strategy = pypolo2.strategies.SALatticePlanningMISprinklerControl_mimethod2(
            task_extent=Setting.task_extent,
            rng=rng,
            vehicle_team=vehicle_team,
        )
    return strategy


def get_evaluator():
    evaluator = pypolo2.experiments.Evaluator()
    return evaluator

def get_env_model(Setting):
    model = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                    initial_field =  Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = 0) # build model
    return model

def get_gprmodel(Setting, y_init, kernel):
    model = pypolo2.models.GPR(
        x_train=Setting.x_init,
        y_train=y_init,
        kernel=kernel,
        noise=Setting.init_noise,
        lr_hyper=Setting.lr_hyper,
        lr_nn=Setting.lr_nn,
        is_normalized = True,
        time_stamp = Setting.time_stamp,
    )
    return model

#定义需要随时间更新的训练过程
def run(rng, model, Setting, sensor, evaluator, logger, vehicle_team) -> None:
    current_step = 0 #总规划长度
    adaptive_step = Setting.adaptive_step #自适应长度
    # change_step = Setting.R_change_interval - 0*Setting.adaptive_step # 污染源变化间隔
    change_step = 0
    spray_effect = 0 # 洒水效果
    result, MI_information, observed_env, computed_effect = None, None, None, None
    while current_step < Setting.max_num_samples:
        # 计算用于显示的信息量，目标估计，洒水效果
        allpoint_list = []
        for i in range (Setting.task_extent[0],Setting.task_extent[1]):
            for j in range (Setting.task_extent[2],Setting.task_extent[3]):
                allpoint_list.append([i, j, model.time_stamp])
        allpoint = np.array(allpoint_list)
        mean, _ = model(allpoint)
        sprayeffect_all = pypolo2.objectives.sprayeffect.spray_effect(allpoint, allpoint, mean, Setting.task_extent).ravel()
        prior_diag_std, poste_diag_std, _, _ = model.prior_poste(allpoint)
        hprior = pypolo2.objectives.entropy.gaussian_entropy(prior_diag_std.ravel())
        hposterior = pypolo2.objectives.entropy.gaussian_entropy(poste_diag_std.ravel())
        mi_all = hprior - hposterior
        if np.any(mi_all < 0.0):
            print(mi_all.ravel())
            raise ValueError("Predictive MI < 0.0!")
        MI_information = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        observed_env = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        computed_effect = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        for i in range (Setting.task_extent[0],Setting.task_extent[1]):
            for j in range (Setting.task_extent[2],Setting.task_extent[3]):
                MI_information[i,j] = mi_all[i*(Setting.task_extent[3]-Setting.task_extent[2])+j]
                observed_env[i,j] = mean[i*(Setting.task_extent[3]-Setting.task_extent[2])+j,0]
                computed_effect[i,j] = sprayeffect_all[i*(Setting.task_extent[3]-Setting.task_extent[2])+j]
                
        Setting.current_step = current_step
        # scheduling and update agent goals 计算搜索时间
        if adaptive_step >= Setting.adaptive_step:
            start = tm.time()
            result = Setting.strategy.get(model = model, Setting = Setting, pred = observed_env)
            adaptive_step = 0
            for id, vehicle in vehicle_team.items():
                vehicle.set_goals(result[id][0],result[id][1])
            end = tm.time()
            print('search_time')
            print(end-start)    
            
        # calculate metrix and save 
        coverage, mean_airpollution, max_airpollution = evaluator.eval_results(Setting.env, Setting.task_extent, vehicle_team)
        logger.append(current_step, Setting.env, observed_env, MI_information, computed_effect, vehicle_team, coverage, mean_airpollution, max_airpollution, spray_effect)
           
        # change source,每经过R_change_interval后，改变源分布和强度，
        if change_step >= Setting.R_change_interval:
            change_step = 0
            if Setting.randomsource == True:
                # gengerate two set of random numbers for source locations
                numbers = rng.randint(0, 19, size=Setting.sourcenum * 2)
                pairs = rng.choice(numbers, size=(Setting.sourcenum, 2), replace=False)
                for i in range(Setting.sourcenum):
                    number = rng.randint(150, 300, size=1)
                    Setting.RR[i,0] = int(pairs[i,0])
                    Setting.RR[i,1] = int(pairs[i,1])
                    Setting.RR[i,2] = number

        print(Setting.RR)
        #  每周期更新源信息,源是缓慢变化的，源会不断变强到顶峰，然后变弱。定义一个强度系数
        s = 1
        if change_step == 0 or change_step == Setting.R_change_interval - 1:
            s = 0.5
        Setting.R =  -6 * np.ones((Setting.grid_x, Setting.grid_y)) + 13 * np.random.random((Setting.grid_x, Setting.grid_y))
        for i in range(Setting.sourcenum):
             Setting.R[Setting.RR[i,0],Setting.RR[i,1]] = s*Setting.RR[i,2]
             
        # 计算如果没有更新洒水时的环境变化
        env_model1 = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                 initial_field =  Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = Setting.current_step * Setting.delta_t) # build model
        env_withoutspray = env_model1.solve(Setting.delta_t)
        # update state 并将车辆的轨迹和洒水轨迹取出来
        x_new = []
        y_new = []
        for id, vehicle in vehicle_team.items():
            vehicle.update()
            current_state = vehicle.state.copy().reshape(1, -1)
            x_new.append(current_state)
            y_new.append(sensor.sense(current_state, rng).reshape(-1, 1))
            if Setting.current_step == 0:
                Setting.data_sprayer_train.append(pd.DataFrame())
            else:
                if vehicle.spray_flag == True:
                    new_pd = pd.DataFrame({"time":(Setting.current_step + 1) * Setting.delta_t, "x":current_state[0,0],\
                                            "y":current_state[0,1], "spray_volume":500},index=[0])
                    # Setting.data_sprayer_train[id-1] = Setting.data_sprayer_train[id-1].append(new_pd, ignore_index=True)
                    Setting.data_sprayer_train[id-1] = pd.concat([Setting.data_sprayer_train[id-1],new_pd])
        # 计算带入洒水后的环境情况
        env_model2 = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                 initial_field =  Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = Setting.current_step * Setting.delta_t) # build model
        Setting.env = env_model2.solve(Setting.delta_t)
        sensor.set_env(Setting.env)
        # 计算洒水效果
        spray_effect = np.sum(env_withoutspray - Setting.env)
            
        # using new data to update gpr model
        x_new = np.concatenate(x_new, axis=0)
        y_new = np.concatenate(y_new, axis=0)
        #add time dim
        model.time_stamp = model.time_stamp + Setting.time_co
        Setting.time_stamp = model.time_stamp
        model_input = np.zeros((x_new.shape[0],3))
        model_input[:,0:2] = x_new
        model_input[:,2:3] = model.time_stamp
        #optimize model
        model.add_data(model_input, y_new)
        model.optimize(num_iter=len(y_new), verbose=False)
        
        adaptive_step = adaptive_step + 1    
        current_step = current_step + 1
        change_step = change_step + 1  
    return 0

def Set_initual_data(rng,Setting,sensor):
    # 初始化污染源
    if Setting.randomsource == True:
        # gengerate two set of random numbers for source locations
        numbers = rng.randint(0, 19, size=Setting.sourcenum * 2)
        pairs = rng.choice(numbers, size=(Setting.sourcenum, 2), replace=False)
        for i in range(Setting.sourcenum):
            number = rng.randint(150, 300, size=1)
            Setting.RR[i,0] = int(pairs[i,0])
            Setting.RR[i,1] = int(pairs[i,1])
            Setting.RR[i,2] = number

    print(Setting.RR)
    # #  每周期更新源信息,源是缓慢变化的，源会不断变强到顶峰，然后变弱。定义一个强度系数
    s = 1
    Setting.R =  -6 * np.ones((Setting.grid_x, Setting.grid_y)) + 13 * np.random.random((Setting.grid_x, Setting.grid_y))
    for i in range(Setting.sourcenum):
            Setting.R[Setting.RR[i,0],Setting.RR[i,1]] = s*Setting.RR[i,2]
    

    env_model = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                    initial_field =  Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = 0) # build model

    y_init = np.zeros((Setting.num_init_samples,1))
    y_stations = np.zeros((Setting.station_size*Setting.time_before_sche,1))
    time_init = np.zeros((Setting.num_init_samples,1))
    time_stations = np.zeros((Setting.station_size*Setting.time_before_sche,1))

    #固定站的观测的观测
    for time in range(Setting.time_before_sche):
        # y_stations[Setting.station_size*time:Setting.station_size*(time+1)] = sensor.sense(states=Setting.x_station, rng=rng).reshape(-1, 1)
        # time_stations[Setting.station_size*time:Setting.station_size*(time+1)] = (time-Setting.time_before_sche+1)*Setting.time_co
        time_stations[Setting.station_size*time:Setting.station_size*(time+1)] = (time-Setting.time_before_sche+1)*1
        # time_stations[Setting.station_size*time:Setting.station_size*(time+1)] = (time-10+1)*Setting.time_co
        Setting.env = env_model.solve((time+1)*Setting.delta_t)
        sensor.set_env(Setting.env)
        
    #假设每次观测后均变化时间，环境也随之发生变化
    for time in range(Setting.num_init_samples):
        y_init[time] = sensor.sense(states=Setting.x_init[time], rng=rng).reshape(-1, 1)
        if time == 0:
            y_stations[:] = y_init[time] - 20
        # time_init[time] = (time+1)*Setting.time_co
        time_init[time] = (time+1)*1
        Setting.env = env_model.solve((1+Setting.time_before_sche+time)*Setting.delta_t)
        sensor.set_env(Setting.env)
        
    Setting.x_init = np.hstack((Setting.x_init,time_init))

    Setting.x_stations = Setting.x_station
    for i in range(Setting.time_before_sche-1):
        Setting.x_stations = np.vstack((Setting.x_stations,Setting.x_station))
    Setting.x_stations = np.hstack((Setting.x_stations,time_stations))

    Setting.x_init = np.vstack((Setting.x_stations,Setting.x_init))
    return np.vstack((y_stations,y_init))

def main():
    args = pypolo2.experiments.argparser.parse_arguments()
    
    Setting = pypolo2.utilities.Config(diffusivity_K = args.diffusivity_K, grid_x = args.grid_x, grid_y = args.grid_y, time_co = args.time_co, delta_t = args.delta_t,
                sensing_rate = args.sensing_rate, noise_scale = args.noise_scale, num_init_samples = args.num_init_samples, seed = args.seed,
                time_before_sche = args.time_before_sche, sourcenum = args.sourcenum, R_change_interval = args.R_change_interval,
                init_amplitude = args.amplitude, init_lengthscale = args.lengthscale, init_noise = args.init_noise,
                lr_hyper = args.lr_hyper, lr_nn = args.lr_nn,
                team_size = args.team_size, water_volume = args.water_volume, replenish_speed = args.replenish_speed,
                max_num_samples = args.max_num_samples ,
                alpha = args.alpha,
                Strategy_Name = args.strategy_name,
                sche_step = args.sche_step, adaptive_step = args.adaptive_step, Env = args.Env,
                effect_threshold = args.effect_threshold)

    # environment initual
    env_model = get_env_model(Setting)
    Setting.env = env_model.solve(Setting.delta_t)
    
    # save directory
    # starttime = Setting.starttime.replace(' ', '-').replace(':', '-')
    Setting.Savedir = '{}/{}/schestep_{}'.format(Setting.save_dir, Setting.strategy_name, Setting.sche_step)
    Setting.save_name = args.save_name
    evaluator = get_evaluator()
    logger = pypolo2.experiments.Logger(None, Setting)
    
    sensor = pypolo2.sensors.Sprinkler(Setting = Setting)
    rng = pypolo2.experiments.utilities.seed_everything(Setting = Setting)
    
    # model
    y_init = Set_initual_data(rng,Setting,sensor)
    Setting.time_stamp = Setting.x_init[:,2].max(axis=0, keepdims=False)
    kernel = pypolo2.kernels.RBF(Setting)
    model = get_gprmodel(Setting, y_init, kernel)
    model.optimize(num_iter=model.num_train, verbose=True)
    
    # robot
    vehicle_team =  get_multi_robots(Setting)
        
    # strategy
    Setting.strategy = get_strategy(rng, Setting, vehicle_team)

    # experiment search
    start = tm.time()
    run(rng, model, Setting, sensor, evaluator, logger, vehicle_team)
    end = tm.time()
    logger.save(end-start)  # I temporarily removed "makefile()".
    # pypolo2.experiments.utilities.print_metrics(logger, Setting.max_num_samples-1)
    print(f"Time used: {end - start:.1f} seconds")


if __name__ == "__main__":
    main()  # remove parameter for old one