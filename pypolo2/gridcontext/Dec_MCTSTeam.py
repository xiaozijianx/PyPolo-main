from typing import List

import numpy as np
import redis
import sys

from ..objectives.entropy import gaussian_entropy
from ..objectives.sprayeffect import spray_effect

import threading
import random
from scipy import stats
from collections import deque
import math
import pickle
import codecs
import itertools
import multiprocessing
import copy

c_param = 0.5  # Exploration constant, greater than 1/sqrt(8)
discount_param = 0.99
alpha = 0.01
movementsforteam = list(itertools.product([-4, 0, 4], [-4, 0, 4]))

def printsparse(matrix):
    print(matrix.toarray())

# 用与传递消息,同时在树节点中定义了节点状态
class Team_State():
    def __init__(self, location):
        self.loc = location


def move(loc, action):
    loc = loc + np.array(movementsforteam[action])
    return loc


def update_agent_state(state, action):
    loc = move(state.loc, action)
    return Team_State(loc)


# 用于在各agent之间传递消息
class Team_Info():
    def __init__(self, team_id, state, probs, time):
        self.state = state  # Agent_State
        self.probs = probs  # dict
        self.time = time  # int
        self.team_id = team_id  # arbitrary

    def select_random_plan(self):
        return np.random.choice(list(self.probs.keys()), p=list(self.probs.values())).get_location_sequence()


def uniform_sample_from_all_action_sequences(probs, other_team_info):
    other_locations = {}
    other_qs = {}
    for i, x in other_team_info.items():
        chosen_node = random.choice(list(x.probs.keys()))
        chosen_q = x.probs[chosen_node]
        chosen_locations = chosen_node.get_location_sequence()
        other_locations[i] = chosen_locations
        other_qs[i] = chosen_q

    our_node = random.choice(list(probs.keys()))
    our_q = probs[our_node]
    our_locations = our_node.get_location_sequence()
    return our_locations, other_locations, our_q, other_qs


def get_new_prob(i,node,probs,distribution_sample_iterations,other_team_info,determinization_iterations,time,beta,model,Setting):
    #print("updating probability for node " + str(i) +" of "+str(len(probs))+" on pid "+str(os.getpid()))
    # 获取node动作序列的原始发生概率
    q = probs[node]
    e_f = 0
    e_f_x = 0
    for _ in range(distribution_sample_iterations):
        # Evaluate nodes based off of what we actually know
        # 返回随机抽取的字典
        our_locations, other_locations, our_q, other_qs = \
            uniform_sample_from_all_action_sequences(probs, other_team_info)
        f = 0
        f_x = 0
        # compute_f(node_locations, other_team_locations, model, distribution_sample_iterations, start_time, Setting) / determinization_iterations
        for _ in range(determinization_iterations // 2):
            f += compute_f(our_locations, other_locations,model,distribution_sample_iterations, time,Setting) \
                 / (determinization_iterations // 2)

            f_x += compute_f(node.get_location_sequence(), other_locations, model,distribution_sample_iterations, time,Setting) \
                   / (determinization_iterations // 2)
            
        # np.prod计算列表乘积
        e_f += np.prod(list(other_qs.values()) + [our_q]) * f
        if len(other_qs) > 0:
            e_f_x += np.prod(list(other_qs.values())) * f_x
        else:
            e_f_x = 0
        return i, q - alpha * q * (
                (e_f - e_f_x) / beta
                + stats.entropy(list(probs.values())) + np.log(q))


class DecMCTS_Team():
    def __init__(self, team_id, start_loc,model,obs_distribution,obs_value,Setting,
                 horizon=10,
                 prob_update_iterations=5, # 动作序列概率优化的轮数
                 plan_growth_iterations=30, # 每次生长树时循环的轮数
                 distribution_sample_iterations=5, # 更新动作序列概率时，使用蒙特卡洛法近似计算期望的轮数
                 determinization_iterations=3, # 计算f时分解的步数
                 probs_size=12, # 可选动作序列列表的大小
                 actionlen = 18,
                 out_of_date_timeout=None, # 信息保存时长
                 comms_drop=None, # 定义了传输损失类型
                 comms_drop_rate=None, # 定义了传输损失概率
                 comms_aware_planning=False):
        self.Setting = Setting
        self.r = redis.Redis(host='localhost', port=6379)
        self.horizon = horizon
        self.executed_action_last_update = True # 上一轮执行了动作
        self.other_team_info = {}
        self.tree = None
        self.Xrn = [] # 包含所有达到要求的叶节点，每个节点都代表了一种可能的路径
        self.best_location_plan = None
        self.best_action_plan = None
        self.queue_size_limit = 100
        self.actionlen = actionlen
        self.reception_queue = deque(maxlen=self.queue_size_limit) # 消息接收队列

        self.pub_obs = 'team_obs'
        # self.pub_obs = rospy.Publisher('robot_obs', String, queue_size=10)
        self.update_iterations = 0 # update迭代轮数
        self.prob_update_iterations = prob_update_iterations
        self.plan_growth_iterations = plan_growth_iterations
        self.determinization_iterations = determinization_iterations
        self.distribution_sample_iterations = distribution_sample_iterations
        self.out_of_date_timeout = out_of_date_timeout
        self.time = 0
        self.executed_round = 0 # 当前树在此轮生长前已经生长的轮数
        self.probs_size = probs_size
        self.comms_aware_planning = comms_aware_planning
        self.times_removed_other_team = 0

        self.team_id = team_id
        self.start_loc = start_loc
        self.loc = start_loc
        self.loc_log = [start_loc]
        self.comms_drop = comms_drop
        self.comms_drop_rate = comms_drop_rate
        self.complete = False

        # 观测相关
        # obs存放观测点和相应的观测值
        self.obs_location = start_loc.reshape(1,2)
        self.obs_value = np.array([obs_value])
        self.model = model
        self.obs_distribution = obs_distribution
        self.setup_listener()

        print("Creating team " + str(self.team_id) + " at position " + str(start_loc) +
              ", comms aware planning is " + ("enabled" if comms_aware_planning else "disabled"))

    def sense(self,loc):
        return 

    def get_time(self):
        return self.time

    # 生长树
    def growSearchTree(self,current_prob_update_iterations):
        for i in range(self.plan_growth_iterations):
            # Perform Dec_MCTS step
            # 在拓展节点时只有第一层是有严格观测可依的，再往下的子节点可能无严格的观测
            round = self.executed_round + current_prob_update_iterations*self.plan_growth_iterations + i
            node = self.tree.select_node(round).expand()
            other_team_locations = self.sample_other_agents()
            # rollout并回溯，主要修改的部分
            score = node.perform_rollout(other_team_locations, self.actionlen, self.model, self.get_time(), self.Setting,
                                         self.determinization_iterations, self.distribution_sample_iterations())
            
            node.backpropagate(score, round)

    # 重置树
    def reset_tree(self):
        self.Xrn = []
        self.tree = DecMCTSTeamNode(Team_State(self.loc), row=self.obs_distribution.shape(0),
                                colume=self.obs_distribution.shape(1), depth=0, Xrn=self.Xrn,
                                comms_aware_planning=self.comms_aware_planning)

    # 获取具有较高折扣分数的动作序列字典
    def get_Xrn_probs(self):
        # 将Xrn内的节点根据折扣分数进行排序
        self.Xrn.sort(reverse=True, key=(lambda node: node.discounted_score))
        probs = {}
        for node in self.Xrn[1:min(len(self.Xrn), self.probs_size)]:
            probs[node] = 1 / self.probs_size
        return probs

    # 消息打包，消息为agent_info类，注意观测在这里也会被传递出去
    def package_comms(self, probs):
        return Team_Info(self.team_id, Team_State(self.loc), probs, self.get_time())

    # 接收消息
    def unpack_comms(self):
        copied_queue = copy.copy(self.reception_queue)
        for message_str in copied_queue:
            message = pickle.loads(codecs.decode(message_str, 'base64'))
            print("receive:",message)
            if message.team_id == self.team_id:
                pass
            # 计算发送消息的agent与本节点的距离
            distance = max(
                math.sqrt((message.state.loc[0] - self.loc[0]) ** 2 + (message.state.loc[1] - self.loc[1]) ** 2), 1)
            # 判断是否传输损失
            if self.comms_drop == "uniform" and random.random() < self.comms_drop_rate:
                pass
                #print("Packet drop")
            elif self.comms_drop == "distance" and random.random() < self.comms_drop_rate / (distance) ** 2:
                pass
                #print("Packet drop")
            else:
                team_id = message.team_id
                # If seen before
                if team_id in self.other_team_info.keys():
                    # If fresh message
                    if message.time >= self.other_team_info[team_id].time:
                        self.other_team_info[team_id] = message
                else:
                    self.other_team_info[team_id] = message
                # self.observations_list = merge_observations(self.observations_list, message.state.obs)

        time = self.get_time()
        # Filter out-of-date messages
        if self.out_of_date_timeout is not None:
            new_other_team = {}
            for k, v in self.other_team_info:
                if v.time + self.out_of_date_timeout >= time:
                    new_other_team[k] = v
                else:
                    self.times_removed_other_team += 1
            self.other_team_info = new_other_team

        self.reception_queue.clear()

    # Execute_movement should be true if this is a move step, rather than just part of the computation
    def update(self,execute_action):
        '''
        Move to next position, update observations, update locations, run MCTS,
        publish to ROS topic
        '''
        print("-------- Update ", self.update_iterations, " team id ", self.team_id, "Thread id ", threading.current_thread().name, "---------")
        self.update_iterations += 1
        # 上轮如果执行了动作，则一定重置树，论文中初始化树在主循环外。
        if self.executed_action_last_update:
            self.reset_tree()
            self.beta = 0.1
            self.executed_action_last_update = False
            self.executed_round = 0

        # 获取具有较高折扣分数的动作序列字典,字典的键为节点，值为概率
        probs = self.get_Xrn_probs()

        # 循环迭代优化各动作序列的概率
        for i in range(self.prob_update_iterations):
            self.growSearchTree(i)
            if len(probs) > 0:
                # 更新各动作序列的概率用于发送给其他agent
                self.update_distribution(probs)
                # 将消息打包
                message = self.package_comms(probs)
                # 消息发布，发布观测和动作概率
                print("publish:",message)
                self.r.publish(self.pub_obs, codecs.encode(pickle.dumps(message), "base64").decode())

            self.unpack_comms()
            self.cool_beta()
        self.executed_round = self.executed_round + self.prob_update_iterations*self.plan_growth_iterations

        # 如果为执行动作周期，则更新best_plan
        if execute_action:
            self.executed_action_last_update = True
            if len(probs) > 0:
                # 获取具有最大概率的概率值对应的键，并返回相应的动作序列
                self.best_location_plan = max(probs, key=probs.get).get_location_sequence()
                self.best_action_plan = max(probs, key=probs.get).get_action_sequence()
            else:
                self.best_action_plan = None
                self.best_location_plan = None

            if self.time == self.Setting.teamsche_step:
                self.complete = True
                self.close_listener()

    def cool_beta(self):
        self.beta = self.beta * 0.9

    def sample_other_agents(self):
        return {i: team.select_random_plan() for (i, team) in self.other_team_info.items()}

    def update_distribution(self, probs):
        enumerated_keys = list(enumerate(list(probs.keys())))
        print(len(enumerated_keys))
        # 构造了一个列表，除了叶节点和序号不同外，包含了全部的树信息
        args = [(i, node, probs, self.distribution_sample_iterations, self.other_team_info, self.determinization_iterations,
        self.get_time(), self.beta, self.model, self.Setting) for i, node in enumerated_keys]

        # 并行执行，提高效率，也可以限制使用的核心数量
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            #p.starmap与map函数的区别为p.starmap会将元组解包为一系列参数传入而map函数将元组视为一整个参数传入，
            #所以如果使用map函数会报错缺一些列需要的参数。
            newprobs = p.starmap(get_new_prob, args) 
            # newprobs = map(get_new_prob, args)

            key_dict = dict(enumerated_keys)
            # normalize
            factor = 1.0 / sum([prob for i, prob in newprobs])
            for i, prob in newprobs:
                probs[key_dict[i]] = prob*factor

    def receive_info(self, message):
        '''
        Callback function
        '''
        channel = message['channel']
        infomsg = message['data']
        # if len(self.reception_queue) == self.reception_queue.maxlen:
        # self.reception_queue.popleft()  # 移除最旧的消息
        self.reception_queue.append(infomsg)  # 添加新消息
        # self.reception_queue.append(message)  # 添加新消息



    def setup_listener(self):
        '''
        Implement listener code, call to update_loc()
        Implement timer listener at frequency 1/render_interval to call to render()
        '''
        p = self.r.pubsub()
        p.subscribe(**{self.pub_obs: self.receive_info})
        # 启动订阅线程
        self.thread = p.run_in_thread(sleep_time=0.01)

    def close_listener(self):
        self.thread.stop()
        p = self.r.pubsub()
        p.unsubscribe()
        p.close()

class DecMCTSTeamNode():
    def __init__(self, state, depth, row, colume, Xrn, parent=None, parent_action=None, comms_aware_planning=False):
        self.depth = depth # 节点所在层次
        self.state = state # agent_state包括机器人的当前位置和观测
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.Xrn = Xrn # 存放所有可能得执行路径，初始化根节点时定义，随后不断维护
        self.row = row
        self.colume = colume

        # Incremented during backpropagation
        self.discounted_visits = 0 #对应算法中的折扣访问奖励
        self.discounted_score = 0 #对应算法中的折扣分数
        self.last_round_visited = 0 #上次访问时对饮的轮数
        self.unexplored_actions = self.get_legal_actions()
        self.action_sequence = None
        self.location_sequence = None
        self.comms_aware_planning = comms_aware_planning

    # 获取节点所属的动作列表，
    def get_action_sequence(self):
        if self.action_sequence is not None:
            return self.action_sequence
        else:
            _, self.action_sequence, _ = self.tree_history()
            return self.action_sequence
        
    # 获取节点所属的位置列表，
    def get_location_sequence(self):
        if self.location_sequence is not None:
            return self.location_sequence
        else:
            _, _, self.location_sequence = self.tree_history()
            return self.location_sequence

    # 通过d_uct选择节点
    def select_node(self, round_n):
        return self.select_node_d_uct(round_n)

    def select_node_d_uct(self, round_n):
        if self.not_fully_explored():
            return self
        else:
            t_js = [child.discounted_visits * math.pow(discount_param, round_n - child.last_round_visited)
                    for child in self.children]
            t_d = sum(t_js)

            child_scores = []
            for i, child in enumerate(self.children):
                if child.discounted_visits == 0:
                    print("WARNING, CHILDREN SHOULD NOT BE UNVISITED (decmcts.py  select_node_d_uct)")
                    print(child.discounted_visits)
                    print(self.depth)
                    print(self.discounted_visits)
                    f = child.discounted_score
                    c = 2 * math.sqrt(max(np.log(t_d), 0))
                else:
                    f = child.discounted_score / child.discounted_visits
                    c = 2 * math.sqrt(max(np.log(t_d), 0) / t_js[i])
                child_scores.append(f + c_param * c)

            return self.children[np.argmax(np.asarray(child_scores))].select_node_d_uct(round_n)

    # 所有合法动作
    def get_legal_actions(self):
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        row, colume = self.state.loc
        actions = []
        for i in range(len(movementsforteam)):
            new_row = row + movementsforteam[i][0]
            new_colume = colume + movementsforteam[i][1]
        if new_row >= 0 and new_row <= self.row and new_colume >= 0 and new_colume <= self.colume:
            actions.append(i)
        return actions

    def get_stochastic_action(self):
        '''
        Pick a random legal action.
        '''
        return random.choice(self.unexplored_actions)

    def expand(self):
        action = self.get_stochastic_action()
        self.unexplored_actions.remove(action)
        next_state = update_agent_state(self.state, action)
        child_node = DecMCTSTeamNode(next_state, self.depth + 1, self.row, self.colume, self.Xrn, parent=self,
                                 parent_action=action, comms_aware_planning=self.comms_aware_planning)

        self.children.append(child_node)
        return child_node

    def backpropagate(self, score, iteration):
        self.discounted_visits = 1 + \
                                 math.pow(discount_param, iteration - self.last_round_visited) * self.discounted_visits
        self.discounted_score = score + \
                                math.pow(discount_param, iteration - self.last_round_visited) * self.discounted_score
        self.last_round_visited = iteration
        if self.parent is not None:
            self.parent.backpropagate(score, iteration)

    def not_fully_explored(self):
        return len(self.unexplored_actions) != 0

    def tree_history(self):
        curr_node = self
        node_actions = []
        node_locations = []
        start_state = None
        while curr_node.parent is not None:
            node_actions.append(curr_node.parent_action)
            node_locations.append(curr_node.state)
            curr_node = curr_node.parent
            if curr_node.parent is None:
                start_state = curr_node.state
        node_actions.reverse()
        node_locations.reverse()
        return start_state, node_actions, node_locations

    # 演化，长度不足时使用随机测略
    def perform_rollout(self, other_team_locations, actionlen, model,start_time, Setting,
                        determinization_iterations,distribution_sample_iterations):
        if self.depth == actionlen:
            self.Xrn.append(self)
        
        # horizon_time = time + self.depth + horizon
        # 深度不足时，采用随机/固定策略补全
        start_state, node_actions, node_locations = self.tree_history()
        i = 0
        while self.depth + i < actionlen:
            i = i + 1
            node_actions.append(4)
            node_locations.append(node_locations[-1])
        
        avg = 0
        for _ in range(determinization_iterations):
            avg += compute_f(node_locations, other_team_locations, model, distribution_sample_iterations, start_time, Setting) / determinization_iterations

        return avg


def compute_f(our_locations, other_team_locations, model, distribution_sample_iterations, start_time, Setting):
    # Score if we took no actions
    null_score = get_score(None, other_team_locations, model, distribution_sample_iterations, start_time, Setting)

    # Score if we take our actual actions (simulates future plans)
    actuated_score = get_score(our_locations, other_team_locations, model, distribution_sample_iterations, start_time, Setting)
    return actuated_score - null_score

def get_score(our_locations, other_team_locations, model, distribution_sample_iterations, start_time, Setting):
    # 首先对轨迹进行处理
    time_list = np.ones((Setting.teamsche_step,1))
    for i in range(Setting.teamsche_step):
        time_list[i,0] = (start_time + i + 1)*Setting.teamtime_co

    if our_locations != None:
        our_loc = np.hstack(np.array(our_locations),time_list)
    else:
        our_loc = None

    # other_agent_locations:dict{i:locations}
    otherteam_loc = None
    team_loc = None
    for (i, team_loc) in other_team_locations.items():
        team_loc = np.hstack(np.array(team_loc),time_list)
        if otherteam_loc== None:
            otherteam_loc = team_loc
        else:
            otherteam_loc = np.vstack(otherteam_loc,team_loc)
    if our_loc != None:
        team_loc = np.vstack(our_loc,otherteam_loc)
    else:
        if otherteam_loc!=None:
            team_loc = otherteam_loc

    # 分数由两部分计算而来
    # 1 信息收益 ####################################
    # 计算目标点
    allpoint_list = []
    for num in range(0,Setting.teamsche_step):
        for i in np.arange (Setting.task_extent[0]+4//2,Setting.task_extent[1],4):
            for j in np.arange (Setting.task_extent[2]+4//2,Setting.task_extent[3],4):
                allpoint_list.append([i, j, (start_time + num + 1) * Setting.teamtime_co])
    allpoint = np.array(allpoint_list)

    # 计算信息收益
    processed_points = np.unique(team_loc, axis=0)
    # print(processed_points)
    train_data = model.get_data_x()
    nrows, ncols = train_data.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [train_data.dtype]}
    mid_points = np.intersect1d(train_data.view(dtype), processed_points.view(dtype))
    processed_points2 = np.setdiff1d(processed_points.view(dtype), mid_points)
    processed_points2 = processed_points2.view(train_data.dtype).reshape(-1, ncols)
    model.add_data_x(processed_points2)
    _, _, prior_cov, poste_cov = model.prior_poste(allpoint)
    if processed_points2.shape[0] > 0:
        model.reduce_data_x(processed_points2.shape[0])
    # prior_entropy = gaussian_entropy_multivariate(prior_cov)
    # poste_entropy = gaussian_entropy_multivariate(poste_cov)
    # mi = prior_entropy - poste_entropy
    mi = (prior_cov.trace()- poste_cov.trace())/prior_cov.shape[0]

    return mi