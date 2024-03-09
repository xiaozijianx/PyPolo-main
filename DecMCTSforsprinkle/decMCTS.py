import math
import os
import random
import threading
import time

import maze as m
import numpy as np

from scipy import stats
from maze import Action
from scipy import sparse

import pickle
import codecs
import redis
import ast
import copy
from collections import deque

import multiprocessing

c_param = 0.5  # Exploration constant, greater than 1/sqrt(8)
discount_param = 0.99
alpha = 0.01


def printsparse(matrix):
    print(matrix.toarray())

# 用与传递消息,同时在树节点中定义了节点状态
class Agent_State():
    def __init__(self, location, observations):
        self.loc = location
        self.obs = observations


def move(loc, action):
    x, y = loc
    if action == Action.UP:
        loc = (x, y - 1)
    elif action == Action.DOWN:
        loc = (x, y + 1)
    elif action == Action.LEFT:
        loc = (x - 1, y)
    elif action == Action.RIGHT:
        loc = (x + 1, y)
    else:
        loc = (x, y)
    return loc


def update_agent_state(state, action):
    loc = move(state.loc, action)
    x, y = loc
    obs = state.obs.copy()
    obs[y, x] = 1
    return Agent_State(loc, obs)


# 用于在各agent之间传递消息
class Agent_Info():
    def __init__(self, robot_id, state, probs, time):
        self.state = state  # Agent_State
        self.probs = probs  # dict
        self.time = time  # int
        self.robot_id = robot_id  # arbitrary

    def select_random_plan(self):
        return np.random.choice(list(self.probs.keys()), p=list(self.probs.values())).get_action_sequence()


# ha ha ha, I'm so sorry if you have to read this part
def uniform_sample_from_all_action_sequences(probs, other_agent_info):
    other_actions = {}
    other_qs = {}
    for i, x in other_agent_info.items():
        chosen_node = random.choice(list(x.probs.keys()))
        chosen_q = x.probs[chosen_node]
        chosen_actions = chosen_node.get_action_sequence()
        other_actions[i] = chosen_actions
        other_qs[i] = chosen_q

    our_node = random.choice(list(probs.keys()))
    our_q = probs[our_node]
    our_obs = our_node.state.obs
    our_actions = our_node.get_action_sequence()
    return our_actions, other_actions, our_q, other_qs, our_obs


def get_new_prob(i,node,probs,distribution_sample_iterations,other_agent_info,determinization_iterations,robot_id,observations_list,loc,horizon,time,goal,comms_aware_planning,beta):
    #print("updating probability for node " + str(i) +" of "+str(len(probs))+" on pid "+str(os.getpid()))
    # 获取node动作序列的原始发生概率
    q = probs[node]
    e_f = 0
    e_f_x = 0
    for _ in range(distribution_sample_iterations):
        # Evaluate nodes based off of what we actually know
        our_actions, other_actions, our_q, other_qs, our_obs = \
            uniform_sample_from_all_action_sequences(probs, other_agent_info)
        f = 0
        f_x = 0
        for _ in range(determinization_iterations // 2):
            f += compute_f(robot_id, our_actions, other_actions, observations_list, loc, our_obs,
                           other_agent_info, horizon, time, goal,
                           comms_aware_planning=comms_aware_planning) \
                 / (determinization_iterations // 2)

            f_x += compute_f(robot_id, node.get_action_sequence(), other_actions, observations_list,
                             loc, our_obs,
                             other_agent_info, horizon, time,goal,
                             comms_aware_planning=comms_aware_planning) \
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


class DecMCTS_Agent():
    def __init__(self, robot_id, start_loc, goal_loc, env,
                 horizon=10,
                 prob_update_iterations=5, # 动作序列概率优化的轮数
                 plan_growth_iterations=30, # 每次生长树时循环的轮数
                 distribution_sample_iterations=5, # 更新动作序列概率时，使用蒙特卡洛法近似计算期望的轮数
                 determinization_iterations=3, # 计算f时分解的步数
                 probs_size=12, # 可选动作序列列表的大小
                 out_of_date_timeout=None, # 信息保存时长
                 comms_drop=None, # 定义了传输损失类型
                 comms_drop_rate=None, # 定义了传输损失概率
                 comms_aware_planning=False):
        self.r = redis.Redis(host='localhost', port=6379)
        self.horizon = horizon
        self.other_agent_info = {}
        self.executed_action_last_update = True # 上一轮执行了动作
        self.tree = None
        self.Xrn = [] # 包含所有达到要求的叶节点，每个节点都代表了一种可能的路径
        self.queue_size_limit = 100
        self.reception_queue = deque(maxlen=self.queue_size_limit) # 消息接收队列

        self.pub_obs = 'robot_obs'
        # self.pub_obs = rospy.Publisher('robot_obs', String, queue_size=10)
        self.update_iterations = 0 # update迭代轮数
        self.prob_update_iterations = prob_update_iterations
        self.plan_growth_iterations = plan_growth_iterations
        self.determinization_iterations = determinization_iterations
        self.distribution_sample_iterations = distribution_sample_iterations
        self.out_of_date_timeout = out_of_date_timeout
        self.time = 0
        self.executed_round = 0
        self.probs_size = probs_size
        self.comms_aware_planning = comms_aware_planning
        self.times_removed_other_agent = 0

        self.robot_id = robot_id
        self.start_loc = start_loc
        self.goal_loc = goal_loc
        self.loc = start_loc
        self.loc_log = [start_loc]
        self.env = env
        self.pub_loc = 'robot_loc_' + str(robot_id)
        # self.pub_loc = rospy.Publisher('robot_loc_' + str(robot_id), Point, queue_size=10)
        self.comms_drop = comms_drop
        self.comms_drop_rate = comms_drop_rate
        self.complete = False

        self.observations_list = sparse.dok_matrix((self.env.height, self.env.width)) # 机器人的观测列表，0为未观测，1为通路，2为墙
        self.add_edges_to_observations()
        self.update_observations_from_location()
        self.setup_listener()
        locmsg = (self.loc[0], self.loc[1], self.robot_id)
        locmsg_str = str(locmsg)
        self.r.publish(self.pub_loc, locmsg_str)

        print("Creating robot " + str(self.robot_id) + " at position " + str(start_loc) +
              ", comms aware planning is " + ("enabled" if comms_aware_planning else "disabled"))

    # 在初始化时，默认最外圈都是墙
    def add_edges_to_observations(self):
        for i in range(self.env.width):
            self.observations_list[0, i] = 2
            self.observations_list[self.env.height - 1, i] = 2

        for i in range(self.env.height):
            self.observations_list[i, 0] = 2
            self.observations_list[i, self.env.width - 1] = 2

    def get_time(self):
        return self.time

    # 生长树
    def growSearchTree(self,current_prob_update_iterations):
        for i in range(self.plan_growth_iterations):
            # Perform Dec_MCTS step
            # 在拓展节点时只有第一层是有严格观测可依的，再往下的子节点可能无严格的观测
            round = self.executed_round + current_prob_update_iterations*self.plan_growth_iterations + i
            node = self.tree.select_node(round).expand()
            other_agent_policies = self.sample_other_agents()
            score = node.perform_rollout(self.robot_id, other_agent_policies, self.other_agent_info,
                                         self.observations_list, self.horizon,
                                         self.get_time(), self.determinization_iterations, self.env.get_goal())
            node.backpropagate(score, round)

    # 重置树
    def reset_tree(self):
        self.Xrn = []
        self.tree = DecMCTSNode(Agent_State(self.loc, self.observations_list), depth=0,
                                maze_dims=(self.env.height, self.env.width), Xrn=self.Xrn,
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
        return Agent_Info(self.robot_id, Agent_State(self.loc, self.observations_list), probs, self.get_time())

    # 接收消息
    def unpack_comms(self):
        copied_queue = copy.copy(self.reception_queue)
        for message_str in copied_queue:
            message = pickle.loads(codecs.decode(message_str, 'base64'))
            print("receive:",message)
            if message.robot_id == self.robot_id:
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
                robot_id = message.robot_id
                # If seen before
                if robot_id in self.other_agent_info.keys():
                    # If fresh message
                    if message.time >= self.other_agent_info[robot_id].time:
                        self.other_agent_info[robot_id] = message
                else:
                    self.other_agent_info[robot_id] = message
                self.observations_list = merge_observations(self.observations_list, message.state.obs)

        time = self.get_time()
        # Filter out-of-date messages
        if self.out_of_date_timeout is not None:
            new_other_agent = {}
            for k, v in self.other_agent_info:
                if v.time + self.out_of_date_timeout >= time:
                    new_other_agent[k] = v
                else:
                    self.times_removed_other_agent += 1
            self.other_agent_info = new_other_agent

        self.reception_queue.clear()

    # Execute_movement should be true if this is a move step, rather than just part of the computation
    def update(self, execute_action=True):
        '''
        Move to next position, update observations, update locations, run MCTS,
        publish to ROS topic
        '''
        print("-------- Update ", self.update_iterations, " Robot id ", self.robot_id, " Execute action ",
              execute_action, "Thread id ", threading.current_thread().name, "---------")
        self.update_iterations += 1
        # 上轮如果执行了动作，则一定重置树，论文中是在发生断点后才重置树，同时论文中初始化树在主循环外。
        if self.executed_action_last_update:
            self.reset_tree()
            self.beta = 0.1
            self.executed_action_last_update = False
            self.executed_round = 0

        # 获取具有较高折扣分数的动作序列字典
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

        # 执行动作
        if execute_action:
            self.executed_action_last_update = True
            if len(probs) > 0:
                # 获取具有最大概率的概率值对应的键，并返回相应的动作序列
                best_plan = max(probs, key=probs.get).get_action_sequence()
                best_action = best_plan[0]
            else:
                best_action = Action.STAY
            print("Executing action: " + str(best_action))
            self.loc = move(self.loc, best_action)
            if self.loc == self.goal_loc:
                self.complete = True
                self.close_listener()
            locmsg = (self.loc[0], self.loc[1], self.robot_id)
            locmsg_str = str(locmsg)
            self.r.publish(self.pub_loc, locmsg_str)
            self.update_observations_from_location()

    def update_observations_from_location(self):
        x, y = self.loc
        walls = self.env.get_walls_from_loc(self.loc)
        if walls[Action.UP]:
            self.observations_list[y - 1, x] = 2
        else:
            self.observations_list[y - 1, x] = 1
        if walls[Action.DOWN]:
            self.observations_list[y + 1, x] = 2
        else:
            self.observations_list[y + 1, x] = 1
        if walls[Action.LEFT]:
            self.observations_list[y, x - 1] = 2
        else:
            self.observations_list[y, x - 1] = 1
        if walls[Action.RIGHT]:
            self.observations_list[y, x + 1] = 2
        else:
            self.observations_list[y, x + 1] = 1
        self.observations_list[y, x] = 1

    def cool_beta(self):
        self.beta = self.beta * 0.9

    def sample_other_agents(self):
        return {i: agent.select_random_plan() for (i, agent) in self.other_agent_info.items()}

    def update_distribution(self, probs):
        enumerated_keys = list(enumerate(list(probs.keys())))
        print(len(enumerated_keys))

        # 构造了一个列表，除了叶节点和序号不同外，包含了全部的树信息
        args = [(i, node, probs, self.distribution_sample_iterations, self.other_agent_info, self.determinization_iterations,
          self.robot_id, self.observations_list, self.loc, self.horizon, self.get_time(),
          self.env.get_goal(), self.comms_aware_planning, self.beta) for i, node in enumerated_keys]

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


class DecMCTSNode():
    def __init__(self, state, depth, maze_dims, Xrn, parent=None, parent_action=None, comms_aware_planning=False):
        self.depth = depth # 节点所在层次
        self.state = state # agent_state包括机器人的当前位置和观测
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.Xrn = Xrn # 存放所有可能得执行路径，初始化根节点时定义，随后不断维护
        self.maze_dims = maze_dims # 元组，环境的尺寸

        # Incremented during backpropagation
        self.discounted_visits = 0 #对应算法中的折扣访问奖励
        self.discounted_score = 0 #对应算法中的折扣分数
        self.last_round_visited = 0 #上次访问时对饮的轮数
        self.unexplored_actions = self.get_legal_actions()
        self.action_sequence = None
        self.comms_aware_planning = comms_aware_planning

    # 获取节点所属的动作列表，
    def get_action_sequence(self):
        if self.action_sequence is not None:
            return self.action_sequence
        else:
            _, self.action_sequence = self.tree_history()
            return self.action_sequence

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

    # 每次只走一步，因此只要不明确为墙的位置均为可选路径
    def get_legal_actions(self):
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        x, y = self.state.loc
        actions = []
        if y - 1 >= 0 and self.state.obs[y - 1, x] != 2:
            actions.append(Action.UP)
        if y + 1 < self.maze_dims[0] and self.state.obs[y + 1, x] != 2:
            actions.append(Action.DOWN)
        if x - 1 >= 0 and self.state.obs[y, x - 1] != 2:
            actions.append(Action.LEFT)
        if x + 1 < self.maze_dims[1] and self.state.obs[y, x + 1] != 2:
            actions.append(Action.RIGHT)
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
        child_node = DecMCTSNode(next_state, self.depth + 1, self.maze_dims, self.Xrn, parent=self,
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
        start_state = None
        while curr_node.parent is not None:
            node_actions.append(curr_node.parent_action)
            curr_node = curr_node.parent
            if curr_node.parent is None:
                start_state = curr_node.state
        node_actions.reverse()
        return start_state, node_actions

    def perform_rollout(self, this_id, other_agent_policies, other_agent_info, real_obs, horizon, time,
                        determinization_iterations, goal):

        horizon_time = time + self.depth + horizon
        start_state, node_actions = self.tree_history()
        self.Xrn.append(self)

        avg = 0
        for _ in range(determinization_iterations):
            avg += compute_f(this_id, node_actions, other_agent_policies, real_obs, start_state.loc, self.state.obs,
                             other_agent_info, horizon_time, time, goal,
                             comms_aware_planning=self.comms_aware_planning) / determinization_iterations

        return avg


def compute_f(our_id, our_policy, other_agent_policies, real_obs, our_loc, our_obs, other_agent_info, steps,
              current_time, goal, comms_aware_planning):
    maze = m.generate_maze(our_obs, goal)
    # Simulate each agent separately (simulates both history and future plans)
    for id, agent in other_agent_info.items():
        maze.add_robot(id, agent.state.loc)
        # if id in other_agent_policies.keys():
        maze.simulate_i_steps(steps - agent.time, id, other_agent_policies[id])
        # else:
        #    print(id," not in other_agent policies")
        #    print("other agent policies is ",list(other_agent_policies.keys()))
        #    print("other agent info is ",list(other_agent_info.keys()))

    # Score if we took no actions
    maze.add_robot(our_id, our_loc)
    null_score = maze.get_score(real_obs, comms_aware=comms_aware_planning)

    # Score if we take our actual actions (simulates future plans)
    maze.simulate_i_steps(steps - current_time, our_id, our_policy)
    actuated_score = maze.get_score(real_obs, comms_aware=comms_aware_planning)
    return actuated_score - null_score


def merge_observations(obs1, obs2):
    return obs1.maximum(obs2)
