import sys 
# sys.path.append("..") 
# sys.path.append("quest_scheduler")
# from Context.IContext import TakeTurnContext
from pypolo2.gridcontext.MCTSContext import MCTSContext
from pypolo2.gridcontext.MCTS_concurrent import MCTSConcurrentPlayer
# from DataTools.data_density_evaluation import show_single_pcolor_graph
import numpy as np
# from matplotlib import pyplot as plt
import numpy as np

test_map_size = (15,15)
test_positions = [np.array([1,1]), np.array([2,2])]
test_number = len(test_positions)

# 定义了一个均匀概率的函数
def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0

# 初始化context，是否和gridmovingcontext一致？应该是MCcontext
initial_state = MCTSContext(test_map_size, 100, 
                                    agent_init_position=test_positions,
                                    agent_number=test_number)


# mcts = MCTS(policy_value_fn)
# print(mcts.get_move(initial_state))
# print("Finished")

player = MCTSConcurrentPlayer(policy_value_fn, n_playout=2000)
if __name__ == '__main__':
  states = []
  step = 1
  while True:
    move = player.get_action(initial_state)
    states.append(initial_state.agent_curr_position.copy())
    initial_state.do_move(move)
    game_end, sq = initial_state.game_end()
    # show_single_pcolor_graph(initial_state.calculate_matrix(), *test_map_size, plt, vmax=4)
    print(len(initial_state.record))
    # plt.savefig("sq" + str(step) + ".png")
    step += 1
    if(game_end):
      print(initial_state.calculate_trace())
      break