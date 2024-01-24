import sys
from collections import defaultdict

import numpy as np


# from main import write_log
def write_log(info, file_dir):
    with open(file_dir + ".txt", 'a') as file:
        file.write(info + '\n')


class MCTS():
    def __init__(self, data_sample, base_grammars, aug_grammars, nt_nodes, max_len, max_module, aug_grammars_allowed,
                 func_score, exploration_rate=1 / np.sqrt(2), eta=0.999, train=True, aug_grammar_table=None):
        self.data_sample = data_sample
        self.base_grammars = base_grammars
        aug_grammars = aug_grammars if train else ["A->placeholder"]
        self.grammars = base_grammars + [x for x in aug_grammars if
                                         x not in base_grammars]

        self.nt_nodes = nt_nodes
        self.max_len = max_len
        self.max_module = max_module
        self.max_aug = aug_grammars_allowed

        self.good_modules = []  # 初始化一个空列表，用于存储好的模块。
        self.score = func_score
        self.exploration_rate = exploration_rate
        self.UCBs = defaultdict(
            lambda: np.zeros(len(self.grammars)))  # 初始化一个默认字典，每个键的默认值是一个零数组，数组的长度等于语法规则的数量。这个字典可能用于存储每个状态和动作对的UCB值。
        self.QN = defaultdict(
            lambda: np.zeros(2))  # 初始化一个默认字典，每个键的默认值是一个长度为2的零数组。数组的第一个元素可能表示状态的质量（Q），数组的第二个元素可能表示状态的访问次数（N）。
        self.scale = 0
        self.eta = eta
        self.train = train
        if not self.train:
            self.aug_grammar_table = aug_grammar_table

    def valid_prods(self, Node):
        """
        获取所有以给定节点开始的可能的产生规则的索引。
        """
        # 通过检查每个语法规则是否以给定的节点开始，找出所有有效的语法规则。
        valid_grammars = [x for x in self.grammars if x.startswith(Node)]
        # 返回有效语法规则在总语法规则列表中的索引。
        return [self.grammars.index(x) for x in valid_grammars]

    def tree_to_eq(self, prods):
        """
        将解析树转换为等式形式。
        """
        # print(prods)
        seq = ['f']
        for prod in prods:
            if str(prod[0]) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod[0]:
                    seq1 = seq[:ix]
                    seq2 = list(prod[3:])
                    seq3 = seq[ix + 1:]
                    seq = seq1 + seq2 + seq3
                    break
        try:
            return ''.join(seq)
        except:
            return ''

    def state_to_seq(self, state):
        """
        将状态转换为索引序列。
        """
        aug_grammars = ['f->A'] + self.grammars
        seq = np.zeros(self.max_len)
        prods = state.split(',')
        for i, prod in enumerate(prods):
            seq[i] = aug_grammars.index(prod)
        return seq

    def state_to_onehot(self, state):
        """
        将状态转换为one-hot矩阵。
        """
        aug_grammars = ['f->A'] + self.grammars
        state_oh = np.zeros([self.max_len, len(aug_grammars)])
        prods = state.split(',')
        for i in range(len(prods)):
            state_oh[i, aug_grammars.index(prods[i])] = 1
        return state_oh

    def get_ntn(self, prod, prod_idx):
        """
        从产生式（production rule）的右侧获取所有非终端节点。
        """

        # 检查产生式的索引是否大于或等于基础语法规则（base_grammars）的长度
        if prod_idx >= len(self.base_grammars):
            # 如果是，返回空列表。这是因为这种情况下，我们可以推断出prod产生式是从扩展语法规则中来的，而扩展规则通常没有非终端节点。
            return []
        else:
            # 如果不是，那么从prod的第三个字符开始（因为前三个字符一般是产生式的左侧和箭头符号），返回所有在非终端节点列表（nt_nodes）中的字符。
            # 这样，我们就得到了产生式右侧所有的非终端节点。
            ret = [i for i in prod[3:] if i in self.nt_nodes]
            return ret

    def get_unvisited(self, state, node):
        """
        Get index of all unvisited child
        此方法获取所有未访问的子节点的索引。
        """
        valid_action = self.valid_prods(node)  # 给定node开始的所有可能的产生规则的索引。
        # 使用QN获取是否是未访问节点
        valid_ret_action = [a for a in valid_action if
                            self.QN[state + ',' + self.grammars[a]][1] == 0]  # QN的第1维储存了是否访问过的信息

        return valid_ret_action

    def print_solution(self, solu, i_episode):
        '''
        此方法将解决方案打印出来。
        Parameters
        ----------
        solu
        i_episode

        Returns
        -------

        '''
        print('Episode', i_episode, solu)

    def secondary_sample(self, remain_count):
        assert not self.train
        sorted_list = sorted(self.aug_grammar_table.items(), key=lambda x: x[1], reverse=True)[:remain_count]
        # total = sum(value for _, value in sorted_list)
        # probabilities = self.softmax([value / total for _, value in sorted_list])
        probabilities = self.softmax([value for _, value in sorted_list])
        action = np.random.choice(np.arange(len(probabilities)), p=probabilities)
        return sorted_list[action][0]

    def step(self, state, action_idx, ntn):
        """
        这个方法定义了解析树遍历的一步。它会返回下一个状态，剩余的非终端节点，奖励，是否完成遍历，以及解析出的等式。
        """

        # 从grammars属性中获取索引为action_idx的语法动作(随机选择到的动作)。
        action = self.grammars[action_idx]
        action = self.secondary_sample(remain_count=10) if action == "A->placeholder" else action
        # 将选择的动作添加到当前的状态字符串。
        state = state + ',' + action
        # print(state)

        # 获取由新的动作产生的非终端节点，并更新ntn列表。注意，这里ntn[1:]是将原有ntn列表中的第一个元素（也就是被替换的非终端节点）去掉。
        ntn = self.get_ntn(action, action_idx) + ntn[1:]

        # 检查是否还有剩余的非终端节点。
        if not ntn:
            # 如果没有剩余的非终端节点，那么就通过score方法计算当前状态的得分，并将状态字符串转化为等式。
            reward, eq = self.score(self.tree_to_eq(state.split(',')), len(state.split(',')), self.data_sample,
                                    eta=self.eta)

            # 返回新的状态，空的非终端节点列表，奖励，结束标志（为真），以及等式。
            return state, ntn, reward, True, eq
        else:
            # 如果还有剩余的非终端节点，那么返回新的状态，新的非终端节点列表，奖励为0（因为还未到达终止状态），结束标志（为假），以及等式为None（因为还未到达终止状态，无法形成完整等式）。
            return state, ntn, 0, False, None

    def rollout(self, num_play, state_initial, ntn_initial):
        """
        Perform a n-play rollout simulation, get the maximum reward
        此方法执行一个n-play滚动模拟，获取最大的奖励。
        """
        best_eq = ''
        best_r = 0
        for n in range(num_play):
            done = False
            state = state_initial
            ntn = ntn_initial

            while not done:
                valid_index = self.valid_prods(ntn[0])
                action = np.random.choice(valid_index)
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)
                state = next_state
                ntn = ntn_next

                if state.count(',') >= self.max_len:
                    break

            if done:
                if reward > best_r:
                    self.update_modules(next_state, reward, eq)
                    best_eq = eq
                    best_r = reward

        return best_r, best_eq

    def nn_est_reward(self, state, network):
        reward = self.aquire_nn(state, network)[2].item()
        eq = self.tree_to_eq(state.split(','))
        return reward, eq

    def update_ucb_mcts(self, state, action):
        """
        Get the ucb score for a given child of current node
        此方法获取给定节点的子节点的UCB分数。
        """
        next_state = state + ',' + action
        Q_child = self.QN[next_state][0]
        N_parent = self.QN[state][1]
        N_child = self.QN[next_state][1]
        return Q_child / N_child + self.exploration_rate * np.sqrt(np.log(N_parent) / N_child)

    def update_QN_scale(self, new_scale):
        """
        Update the Q values self.scaled by the new best reward.
        此方法更新Q值。
        """

        if self.scale != 0:
            for s in self.QN:
                self.QN[s][0] *= (self.scale / new_scale)

        self.scale = new_scale

    def backpropogate(self, state, action_index, reward):
        """
        Update the Q, N and ucb for all corresponding decedent after a complete rollout
        此方法更新完整滚动后所有相应后代的Q，N和UCB。
        """

        # 通过动作索引从所有可能的动作列表中获取实际采取的动作
        action = self.grammars[action_index]

        # 更新该状态动作对的Q值（平均奖励值），如果self.scale为0，Q值不变
        if self.scale != 0:
            self.QN[state + ',' + action][0] += reward / self.scale
        else:
            self.QN[state + ',' + action][0] += 0

        # 更新该状态动作对被访问的次数N
        self.QN[state + ',' + action][1] += 1

        # 对于完整探索路径中的每一个状态，都更新其Q值、N值和UCB值
        while state:

            # 更新当前状态的Q值
            if self.scale != 0:
                self.QN[state][0] += reward / self.scale
            else:
                self.QN[state][0] += 0

            # 更新当前状态被访问的次数N
            self.QN[state][1] += 1

            # 根据当前的Q值和N值，更新当前状态的UCB值
            self.UCBs[state][self.grammars.index(action)] = self.update_ucb_mcts(state, action)

            # 如果当前状态是由多个子状态以“,”分隔的组合，则移除最后一个子状态，并将移除的子状态赋值给action，否则将state置为空
            if ',' in state:
                state, action = state.rsplit(',', 1)
            else:
                state = ''

    def get_policy1(self, nA, state, node, network):
        valid_action = self.valid_prods(node)
        policy_valid, _, _ = self.aquire_nn(state, network)
        policy_valid = self.softmax(policy_valid.squeeze(0).cpu().detach().numpy()[:len(valid_action)])
        sum_ucb = sum(self.UCBs[state][valid_action])
        for idx, a in enumerate(valid_action):
            policy_mcts = self.UCBs[state][a] / sum_ucb
            policy_valid[idx] = policy_valid[idx] * policy_mcts

        if len(set(policy_valid)) == 1:
            A = np.zeros(nA)
            A[valid_action] = float(1 / len(valid_action))
            return A

        A = np.zeros(nA, dtype=float)
        best_action = valid_action[np.argmax(policy_valid)]
        A[best_action] += 0.8
        A[valid_action] += float(0.2 / len(valid_action))
        return A

    def get_policy2(self, nA, UC):
        """
        Creates a random policy to select an unvisited child.（均匀分布）
        """
        if len(UC) != len(set(UC)):
            print(UC)
            print(self.grammars)
        A = np.zeros(nA, dtype=float)
        A[UC] += float(1 / len(UC))
        return A

    def get_policy3(self, state, network, node, UC):
        valid_action = self.valid_prods(node)
        _, policy_expand, _ = self.aquire_nn(state, network)
        policy_expand = policy_expand.squeeze(0).cpu().detach().numpy()[:len(valid_action)]
        return self.softmax(policy_expand), self.softmax(policy_expand[UC])

    def aquire_nn(self, state, network):
        seq = self.data_sample
        selection_dist_out, expand_dist_out, value_out = network.policy_value(seq, state)
        return selection_dist_out, expand_dist_out, value_out

    def update_modules(self, state, reward, eq):
        """
        If we pass by a concise solution with high score, we store it as an
        single action for future use.
        如果我们经过一个具有高分的简洁解决方案，我们将其存储为以后使用的单个动作。
        """
        module = state[5:]
        if state.count(',') <= self.max_module:
            if not self.good_modules:
                self.good_modules = [(module, reward, eq)]
            elif eq not in [x[2] for x in self.good_modules]:
                if len(self.good_modules) < self.max_aug:
                    self.good_modules = sorted(self.good_modules + [(module, reward, eq)], key=lambda x: x[1])
                else:
                    if reward > self.good_modules[0][1]:
                        self.good_modules = sorted(self.good_modules[1:] + [(module, reward, eq)], key=lambda x: x[1])

    def run(self, num_episodes, network, num_play=50, print_flag=False, print_freq=100):
        """
        Monte Carlo Tree Search algorithm
        此方法实现了蒙特卡洛树搜索算法。
        """
        # 获取所有语法（动作）的数量

        nA = len(self.grammars)

        # search history
        # 初始化一个用于存储搜索历史的空列表
        states = []

        # The policy we're following:
        # policy1 for fully expanded node and policy2 for not fully expanded node
        # 获取两种策略：对于已完全扩展的节点使用策略1，对于未完全扩展的节点使用策略2

        # 初始化一个用于存储奖励历史的空列表
        reward_his = []
        if self.train:
            state_records = []
            seq_records = []
            expand_policy_records = []
            value_records = []
            selection_policy_records = []
            selection_state_records = []
            selection_seq_records = []

        # 初始化最佳解决方案及其奖励为0
        best_solution = ('nothing', 0)

        for i_episode in range(1, num_episodes + 1):
            if (i_episode) % print_freq == 0 and print_flag:
                print("\rEpisode {}/{}, current best reward {}.".format(i_episode, num_episodes, best_solution[1]),
                      end="")
                sys.stdout.flush()

            # 初始化状态，非终止节点ntn和未访问的节点UC
            state = 'f->A'
            ntn = ['A']
            UC = self.get_unvisited(state, ntn[0])  # unvisited child，获取未访问的节点索引列表
            # print(UC)

            ##### check scenario: if parent node fully expanded or not ####

            # scenario 1: if current parent node fully expanded, follow policy1
            # 如果当前节点已经被完全扩展（即没有未访问的子节点）
            while not UC:
                # 按照策略1选择一个动作
                policy = self.get_policy1(nA, state, ntn[0], network)
                action = np.random.choice(np.arange(nA), p=policy)
                if self.train:
                    selection_policy_records.append(policy)
                    selection_state_records.append(state)
                    selection_seq_records.append(self.data_sample)

                # 执行选定的动作，获得新的状态、非终止节点、奖励、是否完成以及方程
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)

                # 如果新的状态之前没有出现过，那么就将它加入到搜索历史中
                if state not in states:
                    states.append(state)

                # 如果新的状态不是终止状态
                if not done:
                    # 更新状态、非终止节点和未访问的节点
                    state = next_state
                    ntn = ntn_next
                    UC = self.get_unvisited(state, ntn[0])

                    # 如果新的状态的长度超过了最大长度限制
                    if state.count(',') >= self.max_len:
                        # 将未访问的节点设置为空，进行反向传播，然后将最佳解的奖励加入到奖励历史中，并跳出循环
                        UC = []
                        self.backpropogate(state, action, 0)
                        reward_his.append(best_solution[1])
                        break

                # 如果新的状态是终止状态
                else:
                    UC = []

                    # 如果新的奖励大于之前最佳解的奖励，那么就更新模块、Q/N值和最佳解
                    if reward > best_solution[1]:
                        self.update_modules(next_state, reward, eq)
                        self.update_QN_scale(reward)
                        best_solution = (eq, reward)

                    # 进行反向传播，并将最佳解的奖励加入到奖励历史中，然后跳出循环
                    self.backpropogate(state, action, reward)
                    reward_his.append(best_solution[1])
                    break

            # scenario 2: if current parent node not fully expanded, follow policy2
            # 如果当前节点还没有被完全扩展（存在未访问的子节点），则执行expansion操作
            if UC:
                # 按照策略2拓展一个动作
                # print(state)
                policy, policy_UC = self.get_policy3(state, network, ntn[0], UC)
                # print(len(policy))
                action = np.random.choice(UC, p=policy_UC)
                # print(str((policy, policy_UC)))
                # write_log(str((self.train, list(policy_UC))), "./records/illness_prob")
                # print(action)
                # action = 11
                # 执行选定的动作的索引，获得新的状态、非终止节点、奖励、是否完成以及方程
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)

                if eq is not None and self.train:
                    state_records.append(state)
                    seq_records.append(self.data_sample)
                    expand_policy_records.append(policy)
                    value_records.append(reward)

                # 如果新的状态不是终止状态，那么进行num_play次滚动模拟，获取最大的奖励和对应的方程
                if not done:
                    # reward, eq = self.rollout(num_play, next_state, ntn_next)
                    reward, eq = self.rollout(num_play, next_state, ntn_next) if self.train else self.nn_est_reward(
                        next_state, network)
                    if state not in states:
                        states.append(state)

                # 如果新的奖励大于之前最佳解的奖励，那么就更新Q/N值和最佳解
                if reward > best_solution[1]:
                    self.update_QN_scale(reward)
                    # print((next_state, eq, reward))
                    # write_log(str((next_state, eq, reward)), "./records/illness")

                    best_solution = (eq, reward)

                # 进行反向传播，并将最佳解的奖励加入到奖励历史中
                self.backpropogate(state, action, reward)
                reward_his.append(best_solution[1])

        # write_log("----------------------------------------", "./records/illness")
        # 返回奖励历史、最佳解，优秀模块，用于训练拓展的样本，用于训练选择的样本
        if self.train:
            return reward_his, best_solution, self.good_modules, zip(state_records, seq_records, expand_policy_records,
                                                                     value_records), zip(selection_state_records,
                                                                                         selection_seq_records,
                                                                                         selection_policy_records)
        else:
            return reward_his, best_solution, None, None, None

    @staticmethod
    def softmax(x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
