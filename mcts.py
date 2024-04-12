import sys
from collections import defaultdict

import numpy as np


# from main import write_log
def write_log(info, file_dir):
    with open(file_dir + ".txt", 'a') as file:
        file.write(info + '\n')


class MCTS():
    def __init__(self, input_data, supervision_data, base_grammars, aug_grammars, nt_nodes, max_len,
                 max_module,
                 aug_grammars_allowed,
                 func_score, exploration_rate=1 / np.sqrt(2), eta=0.999, train=True,
                 aug_grammar_table=None):
        self.input_data = input_data
        self.supervision_data = supervision_data if train else input_data
        self.base_grammars = base_grammars
        aug_grammars = aug_grammars if train else ["A->placeholder"]
        self.grammars = base_grammars + [x for x in aug_grammars if
                                         x not in base_grammars]

        self.nt_nodes = nt_nodes
        self.max_len = max_len
        self.max_module = max_module
        self.max_aug = aug_grammars_allowed

        self.good_modules = []  #  Initilize an empty list to store good modules
        self.score = func_score
        self.exploration_rate = exploration_rate
        self.UCBs = defaultdict(
            lambda: np.zeros(len(self.grammars)))  # Storing UCBs
        self.QN = defaultdict(
            lambda: np.zeros(2))  # Storing Q and N
        self.scale = 0
        self.eta = eta
        self.train = train
        if not self.train:
            self.aug_grammar_table = aug_grammar_table

    def valid_prods(self, Node):
        """
        Obtain the index of produced rules stated with given node
        """
        # Check if a grammar is started with a given node
        valid_grammars = [x for x in self.grammars if x.startswith(Node)]

        return [self.grammars.index(x) for x in valid_grammars]

    def tree_to_eq(self, prods):
        """
        Parsing tree to equations
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

        """
        aug_grammars = ['f->A'] + self.grammars
        seq = np.zeros(self.max_len)
        prods = state.split(',')
        for i, prod in enumerate(prods):
            seq[i] = aug_grammars.index(prod)
        return seq

    def state_to_onehot(self, state):
        """

        """
        aug_grammars = ['f->A'] + self.grammars
        state_oh = np.zeros([self.max_len, len(aug_grammars)])
        prods = state.split(',')
        for i in range(len(prods)):
            state_oh[i, aug_grammars.index(prods[i])] = 1
        return state_oh

    def get_ntn(self, prod, prod_idx):
        """

        """


        if prod_idx >= len(self.base_grammars):

            return []
        else:

            ret = [i for i in prod[3:] if i in self.nt_nodes]
            return ret

    def get_unvisited(self, state, node):
        """
        Get index of all unvisited child

        """
        valid_action = self.valid_prods(node)
        valid_ret_action = [a for a in valid_action if
                            self.QN[state + ',' + self.grammars[a]][1] == 0]

        return valid_ret_action

    def print_solution(self, solu, i_episode):
        '''
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

        """


        action = self.grammars[action_idx]
        action = self.secondary_sample(remain_count=10) if action == "A->placeholder" else action
        # 将选择的动作添加到当前的状态字符串。
        state = state + ',' + action
        # print(state)


        ntn = self.get_ntn(action, action_idx) + ntn[1:]


        if not ntn:

            reward, eq = self.score(self.tree_to_eq(state.split(',')), len(state.split(',')), self.supervision_data,
                                    eta=self.eta)


            return state, ntn, reward, True, eq
        else:

            return state, ntn, 0, False, None

    def rollout(self, num_play, state_initial, ntn_initial):
        """
        Perform a n-play rollout simulation, get the maximum reward

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
        reward = self.aquire_nn(state, network)[1].item()
        states = state.split(',')

        eq = self.tree_to_eq(states)
        return reward, eq.replace("A", "x")

    def update_ucb_mcts(self, state, action):
        """
        Get the ucb score for a given child of current node

        """
        next_state = state + ',' + action
        Q_child = self.QN[next_state][0]
        N_parent = self.QN[state][1]
        N_child = self.QN[next_state][1]
        return Q_child / N_child + self.exploration_rate * np.sqrt(np.log(N_parent) / N_child)

    def update_QN_scale(self, new_scale):
        """
        Update the Q values self.scaled by the new best reward.

        """

        if self.scale != 0:
            for s in self.QN:
                self.QN[s][0] *= (self.scale / new_scale)

        self.scale = new_scale

    def backpropogate(self, state, action_index, reward):
        """
        Update the Q, N and ucb for all corresponding decedent after a complete rollout

        """


        action = self.grammars[action_index]


        if self.scale != 0:
            self.QN[state + ',' + action][0] += reward / self.scale
        else:
            self.QN[state + ',' + action][0] += 0


        self.QN[state + ',' + action][1] += 1


        while state:


            if self.scale != 0:
                self.QN[state][0] += reward / self.scale
            else:
                self.QN[state][0] += 0


            self.QN[state][1] += 1


            self.UCBs[state][self.grammars.index(action)] = self.update_ucb_mcts(state, action)


            if ',' in state:
                state, action = state.rsplit(',', 1)
            else:
                state = ''

    def get_policy1(self, nA, state, node, network):
        valid_action = self.valid_prods(node)
        policy_valid, _, = self.aquire_nn(state, network)
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

    def aquire_nn(self, state, network):
        seq = self.input_data
        selection_dist_out, value_out = network.policy_value(seq, state)
        return selection_dist_out, value_out

    def update_modules(self, state, reward, eq):
        """
        If we pass by a concise solution with high score, we store it as an
        single action for future use.

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

        """


        nA = len(self.grammars)


        states = []

        # The policy we're following:
        # policy1 for fully expanded node and policy2 for not fully expanded node

        reward_his = []
        if self.train:
            rollout_state_records = []
            rollout_seq_records = []
            rollout_value_records = []
            selection_policy_records = []
            selection_state_records = []
            selection_seq_records = []


        best_solution = ('nothing', 0)

        for i_episode in range(1, num_episodes + 1):
            if (i_episode) % print_freq == 0 and print_flag:
                print("\rEpisode {}/{}, current best reward {}.".format(i_episode, num_episodes, best_solution[1]),
                      end="")
                sys.stdout.flush()


            state = 'f->A'
            ntn = ['A']
            UC = self.get_unvisited(state, ntn[0])  # unvisited child，
            # print(UC)

            ##### check scenario: if parent node fully expanded or not ####

            # scenario 1: if current parent node fully expanded, follow policy1

            while not UC:
                # selection phase
                policy = self.get_policy1(nA, state, ntn[0], network)
                action = np.random.choice(np.arange(nA), p=policy)
                if self.train:
                    selection_policy_records.append(policy)
                    selection_state_records.append(state)
                    selection_seq_records.append(self.input_data)


                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)


                if state not in states:
                    states.append(state)


                if not done:

                    state = next_state
                    ntn = ntn_next
                    UC = self.get_unvisited(state, ntn[0])


                    if state.count(',') >= self.max_len:

                        UC = []
                        self.backpropogate(state, action, 0)
                        reward_his.append(best_solution[1])
                        break


                else:
                    UC = []


                    if reward > best_solution[1]:
                        self.update_modules(next_state, reward, eq)
                        self.update_QN_scale(reward)
                        best_solution = (eq, reward)


                    self.backpropogate(state, action, reward)
                    reward_his.append(best_solution[1])
                    break

            # scenario 2: if current parent node not fully expanded, follow policy2
           # expansion phase
            if UC:

                # print(state)
                policy = self.get_policy2(nA, UC)
                # print(UC)
                # print(len(policy))
                action = np.random.choice(np.arange(nA), p=policy)
                # print(str((policy, policy_UC)))
                # write_log(str((self.train, list(policy_UC))), "./records/illness_prob")
                # print(action)
                # action = 11

                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)


                if not done:
                    # reward, eq = self.rollout(num_play, next_state, ntn_next)
                    reward, eq = self.rollout(num_play, next_state, ntn_next) if self.train else self.nn_est_reward(
                        next_state, network)
                    if state not in states:
                        states.append(state)

                    if eq is not "" and self.train:
                        rollout_state_records.append(next_state)
                        rollout_seq_records.append(self.input_data)
                        rollout_value_records.append(reward)


                if reward > best_solution[1]:
                    self.update_QN_scale(reward)
                    # print((next_state, eq, reward))
                    # write_log(str((next_state, eq, reward)), "./records/illness")

                    best_solution = (eq, reward)


                self.backpropogate(state, action, reward)
                reward_his.append(best_solution[1])

        # write_log("----------------------------------------", "./records/illness")

        if self.train:
            return reward_his, best_solution, self.good_modules, \
                   zip(rollout_state_records, rollout_seq_records, rollout_value_records), \
                   zip(selection_state_records, selection_seq_records, selection_policy_records)
        else:
            return reward_his, best_solution, None, None, None

    @staticmethod
    def softmax(x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
