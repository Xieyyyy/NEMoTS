from collections import defaultdict, deque

import numpy as np


import json
import score
import symbolics
from mcts import MCTS
from network import PVNetCtx


class Model():
    def __init__(self, args, train=True):

        # Extract the properties from args
        self.symbolic_lib = args.symbolic_lib
        self.max_len = args.max_len
        self.max_module_init = args.max_module_init
        self.num_transplant = args.num_transplant
        self.num_runs = args.num_runs
        self.eta = args.eta
        self.num_aug = args.num_aug
        self.exploration_rate = args.exploration_rate
        self.transplant_step = args.transplant_step
        self.norm_threshold = args.norm_threshold
        self.device = args.device

        # Other initializations
        self.base_grammar = symbolics.rule_map[self.symbolic_lib]
        self.nt_nodes = symbolics.ntn_map[self.symbolic_lib]
        self.score_with_est = score.score_with_est
        self.data_buffer_selection = deque(maxlen=1024)
        self.data_buffer_selection_augment = deque(maxlen=1024)
        self.data_buffer_expand = deque(maxlen=1024)
        self.data_buffer_expand_augment = deque(maxlen=1024)
        self.pv_net_ctx = PVNetCtx(grammars=self.base_grammar, device=self.device)

        self.aug_grammars_counter = defaultdict(lambda: 0)
        # with open("./aug_grammars.json", "r") as f:
        #     self.aug_grammars_counter = json.load(f)

        self.train_mode = train

    def run(self, X, y=None):

        assert X.size(0) == 1
        if self.train_mode:
            assert y is not None
            X = X.squeeze(0)
            y = y.squeeze(0)

            time_idx = np.arange(X.size(0) + y.shape[0])
            input_data = np.vstack([time_idx[:X.size(0)], X])

            supervision_data = np.vstack([time_idx, np.concatenate([X, y])])
        else:
            X = X.squeeze(0)
            time_idx = np.arange(X.size(0))
            input_data = np.vstack([time_idx[:X.size(0)], X])
            supervision_data = np.vstack([time_idx, X])

        all_eqs, test_scores, supervision_data = self.train(input_data,
                                                            supervision_data) if self.train_mode else self.eval(
            input_data)
        return all_eqs, test_scores, supervision_data

    def eval(self, input_data):
        all_eqs = []
        test_scores = []

        for i_test in range(self.num_runs):
            best_solution = ('nothing', 0)

            max_module = self.max_module_init  # 设置最大模块
            reward_his = []  # 初始化奖励历史
            aug_grammars = []  # 初始化增强语法

            mcts = MCTS(input_data=input_data,
                        supervision_data=None,
                        base_grammars=self.base_grammar,
                        aug_grammars=aug_grammars,
                        nt_nodes=self.nt_nodes,
                        max_len=self.max_len,
                        max_module=max_module,
                        aug_grammars_allowed=self.num_aug,
                        func_score=self.score_with_est,
                        exploration_rate=self.exploration_rate,
                        eta=self.eta,
                        train=False,
                        aug_grammar_table=self.aug_grammars_counter)

            _, current_solution, _, _, _ = mcts.run(self.transplant_step,
                                                    network=self.pv_net_ctx,
                                                    num_play=10,
                                                    print_flag=True)
            reward_his.append(best_solution[1])
            if current_solution[1] > best_solution[1]:
                best_solution = current_solution

            test_score = \
                self.score_with_est(score.simplify_eq(best_solution[0]), 0, input_data, eta=self.eta)[0]
            all_eqs.append(score.simplify_eq(best_solution[0]))
            test_scores.append(test_score)

            print('best solution: {}'.format(score.simplify_eq(best_solution[0])))
            print('test score: {}'.format(test_score))
            print()

        return all_eqs, test_scores, input_data

    def train(self, input_data, supervision_data):
        all_eqs = []
        test_scores = []

        module_grow_step = (self.max_len - self.max_module_init) / self.num_transplant

        for i_test in range(self.num_runs):
            best_solution = ('nothing', 0)

            exploration_rate = self.exploration_rate  # 设置探索率
            max_module = self.max_module_init  # 设置最大模块
            reward_his = []  # 初始化奖励历史
            best_modules = []  # 初始化最佳模块
            aug_grammars = []  # 初始化增强语法

            for i_itr in range(self.num_transplant):
                mcts = MCTS(input_data=input_data,
                            supervision_data=supervision_data,
                            base_grammars=self.base_grammar,
                            aug_grammars=aug_grammars,
                            nt_nodes=self.nt_nodes,
                            max_len=self.max_len,
                            max_module=max_module,
                            aug_grammars_allowed=self.num_aug,
                            func_score=self.score_with_est,
                            exploration_rate=self.exploration_rate,
                            eta=self.eta)

                _, current_solution, good_modules, expand_data, selection_data = mcts.run(self.transplant_step,
                                                                                          network=self.pv_net_ctx,
                                                                                          num_play=10,
                                                                                          print_flag=True)
                if i_itr == 0:
                    self.data_buffer_selection.extend(list(selection_data)[:])
                    self.data_buffer_expand.extend(list(expand_data)[:])
                else:
                    self.data_buffer_selection_augment.extend(list(selection_data)[:])
                    self.data_buffer_expand_augment.extend(list(expand_data)[:])

                # 如果没有最佳模块，则将好的模块赋值给最佳模块
                if not best_modules:
                    best_modules = good_modules
                else:
                    # 否则，将最佳模块和好的模块合并，并按照评分进行排序
                    best_modules = sorted(list(set(best_modules + good_modules)), key=lambda x: x[1])

                # 更新增强语法
                aug_grammars = [x[0] for x in best_modules[-self.num_aug:]]
                for grammar in aug_grammars:
                    self.aug_grammars_counter[grammar] += 1

                # 将最佳解决方案的评分添加到奖励历史中
                reward_his.append(best_solution[1])

                # 如果当前解决方案的评分大于最佳解决方案的评分，则更新最佳解决方案
                if current_solution[1] > best_solution[1]:
                    best_solution = current_solution
                print(current_solution[0])

                # 增加最大模块
                max_module += module_grow_step
                # 增加探索率
                exploration_rate *= 5

                # 检查是否发现了解决方案。如果是，提前停止。
                test_score = \
                    self.score_with_est(score.simplify_eq(best_solution[0]), 0, supervision_data, eta=self.eta)[0]

            all_eqs.append(score.simplify_eq(best_solution[0]))
            test_scores.append(test_score)

            print('\n{} tests complete after {} iterations.'.format(i_test + 1, i_itr + 1))
            print('best solution: {}'.format(score.simplify_eq(best_solution[0])))
            print('test score: {}'.format(test_score))
            print()

        return all_eqs, test_scores, supervision_data
