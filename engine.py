import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as op
from scipy.stats import pearsonr
from torch.distributions import Categorical

from model import Model


def write_log(info, file_dir):
    with open(file_dir + ".txt", 'a') as file:
        file.write(info + '\n')


class Engine(object):
    def __init__(self, args):
        self.args = args
        self.model = Model(args)
        self.optimizer = op.Adam(self.model.pv_net_ctx.network.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def train(self, data):
        self.model.train_mode = True
        X, y = data[:, :self.args.seq_in], data[:, -self.args.seq_out:]
        all_eqs, test_scores, test_data = self.model.run(X, y)
        mae, mse, corr, r_squared, best_exp = Metrics.metrics(all_eqs, test_scores, test_data)
        if all(len(data_buffer) > self.args.train_size for data_buffer in
               [self.model.data_buffer_selection, self.model.data_buffer_selection_augment,
                self.model.data_buffer_expand, self.model.data_buffer_expand_augment]):
            loss = self.optimize()
            return best_exp, test_data, loss, mae, mse, r_squared, corr
        return best_exp, test_data, 0, mae, mse, r_squared, corr

    def eval(self, data):
        self.model.train_mode = False
        X, y = data[:, :self.args.seq_in], data[:, -self.args.seq_out:]
        all_eqs, test_scores, test_data = self.model.run(X)
        mae, mse, corr, r_squared, best_exp = Metrics.metrics(all_eqs, test_scores, test_data)
        return best_exp, test_data, mae, mse, r_squared, corr

    @staticmethod
    def kl_divengence(P, Q):
        dist_target = Categorical(probs=P)
        dist_out = Categorical(probs=Q)

        # Compute KL divergence and take the sum
        kl_div = torch.distributions.kl_divergence(dist_target, dist_out).sum()
        return kl_div

    @staticmethod
    def mix_distribution(distribution_D, r, alpha=2):
        """
        Return a mixed distribution for a batch of samples based on the parameter r and alpha.
        Each sample in the batch is combined with a uniform distribution,
        adjusted non-linearly by r and alpha.

        :param distribution_D: 2D array where each row represents a discrete distribution D.
        :param r: Parameter ranging from 0 to 1.
        :param alpha: Parameter that controls the non-linearity.
        :return: 2D array of mixed distributions.
        """
        # Number of elements in each distribution
        n = distribution_D.shape[1]
        # Create a uniform distribution using NumPy
        uniform_U = torch.full((1, n), 1.0 / n, device=distribution_D.device)
        # Non-linear transformation of r
        r_prime = 1 - torch.exp(-alpha * r)
        # Mix the distributions
        mixed_D = (1 - r_prime) * uniform_U + r_prime * distribution_D
        return mixed_D

    def optimize(self):
        self.model.pv_net_ctx.network.train()
        print("start train neural networks...")
        cumulative_loss = 0
        for _ in range(self.args.epoch):
            self.optimizer.zero_grad()
            value_batch_expand, \
            value_batch_expand_augment, \
            policy_batch_selection, \
            policy_batch_selection_augment, \
            state_batch_selection, \
            seq_batch_selection, \
            state_batch_selection_augment, \
            seq_batch_selection_augment, \
            state_batch_expand, \
            seq_batch_expand, \
            state_batch_expand_augment, \
            seq_batch_expand_augment = self.vectorize_data()

            # no augment in selection
            selection_dist_out = F.softmax(
                self.model.pv_net_ctx.network(seq_batch_selection,
                                              state_batch_selection,
                                              False,
                                              output_expand_dist=False,
                                              output_value=False)[0][:, :-1], dim=-1)
            selection_dist_out_augment = F.softmax(
                self.model.pv_net_ctx.network(seq_batch_selection,
                                              state_batch_selection,
                                              False,
                                              output_expand_dist=False,
                                              output_value=False)[0], dim=-1)

            expand_dist_out, reward_out = self.model.pv_net_ctx.network(seq_batch_expand,
                                                                        state_batch_expand,
                                                                        False,
                                                                        output_selection_dist=False)[-2:]

            expand_dist_out_augment, reward_out_augment = self.model.pv_net_ctx.network(seq_batch_expand_augment,
                                                                                        state_batch_expand_augment,
                                                                                        False,
                                                                                        output_selection_dist=False)[
                                                          -2:]
            expand_dist_out = F.softmax(expand_dist_out, dim=-1)
            expand_dist_out_augment = F.softmax(expand_dist_out_augment, dim=-1)

            kl_d_selection = self.kl_divengence(selection_dist_out, policy_batch_selection)
            kl_d_selection_augment = self.kl_divengence(selection_dist_out_augment, policy_batch_selection_augment)

            mse_value = F.mse_loss(reward_out, value_batch_expand, size_average=False)
            mse_value_augment = F.mse_loss(reward_out_augment, value_batch_expand_augment, size_average=False)

            corrected_distribution = self.mix_distribution(expand_dist_out, reward_out)
            corrected_distribution_augment = self.mix_distribution(expand_dist_out_augment, reward_out_augment)

            kl_d_expand = self.kl_divengence(expand_dist_out, corrected_distribution)
            kl_d_expand_augment = self.kl_divengence(expand_dist_out_augment, corrected_distribution_augment)

            total_loss = kl_d_selection + kl_d_selection_augment + mse_value + mse_value_augment + kl_d_expand + kl_d_expand_augment
            cumulative_loss += total_loss.item()
            total_loss.backward(retain_graph=True)
            if self.args.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.pv_net_ctx.network.parameters(), self.args.clip)
            self.optimizer.step()
        print("end train neural networks...")
        return cumulative_loss / self.args.epoch

    def vectorize_data(self):
        state_batch_selection, \
        seq_batch_selection, \
        policy_batch_selection = self.prepare_data(self.model.data_buffer_selection, value=False)

        state_batch_selection_augment, \
        seq_batch_selection_augment, \
        policy_batch_selection_augment = self.prepare_data(self.model.data_buffer_selection_augment, value=False)

        state_batch_expand, \
        seq_batch_expand, \
        policy_batch_expand, \
        value_batch_expand = self.prepare_data(self.model.data_buffer_expand)

        state_batch_expand_augment, \
        seq_batch_expand_augment, \
        policy_batch_expand_augment, \
        value_batch_expand_augment = self.prepare_data(self.model.data_buffer_expand_augment)

        value_batch_expand, \
        value_batch_expand_augment, \
        policy_batch_selection, \
        policy_batch_selection_augment = \
            torch.Tensor(value_batch_expand).to(self.args.device), \
            torch.Tensor(value_batch_expand_augment).to(self.args.device), \
            torch.Tensor(policy_batch_selection).to(self.args.device), \
            torch.Tensor(policy_batch_selection_augment).to(self.args.device)  # [64],[64],[64,11],[64,11]

        state_batch_selection, \
        seq_batch_selection = self.model.pv_net_ctx.batchfy(seq_batch_selection,
                                                            state_batch_selection)  # 【64,3(lengtn),16】

        state_batch_selection_augment, \
        seq_batch_selection_augment = self.model.pv_net_ctx.batchfy(seq_batch_selection_augment,
                                                                    state_batch_selection_augment)

        state_batch_expand, \
        seq_batch_expand = self.model.pv_net_ctx.batchfy(seq_batch_expand, state_batch_expand)

        state_batch_expand_augment, \
        seq_batch_expand_augment = self.model.pv_net_ctx.batchfy(seq_batch_expand_augment, state_batch_expand_augment)

        return value_batch_expand, value_batch_expand_augment, policy_batch_selection, policy_batch_selection_augment, \
               state_batch_selection, seq_batch_selection, state_batch_selection_augment, seq_batch_selection_augment, \
               state_batch_expand, seq_batch_expand, state_batch_expand_augment, seq_batch_expand_augment

    def prepare_data(self, data_buffer, value=True):
        if value:
            non_nan_indices = [index for index, value in enumerate(data_buffer) if not math.isnan(value[3])]
            sampled_idx = random.sample(non_nan_indices, min(len(non_nan_indices), self.args.train_size))
        else:
            sampled_idx = random.sample(range(len(data_buffer)), self.args.train_size)
        mini_batch = [data_buffer[i] for i in sampled_idx]
        state_batch = [data[0] for data in mini_batch]
        seq_batch = [data[1][1] for data in mini_batch]
        policy_batch = [data[2] for data in mini_batch]
        if value:
            value_batch = [data[3] for data in mini_batch]
            return state_batch, seq_batch, policy_batch, value_batch
        return state_batch, seq_batch, policy_batch


class Metrics:
    @staticmethod
    def metrics(exps, scores, data):
        best_index = np.argmax(scores)
        best_exp = exps[best_index]
        span, gt = data

        # Replacing the lambdify function with the new lambda function
        corrected_expression = best_exp.replace("exp", "np.exp").replace("cos", "np.cos").replace("sin",
                                                                                                  "np.sin").replace(
            "sqrt", "np.sqrt").replace("log", "np.log")
        try:
            f = lambda x: eval(corrected_expression)
        except ValueError or NameError:
            write_log(corrected_expression, "./records/exception_records")
            return np.nan, np.nan, np.nan, np.nan, None

        prediction = f(span)
        mae = np.mean(np.abs(prediction - gt))
        mse = np.mean((prediction - gt) ** 2)

        # Calculating R-squared
        ss_res = np.sum((gt - prediction) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        try:
            corr, _ = pearsonr(prediction, gt)
        except ValueError:
            if (np.isnan(prediction) | np.isinf(prediction)).any():
                corr = 0.
            elif (np.isnan(gt) | np.isinf(gt)).any():
                valid_indices = np.where(~np.isnan(gt) & ~np.isinf(gt))[0]
                valid_gt = gt[valid_indices]
                valid_pred = prediction[valid_indices]
                corr, _ = pearsonr(valid_pred, valid_gt)
        except TypeError:
            if type(prediction) is float:
                corr = 0.

        return mae, mse, corr, r_squared, best_exp
