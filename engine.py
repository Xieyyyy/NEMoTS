import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as op
from scipy.stats import pearsonr
from torch.distributions import Categorical

from model import Model


class Engine(object):
    def __init__(self, args):
        self.args = args
        self.model = Model(args)
        self.model.p_v_net_ctx.pv_net = self.model.p_v_net_ctx.pv_net.to(self.args.device)
        self.optimizer = op.Adam(self.model.p_v_net_ctx.pv_net.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def simulate(self, data):
        X, y = data[:, :self.args.seq_in], data[:, -self.args.seq_out:]
        all_eqs, all_times, test_scores, test_data = self.model.run(X, y)
        mae, mse, corr, best_exp = OptimizedMetrics.metrics(all_eqs, test_scores, test_data)
        if len(self.model.data_buffer) > self.args.train_size:
            loss = self.train()
            return best_exp, all_times, test_data, loss, mae, mse, corr
        return best_exp, all_times, test_data, 0, mae, mse, corr

    def train(self):
        self.model.p_v_net_ctx.pv_net.train()
        print("start train neural networks...")
        cumulative_loss = 0
        for _ in range(self.args.epoch):
            self.optimizer.zero_grad()
            state_batch, seq_batch, policy_batch, value_batch, length_indices = self.preprocess_data()
            value_batch = torch.Tensor(value_batch)
            raw_dis_out, value_out = self.model.p_v_net_ctx.policy_value_batch(seq_batch, state_batch)
            value_batch[torch.isnan(value_batch)] = 0.
            value_loss = F.mse_loss(value_out, value_batch.to(value_out.device))
            dist_loss = []
            for length, sample_id in length_indices.items():
                out_policy = F.softmax(torch.stack([raw_dis_out[i] for i in sample_id])[:, :length], dim=-1)
                gt_policy = torch.Tensor([policy_batch[i] for i in sample_id]).to(out_policy.device)
                dist_target = Categorical(probs=gt_policy)
                dist_out = Categorical(probs=out_policy)
                dist_loss.append(torch.distributions.kl_divergence(dist_target, dist_out).mean())
            total_loss = value_loss + sum(dist_loss)
            cumulative_loss += total_loss.item()
            total_loss.backward(retain_graph=True)
            if self.args.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.p_v_net_ctx.pv_net.parameters(), self.args.clip)
            self.optimizer.step()
        print("end train neural networks...")
        return cumulative_loss / self.args.epoch

    def obtain_policy_length(self, policy):
        length_indices = defaultdict(list)
        for idx, sublist in enumerate(policy):
            length_indices[len(sublist)].append(idx)
        return dict(length_indices)

    def preprocess_data(self):
        non_nan_indices = [index for index, value in enumerate(self.model.data_buffer) if not math.isnan(value[3])]
        sampled_idx = random.sample(non_nan_indices, min(len(non_nan_indices), self.args.train_size))
        mini_batch = [self.model.data_buffer[i] for i in sampled_idx]
        state_batch = [data[0] for data in mini_batch]
        seq_batch = [data[1][1] for data in mini_batch]
        policy_batch = [data[2] for data in mini_batch]
        value_batch = [data[3] for data in mini_batch]
        length_indices = self.obtain_policy_length(policy_batch)
        return state_batch, seq_batch, policy_batch, value_batch, length_indices

    def eval(self, data):
        X, y = data[:, :self.args.seq_in], data[:, -self.args.seq_out:]
        all_eqs, all_times, test_scores, test_data = self.model.run(X, y)



class OptimizedMetrics:
    @staticmethod
    def metrics(exps, scores, data):
        best_index = np.argmax(scores)
        best_exp = exps[best_index]
        span, gt = data

        # Replacing the lambdify function with the new lambda function
        corrected_expression = best_exp.replace("exp", "np.exp").replace("cos", "np.cos").replace("sin",
                                                                                                  "np.sin").replace(
            "sqrt", "np.sqrt").replace("log", "np.log")
        f = lambda x: eval(corrected_expression)

        prediction = f(span)
        mae = np.mean(np.abs(prediction - gt))
        mse = np.mean((prediction - gt) ** 2)
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

        return mae, mse, corr, best_exp

# Example usage (assuming exps, scores, and data are defined)
# metrics = OptimizedMetrics.metrics(exps, scores, data)
