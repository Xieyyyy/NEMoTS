import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.distributions import Categorical


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, output_size=6):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: [batch_size, num_features, seq_length]
        tcn_output = self.tcn(x)  # tcn_output shape: [batch_size, num_channels[-1], seq_length]
        out = self.linear(tcn_output[:, :, -1])  # Process the last element from each sequence
        return out


class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NBeatsBlock, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # 确保输出尺寸是预测期长度
        )

    def forward(self, x):
        return self.fc_layers(x)

class NBeatsModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_blocks=3):
        super(NBeatsModel, self).__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(input_size, hidden_size, output_size) for _ in range(num_blocks)])
        self.output_size = output_size

    def forward(self, x):
        forecast = torch.zeros(x.size(0), self.output_size, device=x.device)
        for block in self.blocks:
            forecast = forecast + block(x)  # 累加每个块的预测
        return forecast



def write_log(info, file_dir):
    with open(file_dir + ".txt", 'a') as file:
        file.write(info + '\n')


def minmax_norm(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val


def inverse_min_max_normalize(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val


class Engine(object):
    def __init__(self, args):
        self.args = args
        num_inputs = self.args.seq_in
        input_size = 30  # Length of the input sequence
        hidden_size = 64  # Size of the hidden layers
        output_size = 6  # Length of the output sequence (forecast length)
        self.model = NBeatsModel(input_size, hidden_size, output_size)
        self.optimizer = op.Adam(self.model.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, data):
        # self.model.train_mode = True
        normed_data, min_val, max_val = minmax_norm(data)
        X, y = normed_data[:, :self.args.seq_in], normed_data[:, -self.args.seq_out:]
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()
        # X.shape (30,)
        # y.shape (6,)
        X = X.reshape(-1, self.args.seq_in)
        y = y.reshape(-1, self.args.seq_out)

        X_tensor = torch.tensor(X).float()
        y_tensor = torch.tensor(y).float()

        # 添加rnn

        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(X_tensor)
            # Calculate the loss
            # print(outputs.shape)
            # print(y_tensor.shape)
            loss = self.criterion(outputs, y_tensor)
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

        # 预测
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 在这个with下，所有计算得出的tensor都不会计算梯度，也就是不会进行反向传播
            forecast = self.model(X_tensor)
        # 将预测结果转换为numpy数组
        forecast_np = forecast[-self.args.seq_out:, :].squeeze(0).cpu().numpy()
        y = y.squeeze(0)
        # print(forecast_np.shape)
        # print(y.shape)
        r2 = r2_score(y, forecast_np)
        pearson_corr, _ = pearsonr(y, forecast_np)

        return r2, pearson_corr

    def eval(self, data):
        # self.model.train_mode = False
        normed_data, min_val, max_val = minmax_norm(data)
        X, y = normed_data[:, :self.args.seq_in], normed_data[:, -self.args.seq_out:]
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()
        # print(X.shape)

        X = X.reshape(-1, self.args.seq_in)
        y = y.reshape(-1, self.args.seq_out)

        X_tensor = torch.tensor(X).float()
        y_tensor = torch.tensor(y).float()
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 在这个with下，所有计算得出的tensor都不会计算梯度，也就是不会进行反向传播
            forecast = self.model(X_tensor)
        # 将预测结果转换为numpy数组
        forecast_np = forecast[-self.args.seq_out:, :].squeeze(0).cpu().numpy()
        y = y.squeeze(0)
        # forecast_np = forecast_np.squeeze(0)
        r2 = r2_score(y, forecast_np)
        pearson_corr, _ = pearsonr(y, forecast_np)

        return r2, pearson_corr

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
            state_batch_selection, seq_batch_selection, policy_batch_selection, \
            state_batch_selection_augment, seq_batch_selection_augment, policy_batch_selection_augment, \
            state_batch_rollout, seq_batch_rollout, value_batch_rollout, \
            state_batch_rollout_augment, seq_batch_rollout_augment, value_batch_rollout_augment = self.vectorize_data()

            # no augment in selection
            selection_dist_out = F.softmax(
                self.model.pv_net_ctx.network(seq_batch_selection,
                                              state_batch_selection,
                                              False,
                                              output_value=False)[0][:, :-1], dim=-1)
            # augment in selection
            selection_dist_out_augment = F.softmax(
                self.model.pv_net_ctx.network(seq_batch_selection_augment,
                                              state_batch_selection_augment,
                                              False,
                                              output_value=False)[0], dim=-1)

            rollout_value_out = self.model.pv_net_ctx.network(seq_batch_rollout,
                                                              state_batch_rollout,
                                                              False,
                                                              output_selection_dist=False)[1]
            rollout_value_out_augment = self.model.pv_net_ctx.network(seq_batch_rollout_augment,
                                                                      state_batch_rollout_augment,
                                                                      False,
                                                                      output_selection_dist=False)[1]

            kl_d_selection = self.kl_divengence(selection_dist_out, policy_batch_selection)
            kl_d_selection_augment = self.kl_divengence(selection_dist_out_augment, policy_batch_selection_augment)

            mse_value = F.mse_loss(rollout_value_out, value_batch_rollout, size_average=False)
            mse_value_augment = F.mse_loss(rollout_value_out_augment, value_batch_rollout_augment, size_average=False)

            total_loss = kl_d_selection + kl_d_selection_augment + 5 * (mse_value + mse_value_augment)
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

        state_batch_rollout, \
        seq_batch_rollout, \
        value_batch_rollout = self.prepare_data(self.model.data_buffer_rollout, value=True)

        state_batch_rollout_augment, \
        seq_batch_rollout_augment, \
        value_batch_rollout_augment = self.prepare_data(self.model.data_buffer_rollout_augment, value=True)

        state_batch_selection, \
        seq_batch_selection = self.model.pv_net_ctx.batchfy(seq_batch_selection,
                                                            state_batch_selection)  # 【64,3(lengtn),16】

        state_batch_selection_augment, \
        seq_batch_selection_augment = self.model.pv_net_ctx.batchfy(seq_batch_selection_augment,
                                                                    state_batch_selection_augment)

        state_batch_rollout, \
        seq_batch_rollout = self.model.pv_net_ctx.batchfy(seq_batch_rollout, state_batch_rollout)

        state_batch_rollout_augment, \
        seq_batch_rollout_augment = self.model.pv_net_ctx.batchfy(seq_batch_rollout_augment,
                                                                  state_batch_rollout_augment)

        policy_batch_selection, \
        policy_batch_selection_augment, \
        value_batch_rollout, \
        value_batch_rollout_augment \
            = torch.Tensor(policy_batch_selection).to(self.args.device), \
              torch.Tensor(policy_batch_selection_augment).to(self.args.device), \
              torch.Tensor(value_batch_rollout).to(self.args.device), \
              torch.Tensor(value_batch_rollout_augment).to(self.args.device)

        return state_batch_selection, seq_batch_selection, policy_batch_selection, \
               state_batch_selection_augment, seq_batch_selection_augment, policy_batch_selection_augment, \
               state_batch_rollout, seq_batch_rollout, value_batch_rollout, \
               state_batch_rollout_augment, seq_batch_rollout_augment, value_batch_rollout_augment

    def prepare_data(self, data_buffer, value=True):
        if value:
            non_nan_indices = [index for index, value in enumerate(data_buffer) if not math.isnan(value[2])]
            sampled_idx = random.sample(non_nan_indices, min(len(non_nan_indices), self.args.train_size))
        else:
            sampled_idx = random.sample(range(len(data_buffer)), self.args.train_size)
        mini_batch = [data_buffer[i] for i in sampled_idx]
        state_batch = [data[0] for data in mini_batch]
        seq_batch = [data[1][1] for data in mini_batch]
        ground_truth = [data[2] for data in mini_batch]
        return state_batch, seq_batch, ground_truth


class Metrics:
    @staticmethod
    def metrics(exps, scores, data, min_val, max_val):
        mae, mse, corr, r_squared, best_exp = None, None, 0., None, None  # 将 corr 的初始值设置为 0
        best_index = np.argmax(scores)
        best_exp = exps[best_index]
        span, gt = data
        gt = inverse_min_max_normalize(gt, min_val.item(), max_val.item())

        # Replacing the lambdify function with the new lambda function
        corrected_expression = best_exp. \
            replace("exp", "np.exp"). \
            replace("cos", "np.cos"). \
            replace("sin", "np.sin"). \
            replace("sqrt", "np.sqrt"). \
            replace("log", "np.log")
        try:
            f = lambda x: eval(corrected_expression)
            prediction = f(span)
        except (ValueError, NameError):
            write_log(corrected_expression, "./records/exception_records")
            return np.nan, np.nan, np.nan, np.nan, None

        prediction = inverse_min_max_normalize(prediction, min_val.item(), max_val.item())

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
