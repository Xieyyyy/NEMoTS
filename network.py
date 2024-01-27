import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 更新 padding
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
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
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size,
                                   dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # TCN 的输出是 (batch_size, num_channels, seq_length)
        # 我们需要最后一个时间步的输出
        tcn_out = self.tcn(x.transpose(1, 2))[:, :, -1]
        # 线性变换以匹配输出尺寸
        out = self.linear(tcn_out)
        # 调整输出形状为 (batch_size, 1, output_size)
        return out.unsqueeze(1)


class PVNet(nn.Module):
    def __init__(self, grammar_vocab, hidden_dim=16):
        super(PVNet, self).__init__()
        self.grammar_vocab = grammar_vocab
        self.embedding_table = nn.Embedding(len(self.grammar_vocab) + 1, hidden_dim)
        self.state_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.seq_tcn = TCNModel(input_size=1, output_size=16, num_channels=[16, 16, 16], kernel_size=5, dropout=0.2)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True))

        self.selection_dist = nn.Linear(hidden_dim * 2, len(self.grammar_vocab) - 1)
        self.expand_dist = nn.Linear(hidden_dim * 2, len(self.grammar_vocab) - 1)
        self.value = nn.Linear(hidden_dim * 2, 1)

    def forward(self, seq, state_id, need_embeddings=True, output_selection_dist=True, output_expand_dist=True,
                output_value=True):
        state = self.embedding_table(state_id.long()) if need_embeddings else state_id
        seq = seq.unsqueeze(-1)
        out_state, _ = self.state_lstm(state)
        out_seq = self.seq_tcn(seq)

        out = torch.cat([out_seq[:, -1, :], out_state[:, -1, :]], dim=-1)
        out = self.mlp(out)
        selection_dist_out, expand_dist_out, value_out = None, None, None

        # Compute outputs conditionally
        if output_selection_dist:
            selection_dist_out = self.selection_dist(out)
        if output_expand_dist:
            expand_dist_out = self.expand_dist(out)
        if output_value:
            value_out = F.sigmoid(self.value(out))

        return selection_dist_out, expand_dist_out, value_out


class PVNetCtx:
    def __init__(self, grammars, device):
        self.device = device
        self.base_grammar = grammars
        self.grammar_vocab = ['f->A'] + grammars + ['augment']
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.grammar_vocab)}
        self.network = PVNet(self.grammar_vocab).to(self.device)

    def policy_value(self, seq, state):
        state_list = state.split(",")
        state_idx = torch.Tensor(
            [self.symbol2idx[item] if item in (['f->A'] + self.base_grammar) else self.symbol2idx['augment'] for item in
             state_list]).to(self.device)
        seq = torch.Tensor(seq).to(self.device)
        selection_dist_out, expand_dist_out, value_out = self.network(seq[1, :].unsqueeze(0), state_idx.unsqueeze(0))
        return selection_dist_out, expand_dist_out, value_out

    def batchfy(self, seqs, states):
        for idx, seq in enumerate(seqs):
            seqs[idx] = torch.Tensor(seq).to(self.device)
        states_list = []
        for idx, state in enumerate(states):
            state_list = state.split(",")
            state_idx = torch.Tensor(
                [self.symbol2idx[item] if item in self.base_grammar else self.symbol2idx['augment'] for item in
                 state_list]).to(self.device)
            state_emb = self.network.embedding_table(state_idx.long())
            states_list.append(state_emb)
        max_len = max(state.shape[0] for state in states_list)
        for idx, state in enumerate(states_list):
            if state.shape[0] < max_len:
                states_list[idx] = F.pad(state, (0, 0, 0, max_len - state.shape[0]), "constant", 0)
        states = torch.stack(states_list).to(self.device)
        seqs = torch.stack(seqs).to(self.device)
        # selection_dist_outs, expand_dist_outs, value_outs = self.network(seqs, states, False)
        return states, seqs
