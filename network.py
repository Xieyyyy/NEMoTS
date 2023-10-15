import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class PVNet(nn.Module):
    def __init__(self, grammar_vocab, num_transplant, hidden_dim=16):
        super(PVNet, self).__init__()
        self.grammar_vocab = grammar_vocab
        self.num_transplant = num_transplant
        self.embedding_table = nn.Embedding(len(self.grammar_vocab), hidden_dim)
        self.lstm_state = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.lstm_seq = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True))

        self.dist_out = nn.Linear(hidden_dim * 2, len(self.grammar_vocab) + num_transplant - 2)
        self.value_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, seq, state_idx, need_embeddings=True):
        if need_embeddings:
            state = self.embedding_table(state_idx.long())
        else:
            state = state_idx

        seq = seq.unsqueeze(-1)
        out_state, _ = self.lstm_state(state)
        out_seq, _ = self.lstm_seq(seq)

        out = torch.cat([out_state[:, -1, :], out_seq[:, -1, :]], dim=-1)
        out = self.mlp(out)
        raw_dist_out = self.dist_out(out)
        raw_dist_out = torch.where(torch.isnan(raw_dist_out), torch.zeros_like(raw_dist_out), raw_dist_out)
        value_out = self.value_out(out)
        return raw_dist_out, value_out


class PVNetCtx:
    def __init__(self, grammars, num_transplant, device):
        self.device = device
        self.grammar_vocab = ['f->A'] + grammars + ['placeholer' + str(i) for i in
                                                    range(num_transplant)]
        self.grammar_vocab_backups = copy.deepcopy(self.grammar_vocab)
        self.num_transplant = num_transplant
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.grammar_vocab)}
        self.pv_net = PVNet(self.grammar_vocab, num_transplant).to(self.device)

    def policy_value(self, seq, state):
        state_list = state.split(",")
        state_idx = torch.Tensor([self.symbol2idx[item] for item in state_list]).to(self.device)
        seq = torch.Tensor(seq).to(self.device)
        raw_dist_out, value_out = self.pv_net(seq[1, :].unsqueeze(0), state_idx.unsqueeze(0))
        return raw_dist_out, value_out

    def process_state(self, state):
        unknown_counter = 0
        for i in range(len(state)):
            if state[i] not in self.grammar_vocab:
                state[i] = "placeholder" + str(unknown_counter)
                unknown_counter += 1
        return state

    def policy_value_batch(self, seqs, states):
        for idx, seq in enumerate(seqs):
            seqs[idx] = torch.Tensor(seq).to(self.device)

        states_list = []
        for idx, state in enumerate(states):
            state_list = state.split(",")
            processed_state_list = self.process_state(state_list)
            state_idx = torch.Tensor([self.symbol2idx[item] for item in processed_state_list]).to(self.device)
            state_emb = self.pv_net.embedding_table(state_idx.long())
            states_list.append(state_emb)
        max_len = max(state.shape[0] for state in states_list)
        for idx, state in enumerate(states_list):
            if state.shape[0] < max_len:
                states_list[idx] = F.pad(state, (0, 0, 0, max_len - state.shape[0]), "constant", 0)

        states = torch.stack(states_list).to(self.device)
        seqs = torch.stack(seqs).to(self.device)
        raw_dist_out, value_out = self.pv_net(seqs, states, False)
        return raw_dist_out, value_out

    def update_grammar_vocab_name(self, aug_grammars):
        for idx, grammar in enumerate(aug_grammars):
            self.grammar_vocab[self.grammar_vocab.index("placeholer" + str(idx))] = grammar

    def reset_grammar_vocab_name(self):
        self.grammar_vocab = copy.deepcopy(self.grammar_vocab_backups)
