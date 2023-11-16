import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, grammar_vocab, hidden_dim):
        super(PVNet, self).__init__()
        self.grammar_vocab = grammar_vocab
        self.embedding_table = nn.Embedding(len(self.grammar_vocab) + 1, hidden_dim)
        self.state_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.seq_lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True))

        self.selection_dist = nn.Linear(hidden_dim * 2, len(self.grammar_vocab) + 1)
        self.expand_dist = nn.Linear(hidden_dim * 2, len(self.grammar_vocab) + 1)
        self.value = nn.Linear(hidden_dim * 2, 1)
