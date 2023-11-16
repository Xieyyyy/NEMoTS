import torch
import torch.nn as nn
import torch.optim as op

from model import Model


class Engine(object):
    def __init__(self, args):
        self.args = args
        self.model = Model(args).to(self.args.device)
        # self.optimizer = op.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # self.lr_sch = op.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
        #                                           milestones=[20, 40, 60, 80, 100, 120, 140, 160, 180],
        #                                           gamma=self.args.lr_decay,
        #                                           verbose=True)
        #
        # self.loss = nn.MSELoss(size_average=False)
        # total_num = sum(p.numel() for p in self.model.parameters())
        # trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('Total:', total_num, 'Trainable:', trainable_num)

    def train(self, inputs):
        # self.model.train()
        # self.optimizer.zero_grad()
        X, y = inputs[:, :self.args.seq_in], inputs[:, -self.args.seq_out:]

        all_eqs, success_rate, all_times, test_data = self.model(X, y)
        print(all_eqs)
        print(test_data)
        print()
        # return loss.item(), samples_num

    def eval(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.seq_out:, f_dim:]
        targets = targets[:, -self.args.seq_out:, f_dim:]
        if self.args.use_gpu_in_metrics:
            preds = outputs.detach()
            trues = targets.detach()
        else:
            preds = outputs.detach().cpu().numpy()
            trues = targets.detach().cpu().numpy()
        return preds, trues
