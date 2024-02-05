import argparse
import time

import numpy as np
import torch

from data_provider.data_factory import data_provider
from engine import Engine

# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

parser.add_argument("--device", type=str, default="cpu")
# -- data processing
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='exchange_rate/exchange_rate.csv', help='data file')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding,'
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, '
                         'MS:multivariate predict univariate')

# -- model argments
parser.add_argument('--used_dimension', type=int, default=0)
parser.add_argument('--symbolic_lib', type=str, default="NEMoTS")
parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--max_module_init', type=int, default=10)
parser.add_argument('--num_transplant', type=int, default=2)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--num_aug', type=int, default=1)
parser.add_argument('--exploration_rate', type=float, default=1 / np.sqrt(2))
parser.add_argument('--transplant_step', type=int, default=1000)
parser.add_argument('--norm_threshold', type=float, default=1e-5)

# -- training
parser.add_argument("--seed", type=int, default=52, help='random seed')
parser.add_argument("--round", type=int, default=1, help='epoch')
parser.add_argument("--epoch", type=int, default=50, help='epoch')
parser.add_argument("--seq_in", type=int, default=30, help='length of input seq')
parser.add_argument("--seq_out", type=int, default=6, help='length of output seq')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument("--batch_size", type=int, default=1, help='default')
parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
parser.add_argument("--dropout", type=float, default=0.05, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument("--clip", type=float, default=5., help='gradient clip')
parser.add_argument("--lr_decay", type=float, default=1)
parser.add_argument("--train_size", type=float, default=64)

# -- analysis
parser.add_argument("--recording", action="store_true", default=False)
parser.add_argument("--tag", type=str, default="illness", help='')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.device = torch.device(args.device)
torch.backends.cudnn.benchmark = True


def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


def write_log(info, file_dir):
    with open(file_dir + ".txt", 'a') as file:
        file.write(info + '\n')


def main():
    train_data, train_loader = get_data(args=args, flag='train')
    vali_data, vali_loader = get_data(args=args, flag='val')
    test_data, test_loader = get_data(args=args, flag='test')

    if args.recording:
        write_log(str(args), "./records/" + args.tag)

    engine = Engine(args)

    print("start training...")
    for round_num in range(args.round + 1):
        train_loss = 0
        train_n_samples = 0
        for iter, (data, _, _, _) in enumerate(train_loader):
            iter_start_time = time.time()  # 记录迭代开始时间

            train_data = data[..., args.used_dimension].float()
            r2, corr = engine.train(train_data)

            # train_loss += loss
            # train_n_samples += 1
            #
            # iter_end_time = time.time()  # 记录迭代结束时间
            # iter_duration = iter_end_time - iter_start_time  # 计算迭代耗时

            log = 'Iter: {:03d}, Train R2 Pred: {:.4f}, Train CORR Pred: {:.4f}'.format(
                iter, r2, corr)

            print(log, flush=True)  # 打印日志

            if args.recording:
                # write_log(str(best_exp), "./records/" + args.tag)
                write_log(str(test_data[1]), "./records/" + args.tag)
                write_log(log, "./records/" + args.tag)  # 将含有时间消耗的日志写入文件
                write_log("----------------------------------------", "./records/" + args.tag)
                # torch.save(engine.model.pv_net_ctx.network.state_dict(), 'model_checkpoint.pth')
                # if train_loss != 0:
                #     sw.add_scalar('Training/Loss', train_loss / train_n_samples, iter)

        torch.cuda.empty_cache()

        for iter, (data, _, _, _) in enumerate(test_loader):
            iter_start_time = time.time()  # 记录迭代开始时间

            test_data = data[..., args.used_dimension].float()
            r2, corr = engine.eval(test_data)

            iter_end_time = time.time()  # 记录迭代结束时间
            iter_duration = iter_end_time - iter_start_time  # 计算迭代耗时

            # 构建日志信息，包括时间消耗
            log = 'Iter: {:03d}, Test R2 Pred: {:.4f}, Test CORR Pred: {:.4f}'.format(
                iter, r2, corr)

            print(log, flush=True)  # 打印日志

            if args.recording:
                # write_log(str(best_exp), "./records/" + args.tag)
                write_log(str(test_data[1]), "./records/" + args.tag)
                write_log(log, "./records/" + args.tag)
                write_log("\n", "./records/" + args.tag)


if __name__ == '__main__':
    main()
