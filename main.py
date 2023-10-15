import argparse
import logging
import time

import numpy as np
import torch

from data_provider.data_factory import data_provider
# from torch.utils.tensorboard import SummaryWriter
from engine import Engine

parser = argparse.ArgumentParser(description="NEMoTS Arguments")

parser.add_argument("--device", type=str, default="cpu")

parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='illness/national_illness.csv', help='data file')
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
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

parser.add_argument('--used_dimension', type=int, default=1)
parser.add_argument('--symbolic_lib', type=str, default="NEMoTS")
parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--max_module_init', type=int, default=10)
parser.add_argument('--num_transplant', type=int, default=2)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--num_aug', type=int, default=0)
parser.add_argument('--exploration_rate', type=float, default=1 / np.sqrt(2))
parser.add_argument('--transplant_step', type=int, default=1000)
parser.add_argument('--norm_threshold', type=float, default=1e-5)

parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--epoch", type=int, default=50, help='epoch')
parser.add_argument("--round", type=int, default=5, help='round')
parser.add_argument("--seq_in", type=int, default=84, help='length of input seq')
parser.add_argument("--seq_out", type=int, default=12, help='length of output seq')
parser.add_argument("--train_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument("--clip", type=float, default=5., help='gradient clip')

parser.add_argument("--recording", action="store_true")
parser.add_argument("--tag", type=str, default="records")
parser.add_argument("--logtag", type=str, default="records_logtag")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.device = torch.device(args.device)
torch.backends.cudnn.benchmark = True


def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


def write_log(info, file_dir):
    logger = logging.getLogger('info_recorder')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(file_dir)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(info)


def main():
    train_data, train_loader = get_data(args=args, flag='train')
    vali_data, vali_loader = get_data(args=args, flag='val')
    test_data, test_loader = get_data(args=args, flag='test')

    if args.recording:
        sw = SummaryWriter(comment=args.tag)
        write_log(str(args), "./records/" + args.tag)
        write_log(str(args), "./records/" + args.logtag)

    engine = Engine(args)
    print("start training...")

    for round_num in range(args.round + 1):
        t1 = time.time()
        train_loss = 0.
        train_n_samples = 0
        train_maes, train_mses, train_corrs, test_maes, test_mses, test_corrs = [], [], [], [], [], []
        for iter, (data, _, _, _) in enumerate(train_loader):
            train_data = data[..., args.used_dimension].float()
            best_exp, all_times, test_data, loss, mae, mse, corr = engine.simulate(train_data)
            train_maes.append(mae)
            train_mses.append(mse)
            train_corrs.append(corr)
            train_loss += loss
            train_n_samples += 1

            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MSE: {:.4f}, Train CORR: {:.4f}'
            print(log.format(iter, train_loss / train_n_samples, mae, mse, corr), flush=True)
            if args.recording:
                write_log(str(best_exp), "./records/" + args.tag)
                write_log(str(test_data[1]), "./records/" + args.tag)
                write_log(log.format(iter, train_loss / train_n_samples, mae, mse, corr), "./records/" + args.logtag)

        torch.cuda.empty_cache()

        print('eval...')
        test_n_samples = 0
        for iter, (data, _, _, _) in enumerate(test_loader):
            test_data = data[..., args.used_dimension].float()
            mae, mse, corr, all_eqs, test_data = engine.eval(test_data)
            test_maes.append(mae)
            test_mses.append(mse)
            test_corrs.append(corr)

        log = 'Epoch: {:03d}, Test MAE: {:.4f}, Test MSE: {:.4f}, Test CORR: {:.4f}, ' \
              'Training Time: {:.4f}/epoch'
        print(log.format(round_num, sum(test_maes) / len(test_maes),
                         sum(test_mses) / len(test_mses), sum(test_corrs) / len(test_corrs), time.time() - t1),
              flush=True)

        if args.recording:
            sw.add_scalar('Loss/train', train_loss / train_n_samples, global_step=round_num)
            sw.add_scalar('MAE/valid', sum(test_maes) / len(test_maes), global_step=round_num)
            sw.add_scalar('MSE/valid', sum(test_mses) / len(test_mses), global_step=round_num)
            sw.add_scalar('RSE/valid', sum(test_corrs) / len(test_corrs), global_step=round_num)


if __name__ == '__main__':
    main()
