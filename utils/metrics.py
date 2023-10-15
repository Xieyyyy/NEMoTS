import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1).mean(0)


def MAE(pred, true):
    # return np.mean(np.abs(pred - true))
    return torch.mean(torch.abs(pred - true))


def MAE_cpu(pred, true):
    return np.mean(np.abs(pred - true))
    # return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    # return np.mean((pred - true) ** 2)
    return torch.mean((pred - true) ** 2)


def MSE_cpu(pred, true):
    return np.mean((pred - true) ** 2)
    # return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, use_gpu):
    if use_gpu:
        mae = MAE(pred, true).cpu().item()
        mse = MSE(pred, true).cpu().item()
        rse = RSE(pred, true).cpu().item()
    else:
        mae = MAE_cpu(pred, true).item()
        mse = MSE_cpu(pred, true).item()
        rse = RSE(pred, true).item()
    # rmse = RMSE(pred, true).item()
    # mape = MAPE(pred, true).item()
    # mspe = MSPE(pred, true).item()
    # rse = RSE(pred, true).item()
    # corr = CORR(pred, true).item()

    return mae, mse, rse
