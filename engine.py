import numpy as np
from scipy.stats import pearsonr

from model import Model


class Engine(object):
    def __init__(self, args):
        self.args = args
        self.model = Model(args)

    def train(self, data):
        X, y = data[:, :self.args.seq_in], data[:, -self.args.seq_out:]
        all_eqs, all_times, test_scores, test_data = self.model.run(X, y)
        mae, mse, corr, best_exp = OptimizedMetrics.metrics(all_eqs, test_scores, test_data)
        if len(self.model.data_buffer) > self.args.train_size:
            loss = self.train()
            return best_exp, all_times, test_data, loss, mae, mse, corr
        return best_exp, all_times, test_data, 0, mae, mse, corr

    def optimize(self):
        pass


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
