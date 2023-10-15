import _thread
import threading
from contextlib import contextmanager

import numpy as np
from numpy import *
from scipy.optimize import minimize
from sympy import simplify, expand


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def simplify_eq(eq):
    return str(expand(simplify(eq)))


def prune_poly_c(eq):
    '''
    if polynomial of C appear in eq, reduce to C for computational efficiency.
    '''
    eq = simplify_eq(eq)
    if 'C**' in eq:
        c_poly = ['C**' + str(i) for i in range(10)]
        for c in c_poly:
            if c in eq:
                eq = eq.replace(c, 'C')
    return simplify_eq(eq)


def score_with_est(eq, tree_size, data, t_limit=1.0, eta=0.999):
    """
    该函数计算一个完整解析树的奖励分数。
    如果方程中包含占位符C，也会为C执行估计。
    奖励 = 1 / (1 + MSE) * Penalty ** num_term

    这是主函数的开始，它接受五个参数：eq（一个字符串，表示解析树生成的方程式），tree_size（整数，解析树的大小），
    data（二维 numpy 数组，表示用于评分的数据），t_limit（表示计算评分的时间限制）和 eta（一个用于计算惩罚因子的超参数）。

    参数:
    eq : 字符串对象，已发现的方程（包含占位符C的系数）。
    tree_size : 整数对象，完整解析树中的产生规则数。
    data : 二维numpy数组，测量数据，包括独立变量和因变量（最后一行）。
    t_limit : 浮点数对象，单次评估的时间限制（秒），默认为1秒。

    返回值:
    score: 浮点数，已发现的方程的奖励分数。
    eq: 字符串，包含估计数值的已发现方程。
    """

    ## 定义独立变量和因变量
    num_var = data.shape[0] - 1
    if num_var <= 3:  ## 大部分情况 ([x], [x,y], 或 [x,y,z])
        current_var = 'x'
        for i in range(num_var):
            globals()[current_var] = data[i, :]
            current_var = chr(ord(current_var) + 1)
        globals()['f_true'] = data[-1, :]
    else:  ## 目前只有双摆案例有超过3个独立变量
        globals()['x1'] = data[0, :]
        globals()['x2'] = data[1, :]
        globals()['w1'] = data[2, :]
        globals()['w2'] = data[3, :]
        globals()['wdot'] = data[4, :]
        globals()['f_true'] = data[5, :]

    ## 计算eq中数值的数量
    c_count = eq.count('C')
    with time_limit(t_limit, 'sleep'):
        try:
            if c_count == 0:  ## 无数值
                f_pred = eval(eq)
            elif c_count >= 10:  ## 不鼓励过于复杂的数值估计
                return 0, eq
            else:  ## 有数值：使用Powell方法进行系数估计

                c_lst = ['c' + str(i) for i in range(c_count)]
                for c in c_lst:
                    eq = eq.replace('C', c, 1)

                def eq_test(c):
                    for i in range(len(c)):
                        globals()['c' + str(i)] = c[i]
                    return np.linalg.norm(eval(eq) - f_true, 2)

                x0 = [1.0] * len(c_lst)
                c_lst = minimize(eq_test, x0, method='Powell', tol=1e-6).x.tolist()
                c_lst = [np.round(x, 4) if abs(x) > 1e-2 else 0 for x in c_lst]
                eq_est = eq
                for i in range(len(c_lst)):
                    eq_est = eq_est.replace('c' + str(i), str(c_lst[i]), 1)
                eq = eq_est.replace('+-', '-')
                f_pred = eval(eq)
        except:
            return 0, eq

    r = float(eta ** tree_size / (1.0 + np.linalg.norm(f_pred - f_true, 2) ** 2 / f_true.shape[0]))

    return r, eq
