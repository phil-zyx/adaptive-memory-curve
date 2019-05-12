from __future__ import print_function, division
import numpy as np
import progressbar


def _log_lognormal(param):
    # 对数正态分布的概率密度分布的对数
    return np.log(1.0 / param) + _log_normal(np.log(param))


def _log_normal(param):
    # 正态分布的概率密度分布的对数
    return param ** 2 * -0.5


def _param_den(slop, threshold, guess):
    # 项目参数联合概率密度
    return _log_normal(threshold) + _log_lognormal(slop) + 4 * np.log(guess) + 16 * np.log(1 - guess)


def logistic(slop, threshold, guess, theta):
    # logistic函数
    return guess + (1 - guess) / (1.0 + np.exp(-1 * (slop * theta + threshold)))


def loglik(slop, threshold, guess, theta, scores, axis=1):
    # 对数似然函数
    p = logistic(slop, threshold, guess, theta)
    p[p <= 0] = 1e-10
    p[p >= 1] = 1 - 1e-10
    return np.sum(scores * np.log(p) + (1 - scores) * np.log(1 - p), axis=axis)


def _tran_theta(slop, threshold, guess, theta, next_theta, scores):
    # 特质的转移函数
    pi = (loglik(slop, threshold, guess, next_theta, scores) + _log_normal(next_theta)[:, 0]) - (
            loglik(slop, threshold, guess, theta, scores) + _log_normal(theta)[:, 0])
    pi = np.exp(pi)
    # 下步可省略
    pi[pi > 1] = 1
    return pi


def _tran_item_para(slop, threshold, guess, next_slop, next_threshold, next_guess, theta, scores):
    # 项目参数的转移函数
    nxt = loglik(next_slop, next_threshold, next_guess, theta, scores, 0) + _param_den(next_slop, next_threshold,
                                                                                       next_guess)
    now = loglik(slop, threshold, guess, theta, scores, 0) + _param_den(slop, threshold, guess)
    pi = nxt - now
    pi.shape = pi.shape[1]
    pi = np.exp(pi)
    # 下步可省略
    pi[pi > 1] = 1
    return pi


def mcmc(chain_size, scores):
    # 样本量
    person_size = scores.shape[0]
    # 项目量
    item_size = scores.shape[1]
    # 潜在特质初值
    theta = np.zeros((person_size, 1))
    # 斜率初值
    slop = np.ones((1, item_size))
    # 阈值初值
    threshold = np.zeros((1, item_size))
    # 猜测参数初值
    guess = np.zeros((1, item_size)) + 0.1
    # 参数储存记录
    theta_list = np.zeros((chain_size, len(theta)))
    slop_list = np.zeros((chain_size, item_size))
    threshold_list = np.zeros((chain_size, item_size))
    guess_list = np.zeros((chain_size, item_size))
    bar = progressbar.ProgressBar()
    for i in bar(range(chain_size)):
        next_theta = np.random.normal(theta, 1)
        theta_pi = _tran_theta(slop, threshold, guess, theta, next_theta, scores)
        theta_r = np.random.uniform(0, 1, len(theta))
        theta[theta_r <= theta_pi] = next_theta[theta_r <= theta_pi]
        theta_list[i] = theta[:, 0]
        next_slop = np.random.normal(slop, 0.3)
        # 防止数值溢出
        next_slop[next_slop < 0] = 1e-10
        next_threshold = np.random.normal(threshold, 0.3)
        next_guess = np.random.uniform(guess - 0.03, guess + 0.03)
        # 防止数值溢出
        next_guess[next_guess <= 0] = 1e-10
        next_guess[next_guess >= 1] = 1 - 1e-10
        param_pi = _tran_item_para(slop, threshold, guess, next_slop, next_threshold, next_guess, theta, scores)
        param_r = np.random.uniform(0, 1, item_size)
        slop[0][param_r <= param_pi] = next_slop[0][param_r <= param_pi]
        threshold[0][param_r <= param_pi] = next_threshold[0][param_r <= param_pi]
        guess[0][param_r <= param_pi] = next_guess[0][param_r <= param_pi]
        slop_list[i] = slop[0]
        threshold_list[i] = threshold[0]
        guess_list[i] = guess[0]
    return theta_list, slop_list, threshold_list, guess_list


# 样本量和题量
PERSON_SIZE = 1000
ITEM_SIZE = 60
# 模拟参数
a = np.random.lognormal(0, 1, (1, ITEM_SIZE))
a[a > 4] = 4
b = np.random.normal(0, 1, (1, ITEM_SIZE))
b[b > 4] = 4
b[b < -4] = -4
c = np.random.beta(5, 17, (1, ITEM_SIZE))
c[c < 0] = 0
c[c > 0.2] = 0.2
true_theta = np.random.normal(0, 1, (PERSON_SIZE, 1))
p_val = logistic(a, b, c, true_theta)
scores = np.random.binomial(1, p_val)
# MCMC参数估计
thetas, slops, thresholds, guesses = mcmc(7000, scores=scores)
est_theta = np.mean(thetas[3000:], axis=0)
est_slop = np.mean(slops[3000:], axis=0)
est_threshold = np.mean(thresholds[3000:], axis=0)
est_guess = np.mean(guesses[3000:], axis=0)
# 打印误差
print(np.mean(np.abs(est_slop - a[0])))
print(np.mean(np.abs(est_threshold - b[0])))
print(np.mean(np.abs(est_guess - c[:, 0])))
print(np.mean(np.abs(est_theta - true_theta[:, 0])))
