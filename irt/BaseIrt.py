from __future__ import print_function, division

import warnings

import numpy as np


class BaseIrt(object):

    def __init__(self, scores=None):
        self.scores = scores

    @staticmethod
    def p(z):
        # 回答正确的概率函数
        e = np.exp(z)
        p = e / (1.0 + e)
        return p


class Irt2PL(BaseIrt):
    ''

    @staticmethod
    def z(slop, threshold, theta):
        # z函数
        _z = slop * theta + threshold
        _z[_z > 35] = 35
        _z[_z < -35] = -35
        return _z


class Irt2PL(BaseIrt):
    # EM算法求解
    def __init__(self, init_slop=None, init_threshold=None, max_iter=10000, tol=1e-5, gp_size=11,
                 m_step_method='newton', *args, **kwargs):
        """
        :param init_slop: 斜率初值
        :param init_threshold: 阈值初值
        :param max_iter: EM算法最大迭代次数
        :param tol: 精度
        :param gp_size: Gauss–Hermite积分点数
        """
        super(Irt2PL, self).__init__(*args, **kwargs)
        # 斜率初值
        if init_slop is not None:
            self._init_slop = init_slop
        else:
            self._init_slop = np.ones(self.scores.shape[1])
        # 阈值初值
        if init_threshold is not None:
            self._init_threshold = init_threshold
        else:
            self._init_threshold = np.zeros(self.scores.shape[1])
        self._max_iter = max_iter
        self._tol = tol
        self._m_step_method = '_{0}'.format(m_step_method)
        self.x_nodes, self.x_weights = self.get_gh_point(gp_size)


class Irt2PL(BaseIrt):
    ''

    @staticmethod
    def get_gh_point(gp_size):
        x_nodes, x_weights = np.polynomial.hermite.hermgauss(gp_size)
        x_nodes = x_nodes * 2 ** 0.5
        x_nodes.shape = x_nodes.shape[0], 1
        x_weights = x_weights / np.pi ** 0.5
        x_weights.shape = x_weights.shape[0], 1
        return x_nodes, x_weights


class BaseIrt(object):
    ''

    def _lik(self, p_val):
        # 似然函数
        scores = self.scores
        loglik_val = np.dot(np.log(p_val + 1e-200), scores.transpose()) + \
                     np.dot(np.log(1 - p_val + 1e-200), (1 - scores).transpose())
        return np.exp(loglik_val)

    def _e_step(self, p_val, weights):
        # EM算法E步
        # 计算theta的分布人数
        scores = self.scores
        lik_wt = self._lik(p_val) * weights
        # 归一化
        lik_wt_sum = np.sum(lik_wt, axis=0)
        _temp = lik_wt / lik_wt_sum
        # theta的人数分布
        full_dis = np.sum(_temp, axis=1)
        # theta下回答正确的人数分布
        right_dis = np.dot(_temp, scores)
        full_dis.shape = full_dis.shape[0], 1
        # 对数似然值
        print(np.sum(np.log(lik_wt_sum)))
        return full_dis, right_dis


class Irt2PL(BaseIrt):
    ''
    def _irls(self, p_val, full_dis, right_dis, slop, threshold, theta):
        # 所有题目误差列表
        e_list = (right_dis - full_dis * p_val) / full_dis * (p_val * (1 - p_val))
        # 所有题目权重列表
        _w_list = full_dis * p_val * (1 - p_val)
        # z函数列表
        z_list = self.z(slop, threshold, theta)
        # 加上了阈值哑变量的数据
        x_list = np.vstack((threshold, slop))
        # 精度
        delta_list = np.zeros((len(slop), 2))
        for i in range(len(slop)):
            e = e_list[:, i]
            _w = _w_list[:, i]
            w = np.diag(_w ** 0.5)
            wa = np.dot(w, np.hstack((np.ones((self.x_nodes.shape[0], 1)), theta)))
            temp1 = np.dot(wa.transpose(), w)
            temp2 = np.linalg.inv(np.dot(wa.transpose(), wa))
            x0_temp = np.dot(np.dot(temp2, temp1), (z_list[:, i] + e))
            delta_list[i] = x_list[:, i] - x0_temp
            slop[i], threshold[i] = x0_temp[1], x0_temp[0]
        return slop, threshold, delta_list


# class Irt2PL(BaseIrt):
#     ''
#    def _est_item_parameter(self, slop, threshold, theta, p_val):
#         full_dis, right_dis = self._e_step(p_val, self.x_weights)
#         return self._m_step(p_val, full_dis, right_dis, slop, threshold, theta)
#
#    def _m_step(self, p_val, full_dis, right_dis, slop, threshold, theta):
#         # EM算法M步
#         m_step_method = getattr(self, self._m_step_method)
#         return m_step_method(p_val, full_dis, right_dis, slop, threshold, theta)
#
#     def em(self):
#         max_iter = self._max_iter
#         tol = self._tol
#         slop = self._init_slop
#         threshold = self._init_threshold
#         for i in range(max_iter):
#             z = self.z(slop, threshold, self.x_nodes)
#             p_val = self.p(z)
#             slop, threshold, delta_list = self._est_item_parameter(slop, threshold, self.x_nodes, p_val)
#             if np.max(np.abs(delta_list)) < tol:
#                 print(i)
#                 return slop, threshold
#         warnings.warn("no convergence")
#         return slop, threshold