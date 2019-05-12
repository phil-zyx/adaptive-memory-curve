from irt.BaseIrt import BaseIrt


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