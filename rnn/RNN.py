# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Optional, Tuple
from tensorflow.contrib import rnn


def RNN(x, weights, biases, n_input, n_steps, n_hidden):
    # 准备数据形状以匹配RNN函数的要求
    # 当前数据输入形状: (batch_size, n_steps, n_input)
    # 要求形状: 'n_steps' 张量形状列表 (batch_size, n_input)
    # 置换 batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # 重定义形状 (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # 分割得到列表 'n_steps' 形状张量 (batch_size, n_input)
    x = tf.split(x, n_steps, axis=0)
    # 用张量流定义一个LSTM单元
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 获取LSTM信元输出
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # 线性激活函数，使用RNN内循环最后输出
    return tf.nn.bias_add(tf.matmul(outputs[-1], weights['out']), biases['out'])


def generate_sample(f: Optional[float] = 1.0, t0: Optional[float] = None, batch_size: int = 1,
                    predict: int = 50, samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成数据样本。
    :参数 f: 用于所有时间序列或没有随机化的频率。
    :参数 t0: 用于所有时间序列或没有随机化的时间偏移。
    :参数 batch_size: 生成的时间序列的数量。
    :参数 predict: 要生成的未来样本的数量。
    :参数 samples: 要生成的过去（和当前）样本的数量。
    :返回: 元组包含过去的时间和值以及未来的时间和值。在所有输出中，每行表示批的一个时间序列。
    """
    Fs = 100
