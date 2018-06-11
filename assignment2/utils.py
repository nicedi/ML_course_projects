# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    return 1 / (np.exp(-Z) + 1)


def kernel(x, center, width):
    return np.exp(- (x-center)**2 / 2 / width**2)


def expand_feature1(x, mu=None, std=None):
    # 因为只在训练集上计算数据的均值和方差
    # 所以对测试数据调用该函数时，mu 和 std 不能为空，要传入之前在训练集上计算的值
    if mu is None:
        mu = x.mean()
    if std is None:
        std = x.std()
    x1 = (x - mu)/std
    # 因为 sigmoid 函数在自变量过大或过小时会饱和，所以这里先对 x1 归一化再计算 x2
    # 同时这里也没有对 x2 归一化，因为 sigmoid 函数的计算结果在[0, 1]区间，归一化不是十分必要
    x2 = sigmoid(x1)
    return mu, std, np.hstack((np.ones((x.shape[0],1)), x1, x2))


def expand_feature2(x, mu=None, std=None):
    # 任务2：添加一个全'1'的列和根据先验知识（或假设）做特征工程后的列
    # 灵活地根据数据的取值范围决定是否进行归一化
    pass

    
def expand_feature3(X, C, W):
    return [kernel(X, c, W) for c in C]

    
def plot_loss(model, n_iter):
    plt.figure()
    plt.semilogy(model.trainloss, 'b-', model.validloss, 'r-')
    plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training loss', 'validation loss'])
    plt.show()
    