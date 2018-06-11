# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def gaussian_pdf(X, mu, sigma):
    # X 是一个 m by n 矩阵，第一个维度表示样本个数，第二个维度表示特征个数
    # 函数返回一个 m by 1 列向量
    _, n = X.shape
    return np.exp(- np.sum(np.matmul((X-mu), np.linalg.inv(sigma)) * (X-mu), axis=1) / 2) / np.linalg.det(sigma)**(0.5) / (2*np.pi)**(n/2)
    
    


def plot_loss(trainloss, testloss):
    plt.figure()
    plt.plot(trainloss, 'b-', testloss, 'r-')
    #plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training loss', 'test loss'])
    plt.show()



def weighted_radius(center, samples, weights):
    # 任务3：计算多个样本点到一个中心的平均加权距离
    pass