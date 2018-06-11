# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def dimrd_degree(s, threshold=0.99):
    # 当数据向特征向量投影时，投影的方差由特征值衡量
    # 参考：https://en.wikipedia.org/wiki/Rayleigh_quotient
    # 该函数返回整数K，使得保留K个维度（K个特征值）的数据刚好能够保留超过threshold的方差
    # 任务2：实现该函数。
    pass


def plot_loss(trainloss, testloss):
    plt.figure()
    plt.plot(trainloss, 'b-', testloss, 'r-')
    #plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training loss', 'test loss'])
    plt.show()


def euclidean_rank(feature, query):
    # Euclidean distance
    compares = np.sqrt(np.sum((feature - np.tile(query, (feature.shape[0], 1)))**2, axis=1))
    # 距离从小到大排序的结果
    rank = np.argsort(compares)
    return compares, rank


def cosine_rank(feature, query):
    # Adjusted Cosine Similarity
    mu = feature.mean(axis=0)
    feature -= mu
    query -= mu
    feature_norm = np.linalg.norm(feature, ord=2, axis=1) # 计算2范数
    query_norm = np.linalg.norm(query, ord=2)
    # 任务3：计算 query 和 feature 的余弦相似度。该相似度用 compares 变量命名。
    pass       
 
    # 相似度从大到小排序的结果。实现上述计算后去除下列语句的注释
    # rank = np.argsort(compares[:,0])[::-1]
    # return compares, rank