# -*- coding: utf-8 -*-
import numpy as np


class LinearRegression(object):
    
    def __init__(self, n_feature, lr):
        self.n_feature = n_feature
        # W 代表模型的参数，其初始值是从均值为0，标准差为0.01的正态分布里随机取得
        self.W = np.random.normal(0, 0.01, (self.n_feature, 1))
        self.lr = lr
        self.trainloss = []
        self.validloss = []
        self.snapshot = []
        
        
    def predict(self, X):
        return np.matmul(X, self.W)
        
        
    def get_loss(self, X, y):
        ys = self.predict(X)
        error = ys - y
        # MSE
        loss = (error ** 2).mean() / 2
        return loss
        
        
    def get_grad(self, X, y):
        # 任务1：返回均方误差代价函数的梯度
        # 用向量化方式实现
        pass
        
        
    def check_grad(self, X, y):
        # 阅读下列代码，理解梯度数值检验的方法
        grad = self.get_grad(X, y)
        numeric_grad = np.zeros_like(grad)
        c = 1e-4
        origin_W = self.W.copy()
        for i in range(numeric_grad.size):
            eps = np.zeros(numeric_grad.shape)
            eps[i] = c
            self.W = origin_W + eps
            Jp = self.get_loss(X, y)
            
            self.W = origin_W - eps
            Jn = self.get_loss(X, y)
            numeric_grad[i] = (Jp - Jn) / 2 / c
        return np.sqrt(((grad - numeric_grad) ** 2).sum())
        

    def update(self, X, y):
        grad = self.get_grad(X, y)
        self.W = self.W - self.lr * grad
        loss = self.get_loss(X, y)
        self.trainloss.append(loss)
        self.snapshot.append(self.W.copy())
        
        
    def evaluate(self, X, y):
        loss = self.get_loss(X, y)
        self.validloss.append(loss)
            
            



