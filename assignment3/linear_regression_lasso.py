# -*- coding: utf-8 -*-
import numpy as np

def soft_threshold(x, lambd):
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)


class LinearRegressionLasso(object):
    
    def __init__(self, n_feature, lambd):
        self.n_feature = n_feature
        self.W = np.random.normal(0, 0.01, (self.n_feature, 1))
        self.lambd = lambd
        self.trainloss = []
        self.validloss = []
        self.snapshot = []
        
        
    def predict(self, X):
        return np.matmul(X, self.W)
        
        
    def get_loss(self, X, y):
        ys = self.predict(X)
        error = ys - y
        # MSE
        mse = (error ** 2).mean() / 2
        # l1 norm penalty
        p = self.lambd * np.sum(self.W[1:])
        return mse + p
        
        
    def update_CD(self, X, y, j):
        # coordinate descent. update j-th parameter
        # 任务2：调用 soft_threshold 函数更新第 j 个参数。
        # 注意：θ^* 的计算方法不变；θ_0 不用调用 soft_threshold 更新。
        pass
              
        loss = self.get_loss(X, y)
        self.trainloss.append(loss)
        self.snapshot.append(self.W.copy())
        
        
    def evaluate(self, X, y):
        loss = self.get_loss(X, y)
        self.validloss.append(loss)
            

            



