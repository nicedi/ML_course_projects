# -*- coding: utf-8 -*-
import numpy as np


class LinearRegressionRidge(object):
    
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
        # l2 norm penalty
        p = self.lambd * np.sum(self.W[1:]**2)
        return mse + p
        
        
    def update_CD(self, X, y, j):
        # coordinate descent. update j-th parameter
        # 任务3
        # 注意：θ_0 的更新方法与其它参数不同，因为它没有被正则化。
        pass
              
        loss = self.get_loss(X, y)
        self.trainloss.append(loss)
        self.snapshot.append(self.W.copy())
        
        
    def evaluate(self, X, y):
        loss = self.get_loss(X, y)
        self.validloss.append(loss)
            

            



