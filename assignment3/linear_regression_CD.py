# -*- coding: utf-8 -*-
import numpy as np


class LinearRegressionCD(object):
    
    def __init__(self, n_feature):
        self.n_feature = n_feature
        self.W = np.random.normal(0, 0.01, (self.n_feature, 1))
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
        
        
    def update_CD(self, X, y, j):
        # 任务1
        # coordinate descent. update j-th parameter
        pass
              
        loss = self.get_loss(X, y)
        self.trainloss.append(loss)
        self.snapshot.append(self.W.copy())
        
        
    def evaluate(self, X, y):
        loss = self.get_loss(X, y)
        self.validloss.append(loss)
            

            



