# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt

class CF(object):
    def __init__(self, n_user, n_item, n_factor, lambd):
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.lambd = lambd
        # 初始化用户偏好矩阵和物品(电影)特征矩阵
        self.U = np.random.normal(0, 0.01, (n_user, n_factor))
        self.I = np.random.normal(0, 0.01, (n_factor, n_item))
        
        self.trainloss = []
        self.testloss = []
        self.snapshot = []
        
        
    def predict(self):
        # 任务1a：根据self.U 和 self.I 计算模型预测的评分
        pass
    
    
    def mse(self, Y, W):
        # Y is rating matrix
        # W is weight(or mask) matrix
        # 计算预测值和实际值的均方误差
        return np.sum(((self.predict() - Y) * W)**2) / W.sum()
    
        
    def update(self, Y, W):
        # Alternating Least Square
        for u, Wu in enumerate(W):
            # 更新self.U的每一行，即每个用户的特征
            self.U[u] = np.linalg.solve(np.dot(self.I, np.dot(np.diag(Wu), self.I.T)) + self.lambd * np.eye(self.n_factor),\
                                        np.dot(self.I, np.dot(np.diag(Wu), Y[u])))

        for i, Wi in enumerate(W.T):
            # 任务1b：根据教学内容和上面对self.U的更新，更新self.I的每一列
            pass
        
        prediction_error = self.mse(Y, W)
        self.trainloss.append(prediction_error)
        self.snapshot.append((self.U.copy(), self.I.copy()))
        print('training error:%.4f' % (prediction_error))
        
    
    def evaluate(self, Y, W):
        prediction_error = self.mse(Y, W)
        self.testloss.append(prediction_error)
        print('testing error:%.4f' % (prediction_error))
        
