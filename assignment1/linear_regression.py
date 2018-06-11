# -*- coding: utf-8 -*-
import numpy as np

class LinearRegression(object):
    
    def __init__(self, lr):
        np.random.seed(1) # 调试阶段请勿删去该条语句
        self.k = np.random.normal(0, 0.01)
        self.b = 0
        self.lr = lr # 学习率（learning rate）
        # 利用数组记录训练过程中的有价值数据
        self.trainloss = []  
        self.testloss = []
        self.snapshot = []
        
        
    def predict(self, X):
        # 任务1
        # 该方法返回模型的预测值
        return self.k * X + self.b

        
        
        
    def get_loss(self, X, y):
        # 任务2
        # 先调用 predict 方法计算模型的预测值
        # 再计算均方误差
        return np.mean((self.predict(X) - y)**2) / 2
#        L = 0
#        for i in range(len(y)):
#            err = (self.predict(X[i]) - y[i]) ** 2
#            L += err
#        return L/2/len(y)
        
        
        
    def get_grad(self, X, y):
        # pass
        # 任务3
        # 计算 k 和 b 的梯度 dk, db
        # return dk, db
        dk = np.mean((self.predict(X) - y) * X)
        db = np.mean((self.predict(X) - y))
        return dk, db
        
        
        
    def check_grad(self, X, y):
        eps = 1e-4
        dk, db = self.get_grad(X, y)
        
        Jk_p = (((self.k + eps) * X + self.b - y) ** 2).mean() / 2
        Jk_n = (((self.k - eps) * X + self.b - y) ** 2).mean() / 2
        dk_ = (Jk_p - Jk_n) / 2 / eps

        Jb_p = ((self.k * X + self.b + eps - y) ** 2).mean() / 2
        Jb_n = ((self.k * X + self.b - eps - y) ** 2).mean() / 2
        db_ = (Jb_p - Jb_n) / 2 / eps
        
        return np.sqrt((dk - dk_)**2 + (db - db_)**2)
        

    def update(self, X, y):
        # 任务4
        # 先调用 get_grad 方法计算 k 和 b 的梯度
        # 然后用梯度下降算法更新 k 和 b
#        pass
        dk, db = self.get_grad(X, y)
        self.k = self.k - self.lr * dk
        self.b = self.b - self.lr * db
        # 以下语句记录训练过程中的 trainloss 和模型参数的 “快照” 
        loss = self.get_loss(X, y)
        self.trainloss.append(loss)
        self.snapshot.append((self.k, self.b))
        
        
    def evaluate(self, X, y):
        # 建立时刻测试（评估）模型的习惯很重要。
        # 记住：我们训练模型的目的是让模型在测试数据上在某种指标衡量下最优，
        # 因此记录模型训练过程中的训练误差、测试误差等数据有助于我们诊断和调试模型。
        loss = self.get_loss(X, y)
        self.testloss.append(loss)
            

            



