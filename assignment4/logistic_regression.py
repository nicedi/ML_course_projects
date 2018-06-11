# -*- coding: utf-8 -*-
import numpy as np
import utils

class LogisticRegression(object):
    
    def __init__(self, n_feature, lr, beta=1):
        self.n_feature = n_feature
        self.W = np.random.normal(0, 0.01, (self.n_feature, 1))
        self.lr = lr
        self.trainloss = []
        self.validloss = []
        self.snapshot = []
        self.trainF1 = []
        self.validF1 = []
        self.beta = beta # 用于调整 NLL 中两项的相对权重，默认值为 1


    def sigmoid(self, Z):
        return 1 / (np.exp(-Z) + 1)
        
        
    def predict(self, X):
        return self.sigmoid(np.matmul(X, self.W))
        
        
    def get_loss(self, X, y):
        ys = self.predict(X)
        # NLL
        loss = np.mean(- self.beta * y * np.log(ys) - (1 - y)*np.log(1 - ys))   
        return loss
        
        
    def get_grad(self, X, y):
        ys = self.predict(X)
        # 任务1：先计算预测值和实际值的误差，再计算梯度
        pass

        # return grad
        
        
    def check_grad(self, X, y):
        grad = self.get_grad(X, y)
        numeric_grad = np.zeros_like(grad)
        origin_W = self.W.copy()
        c = 1e-4
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
        # gradient descent
        self.W = self.W - self.lr * grad
        
        loss = self.get_loss(X, y)
        self.trainloss.append(loss)
        self.trainF1.append(self.measure(X, y)) # 记录 F1 score
        self.snapshot.append(self.W.copy())
        
        
    def evaluate(self, X, y):
        loss = self.get_loss(X, y)
        self.validloss.append(loss)
        self.validF1.append(self.measure(X, y))
        
        
    def measure(self, X, y, threshold=0.5):
        y_hat = self.predict(X)
        TP, FP, FN, TN = utils.confusion_matrix(threshold, y_hat, y)
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        return F1
            

            



