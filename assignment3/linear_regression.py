# -*- coding: utf-8 -*-
import numpy as np

def soft_thresholding(x, lambd):
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)

class LinearRegression(object):
    
    def __init__(self, n_feature, lr):
        self.n_feature = n_feature
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
        error = self.predict(X) - y
        # MSE
        grad = np.dot(np.transpose(X) , error) / X.shape[0]
        return grad
        
        
    def check_grad(self, X, y):
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
        # gradient descent
        grad = self.get_grad(X, y)
        self.W = self.W - self.lr * grad
        
        loss = self.get_loss(X, y)
        self.trainloss.append(loss)
        self.snapshot.append(self.W)
        
        
    def update_CD(self, X, y, j):
        # coordinate gradient descent. update j-th parameter
        Xj = X[:,j].reshape((1, -1))
        mask = np.ones_like(self.W)
        mask[j, 0] = 0
        R = y - np.matmul(X, self.W * mask)
        self.W[j] = np.matmul(Xj, R) / y.shape[0]
              
        loss = self.get_loss(X, y)
        self.trainloss.append(loss)
        self.snapshot.append(self.W.copy())
        
        
    def evaluate(self, X, y):
        loss = self.get_loss(X, y)
        self.validloss.append(loss)
            

            



