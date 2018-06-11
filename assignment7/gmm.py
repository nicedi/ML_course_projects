# -*- coding: utf-8 -*-
import numpy as np
import utils
# from scipy.stats import multivariate_normal

class GMM(object):
    
    def __init__(self, k, tol=1e-5, max_iter=100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.snapshot = []
        
        
    def fit(self, data):
        n_sample, n_feature = data.shape
        
        # initialization: assign hard label
        hardlabels = np.random.choice(self.k, n_sample)
        self.mu = np.empty((self.k, n_feature))
        self.sigma = np.empty((self.k, n_feature, n_feature))
        self.phi = np.empty(self.k)
        
        for i in range(self.k):
            mask = (hardlabels==i)
            self.mu[i] = data[mask].mean(axis=0)
            self.sigma[i] = np.matmul(np.transpose(data[mask]-self.mu[i]), data[mask]-self.mu[i]) / mask.sum()
            self.phi[i] = mask.sum() / n_sample
               
        self.W = np.ones((n_sample, self.k)) / self.k
        
        for i in range(self.max_iter):
            old_W = self.W.copy()
            
            # E-step
            self.W = self.predict(data)
            
            # M-step
            self.phi = self.W.mean(axis=0)
            for j in range(self.k):
                self.mu[j] = np.sum(data * np.tile(self.W[:,j][:,np.newaxis], (1, n_feature)), axis=0) / np.sum(self.W[:,j])     
                self.sigma[j] = np.matmul(np.transpose(data-self.mu[j]), (data-self.mu[j])*np.tile(self.W[:,j][:,np.newaxis], (1, n_feature))) / np.sum(self.W[:,j])    
            
            # save snapshot
            self.snapshot.append((self.phi.copy(), self.mu.copy(), self.sigma.copy()))
            
            # check conversion
            adjustment = np.mean((old_W - self.W) ** 2)
            print('iteration:{0}, adjustment:{1:.6f}'.format(i, adjustment))
            if adjustment < self.tol:
                return
            
            
    def predict(self, data):
        W = np.empty((data.shape[0], self.k))
        # 任务2：计算 W (soft-label)
        pass
    
        return W
        
