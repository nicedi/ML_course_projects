# -*- coding: utf-8 -*-
import numpy as np

class KMeans(object):
    # K-Means Algorithm:
    #   Initialization
    #   Assignment
    #   Update

    def __init__(self, k, tol=1e-4, max_iter=500):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.snapshot = []
        
        
    def fit(self, data):
        n_sample = data.shape[0]
        
        # randomly choose centroids
        idx = np.random.choice(n_sample, self.k)
        self.centroids = data[idx]
        self.tags = np.random.choice(self.k, n_sample)
        
        for i in range(self.max_iter):

            # assignment tags
            # 任务1a：调用 self.predict 方法为每个样本指派标记，各标记保存在 self.tags 中 
            pass            
                
                
            old_centroids = self.centroids.copy()
            
            # log process
            self.snapshot.append((self.centroids.copy(), self.tags.copy()))
            
            # update centroids
            # 任务1b：更新 self.centroids
            pass
                
        
            # check convergence
            adjustment = np.mean((old_centroids - self.centroids) ** 2)
            print('iteration:{0}, adjustment:{1:.6f}'.format(i, adjustment))
            if adjustment < self.tol:
                return

            
    def predict(self, data):
        distances = [np.linalg.norm(data-centroid, ord=2) for centroid in self.centroids]
        return np.argmin(distances)
