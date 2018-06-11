# -*- coding: utf-8 -*-
import numpy as np

class Relu(object):
    
    def __init__(self):
        pass
        
        
    def forward(self, X):
        self.X = X.copy()
        return np.maximum(self.X, 0)
            
            
    def backward(self, err_in):
        dfdX = np.ones_like(self.X)
        dfdX[self.X <= 0] = 0
        err_out = err_in * dfdX
        return err_out
    
        
        