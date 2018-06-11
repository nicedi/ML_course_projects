# -*- coding: utf-8 -*-
import numpy as np

class Tanh(object):
    
    def __init__(self):
        pass
        
        
    def forward(self, X):
        A = np.exp(X)
        B = np.exp(-X)
        self.activation = (A - B) / (A + B)
        return self.activation
            
            
    def backward(self, err_in):
        return err_in * (1 - self.activation ** 2)
    
        
        