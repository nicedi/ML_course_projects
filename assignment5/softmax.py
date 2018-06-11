# -*- coding: utf-8 -*-
import numpy as np

class Softmax(object):
    
    def __init__(self):
        pass
        
        
    def forward(self, X):
        shiftX = X - np.max(X, axis=1).reshape((X.shape[0], -1)) # for numerical stability
        # 任务2。利用上文的shiftX计算softmax的输出p
        pass
        
        return p
    
    
    def backward(self, err_in):
        return err_in

