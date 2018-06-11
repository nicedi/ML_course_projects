# -*- coding: utf-8 -*-
import numpy as np

class Relu(object):
    
    def __init__(self):
        self.X = None
        
        
    def forward(self, X):
        # 任务1。根据该函数的表达式计算函数值，提示：利用 np.maximum 方法。
        pass
            
            
    def backward(self, err_in):
        # 任务1。计算函数输出对输入的导数dfdX
        pass
        
        err_out = err_in * dfdX
        return err_out
    
        
        