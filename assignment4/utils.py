# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(model, n_iter):
    plt.figure()
    plt.plot(model.trainloss, 'b-', model.validloss, 'r-')
    plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training loss', 'validation loss'])
    plt.show()
    
    
def plot_F1(model, n_iter):
    plt.figure()
    plt.plot(model.trainF1, 'b-', model.validF1, 'r-')
    plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('F1 score')
    plt.title('F1 metric curve')
    plt.legend(['training F1', 'validation F1'], loc='lower right')
    plt.show()
    
    
def confusion_matrix(threshold, y_hat, y_target):
    # 任务2：实现该函数。函数应返回 TP, FP, FN, TN 四个值。
    # y_hat = (y_hat > threshold).astype(np.int32) # 高于阈值的预测值置为1，反之为0
    # 提示：对比 y_hat 和 y_target 中的值计算 True Positive，False Positive 等
    tmp = np.hstack((y_target, y_hat > threshold))
    pass
    
    # return TP, FP, FN, TN
    