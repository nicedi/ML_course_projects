# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfnorm


#%% useful functions
def predict(dp, X, Y, K, weighting=False):
    err = X - dp
    abs_err = np.abs(err)
    idx = np.argsort(abs_err)
    
    candidates = Y[idx[:K]]
    
    if weighting:
        weights = halfnorm.pdf(candidates, scale=np.std(Y))
        weights /= np.sum(weights)
        return np.dot(candidates, weights)
    else:
        return np.mean(candidates)



#%% 载入数据集
data = np.genfromtxt('dataset.csv', delimiter=',')
stature = data[:,2]
handsize = data[:,3]
footsize = data[:,4]


#%% 定义常量
K = 5


#%% predict
prediction = np.empty_like(stature)
for i in range(len(handsize)):
    prediction[i] = predict(handsize[i], handsize, stature, K, weighting=False)
    
    
#%% visualize prediction
plt.figure()
plt.scatter(handsize, stature, marker='+')
plt.scatter(handsize, prediction, marker='o', alpha=0.3)
plt.xlabel('hand size (mm)')
plt.ylabel('stature (mm)')
plt.show()


#%% predict
my_size = 190
print('my stature: ', predict(my_size, handsize, stature, K, weighting=False))

#%% 观察 KNN 的预测值
x = np.linspace(130, 260, 200)
y = np.empty_like(x)
for i in range(len(x)):
    y[i] = predict(x[i], handsize, stature, K, weighting=True)
    
plt.figure()
plt.plot(x,y);plt.show()