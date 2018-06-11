# -*- coding: utf-8 -*-
import numpy as np
from logistic_regression import LogisticRegression
import matplotlib.pyplot as plt
import utils


#%% 载入数据
data = np.genfromtxt('myopia.csv', delimiter=',', skip_header=1)


#%% 定义常量
n_sample = data.shape[0]
n_iter = 800
lr = 0.1


#%% 划分数据集
np.random.seed(1)
np.random.shuffle(data)
X = data[:,3:]
y = data[:,2].reshape((n_sample, 1)) 

trainX = X[: int(n_sample * 0.6)]
trainy = y[: int(n_sample * 0.6)]
validX = X[int(n_sample * 0.6) : int(n_sample * 0.8)]           
validy = y[int(n_sample * 0.6) : int(n_sample * 0.8)]
testX = X[int(n_sample * 0.8) :]           
testy = y[int(n_sample * 0.8) :]
         
# 特征归一化
mu = np.mean(trainX, axis=0)
sigma = np.std(trainX, axis=0)
trainX = (trainX - mu) / sigma
validX = (validX - mu) / sigma
testX = (testX - mu) / sigma

# add '1' column
trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
validX = np.hstack((np.ones((validX.shape[0], 1)), validX))
testX = np.hstack((np.ones((testX.shape[0], 1)), testX))


#%% 可视化数据
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

tsne = TSNE(n_components=3, init='pca', random_state=1)
Xtsne = tsne.fit_transform(trainX)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['b', 'r']
markers = ['o', '+']
for i in range(trainX.shape[0]):
    color = colors[ int(trainy[i,0]) ]
    mk = markers[ int(trainy[i,0]) ]
    plt.scatter(Xtsne[i,0], Xtsne[i,1], Xtsne[i,2], 
                c=color, marker=mk)
plt.show()



#%% 创建模型对象
n_feature = trainX.shape[1]
model = LogisticRegression(n_feature, lr)



#%% 梯度检验
# 任务1：实现 LogisticRegression 中的 get_grad 方法
print(model.check_grad(validX, validy))



#%% 训练
for i in range(n_iter):
    model.update(trainX, trainy)
    model.evaluate(validX, validy)

utils.plot_loss(model, n_iter)
utils.plot_F1(model, n_iter)



#%% 评估模型
# 任务2： 实现 utils 中的 confusion_matrix 函数

# 试着从 F1 score 的角度或验证误差的角度选择最佳模型
#idx = np.argmax(model.validF1)
idx = np.argmin(model.validloss)

model.W = model.snapshot[idx]
y_hat = model.predict(testX)
threshold = 0.5
TP, FP, FN, TN = utils.confusion_matrix(threshold, y_hat, testy)
accuracy = (TP + TN) / testy.shape[0]
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
print("准确率:{0}\n查准率:{1}\n查全率:{2}\nF1 值:{3}\n".
      format(accuracy, precision, recall, F1))


#%% 对照真实标记检查预测结果
y_hat = (y_hat >= threshold).astype(np.int32)
for item in np.hstack( (y_hat, testy) ):
    print("预测结果:{0}, 实际结果:{1}".format(item[0], item[1]))


    
#%% 改变负对数似然代价函数中两项的权重，再次实验
# 任务3：修改 LogisticRegression 中的 get_grad 方法，使其适配 beta 不为1的情形
# 创建模型
n_feature = trainX.shape[1]
model = LogisticRegression(n_feature, lr, 7)
# 重复上述训练和评估两个步骤




#%% 增加特征维数，再次实验
data = np.hstack((data, data[:, 6:]**2))
# 重复上述定义常量、划分数据集、创建模型对象(beta=7)、训练和评估几个步骤