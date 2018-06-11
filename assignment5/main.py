# -*- coding: utf-8 -*-
import numpy as np
from mnist_loader import load_mnist
import matplotlib.pyplot as plt
import utils

from nn import NeuralNetwork
from linear import Linear
from relu import Relu
from softmax import Softmax


#%% 载入 mnist 数据集
X, y = load_mnist(dataset="training", path="mnist")
testX, testy = load_mnist(dataset="testing", path="mnist")
print(X.shape)
print(y.shape)
print(testX.shape)
print(testy.shape)


#%% 显示 mnist 数据集中图像
utils.show_mnist(X, y, 5, 5)


#%% 定义常量
n_feature = 28 * 28
n_iter = 2
lr = 0.0001


#%% 处理数据集
#  unroll feature
X = X.reshape( (60000, n_feature) )
testX = testX.reshape( (10000, n_feature) )

# 划分数据集
# 因为 mnist 数据集中数据和标记是分开的，为了保证数据和标记以相同的方式置乱而采用以下方式。
# 大多数数据集都把数据和标记分开，所以采用这种置乱数据的方式很有必要。
# 请思考为什么不能用 np.random.shuffle 分别置乱 X 和 y ？
np.random.seed(1)
idxs = np.random.permutation(60000)

trainX = X[idxs[: 50000]]
trainy = y[idxs[: 50000]]
validX = X[idxs[50000 :]] # 分出10000条数据作为验证集     
validy = y[idxs[50000 :]]
         
          
# 特征归一化
mu = np.mean(trainX, axis=0)
std = np.std(trainX, axis=0)
trainX = (trainX - mu) / (std+np.finfo(np.float32).eps) # 加epsilon，避免出现除‘0’的错误
validX = (validX - mu) / (std+np.finfo(np.float32).eps)
testX = (testX - mu) / (std+np.finfo(np.float32).eps)


#%% 可视化 mnist
# https://colah.github.io/posts/2014-10-Visualizing-MNIST/


#%%  创建模型
model = NeuralNetwork()
# 将神经网络看成由若干完成特定计算的层组成，数据经过这些层完成前馈运算；
# 根据求导链式法则的启示，可以利用误差的反向传播计算代价函数对模型参数的偏导（即梯度）。

# 任务1：实现Relu类中的forward和backward方法
# 任务2：实现Softmax类中的forward方法

model.layers.append(Linear(n_feature, 60, lr))
model.layers.append(Relu())
model.layers.append(Linear(60, 10, lr))
model.layers.append(Softmax())



#%% 训练
# stochastic gradient descent
batchsize = 100
trainloss = []
validloss = []
snapshot = []

for i in range(n_iter):
    # 每一轮迭代前，产生一组新的序号（目的在于置乱数据）
    idxs = np.random.permutation(trainX.shape[0])
    
    for j in range(0, trainX.shape[0], batchsize):
        batchX = trainX[idxs[j : j+batchsize]]
        
        # 任务3：实现utils中 make_onehot 函数
        
        batchy = utils.make_onehot(trainy[idxs[j : j+batchsize]], 10)
        # 数据的前馈(feed forward)和误差的反向传播(back propagation)
        # 是人工神经网络中的两种数据流向，这里用 forward 和 backward 为下列
        # 两个方法命名是与其它大型机器学习框架对人工神经网络中有关方法的命名保持一致
        
        y_hat= model.forward(batchX)
        
        # 任务4：理解utils中cross_entropy的实现代码
        
        loss1 = utils.cross_entropy(y_hat, batchy)
        trainloss.append(loss1)
        error = y_hat - batchy
        model.backward(error)
        
        # 评估模型性能
        loss2 = utils.cross_entropy(model.forward(validX), 
                        utils.make_onehot(validy, 10))
        validloss.append(loss2)

        # 保存当前模型
        params = []
        for ly in model.layers:
            if isinstance(ly, Linear):
                params.append(ly.W.copy())
        snapshot.append(params)
        
    print("iteration:{0}/{1} trainloss:{2:.2f}, validloss:{3:.2f}".\
        format(j, i, loss1, loss2))

utils.plot_loss(trainloss, validloss)


#%% 评估
# 挑出验证集上表现最佳的模型
params_idx = np.argmin(validloss)
best_params = snapshot[params_idx]
for ly in model.layers:
    if isinstance(ly, Linear):
        ly.W = best_params.pop(0)

prediction = np.argmax(model.forward(testX), axis=1)
cmatrix = utils.confusion_matrix(prediction, testy.flatten(), 10)
print(cmatrix)
# accuracy
acc = np.diag(cmatrix).sum() / testy.shape[0]
print("accuracy: %.4f" % (acc))
# heat map
plt.figure()
plt.imshow(cmatrix, cmap='gray', interpolation='none')
plt.show()


#%% 观察错误分类的样本及模型输出
raw_testX = testX * std + mu
y_distribution = model.forward(testX)
diff = prediction != testy[:,0]

# 观察前 N 个错误分类
N = 10
err_idx = np.where(diff)[0]

plt.figure()
for i in range(N):
    im = raw_testX[err_idx[i], :].reshape((28, 28))
    plt.subplot(N, 2, i*2+1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Label: {0}, Prediction: {1}'.format(testy[err_idx[i],0], prediction[err_idx[i]]))
    plt.imshow(im, cmap='gray', interpolation='None')
    
    plt.subplot(N, 2, i*2+2)
    plt.bar(np.arange(10), y_distribution[err_idx[i], :])
    
plt.show()
