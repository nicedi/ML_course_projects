# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import utils

#%% 设定随机数发生器种子，确保我们在同一个“频道”。
# 在下面所有任务完成前请勿更改 seed 方法的参数
np.random.seed(1)
    

#%% 载入数据集
data = np.genfromtxt('dataset.csv', delimiter=',')
# data = data[data[:,1] == 1] # male
# data = data[data[:,1] == 2] # female

stature = data[:,2]
handsize = data[:,3]
footsize = data[:,4]



#%% 手的大小与身高的关系
plt.figure()
plt.scatter(handsize, stature)
plt.xlabel('hand size (mm)')
plt.ylabel('stature (mm)')
plt.show()


#%% 脚的大小与身高的关系
plt.figure()
plt.scatter(footsize, stature)
plt.xlabel('foot size (mm)')
plt.ylabel('stature (mm)')
plt.show()


#%% 手与脚尺寸的相关性
plt.figure()
plt.scatter(handsize, footsize)
plt.xlabel('hand size (mm)')
plt.ylabel('foot size (mm)')
plt.show()


#%% 3D视图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(handsize, footsize, stature)
ax.set_xlabel('handsize (mm)')
ax.set_ylabel('footsize (mm)')
ax.set_zlabel('stature (mm)')
plt.show()


#%% 定义常量
n_iter = 50  # 迭代次数
lr = 0.4     # 学习率 learning rate


#%% 数据归一化
handsize = (handsize - handsize.mean()) / handsize.std()


#%% 任务1. 实现 LinearRegression 的 predict 方法
# 创建模型
model = LinearRegression(lr)

# 检查实现结果，如果出现断言错误（AssertionError），则需要查找实现中的问题。
# 如果没有错误提示，则进行下一步。
temp = np.mean((model.predict(handsize) - stature)**2)
np.testing.assert_almost_equal(temp, 2822820, decimal=0)


#%% 任务2. 实现 LinearRegression 的 get_loss 方法
# 创建模型
model = LinearRegression(lr)

# 检查实现结果，如果出现断言错误（AssertionError），则需要查找实现中的问题。
# 如果没有错误提示，则进行下一步。
temp = model.get_loss(handsize, stature)
np.testing.assert_almost_equal(temp, 1411410, decimal=0)



#%% 观察代价函数在参数空间的样子
# contour graph
l = 100
k = np.linspace(50, 100, l)
b = np.linspace(1650, 1700, l)
K, B = np.meshgrid(k, b)

J = np.zeros((l,l))
for i in range(l):
    for j in range(l):
        model.k, model.b = k[i], b[j]
        J[i, j] = model.get_loss(handsize, stature)
plt.figure()
plt.contour(K, B, J, 20)
plt.xlabel('k')
plt.ylabel('b')
plt.show()


# surface graph
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(K, B, J, cmap=plt.cm.coolwarm,linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('k')
plt.ylabel('b')
# plt.zlabel('L')
plt.show()



#%% 划分数据集
np.random.seed(1)
np.random.shuffle(data)
X = data[:,3]
y = data[:,2]
n_sample = X.shape[0]

trainX = X[: int(n_sample * 0.8)]
trainy = y[: int(n_sample * 0.8)]
testX = X[int(n_sample * 0.8) :]           
testy = y[int(n_sample * 0.8) :]


#%% 特征归一化 (feature normalization)      
mu = trainX.mean()
std = trainX.std()
trainX = (trainX - mu) / std
testX = (testX - mu) / std


#%% 任务3. 实现 LinearRegression 的 get_grad 方法
# 通过观察下列语句的输出值检查梯度实现的正误
# 梯度检查的输出值应非常小，如果不是请检查梯度计算的代码
print(model.check_grad(trainX, trainy))



#%% 任务4. 实现 LinearRegression 的 update 方法
#%% 实现了 update 方法（梯度下降算法）后再进行以下迭代

model = LinearRegression(lr)
for i in range(n_iter):
    model.update(trainX, trainy)
    model.evaluate(testX, testy)
    # 分别解除下列语句的注释，从不同角度观察模型的优化过程
    utils.plot_loss(model, n_iter)
    #utils.show_data_model(model, trainX, trainy)
    #utils.show_contour_model(model, trainX, trainy)
    
    

#%% 任务5. 输入你手的尺寸，观察模型预测的身高值
best_idx = np.argmin(model.testloss)
model.k, model.b = model.snapshot[best_idx]
my_size = float(input('输入你手的尺寸（以毫米为单位）：'))
print("预测你的身高为: %.2f mm" % 
      (model.predict((my_size - mu)/std)))


#%% 理解 supplementary 文件夹中关于 KNN 和 bayesian regression 两个算法