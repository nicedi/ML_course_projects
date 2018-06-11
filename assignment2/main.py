# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import utils


#%% 载入数据
data = np.genfromtxt('dataset.csv', delimiter=',')


#%% 定义常量
n_feature = 3
n_sample = data.shape[0]
n_iter = 50
lr = 0.2 # learning rate


#%% 置乱然后划分数据
np.random.seed(1)
np.random.shuffle(data)
X = data[:,3].reshape((n_sample, 1))
y = data[:,2].reshape((n_sample, 1))

# dividing
trainX = X[: int(n_sample * 0.6)]
trainy = y[: int(n_sample * 0.6)]
validX = X[int(n_sample * 0.6) : int(n_sample * 0.8)]           
validy = y[int(n_sample * 0.6) : int(n_sample * 0.8)]
testX = X[int(n_sample * 0.8) :]           
testy = y[int(n_sample * 0.8) :]

# expand feature （特征工程）
mu, std, trainX = utils.expand_feature1(trainX)
_, _, validX = utils.expand_feature1(validX, mu, std)
_, _, testX = utils.expand_feature1(testX, mu, std)


#%%  任务1：实现 LinearRegression 的 get_grad 方法，向量化梯度的计算。
# 梯度检验。如果输出值较大，请检查梯度的计算过程。
model = LinearRegression(n_feature, lr)
print('梯度数值计算结果与实际值的 Error Sum of Squares(SSE): {0}'.format( 
      model.check_grad(trainX, trainy)))


#%% 训练模型
# 创建模型对象
model = LinearRegression(n_feature, lr)

# 用梯度下降方法进行迭代优化
for i in range(n_iter):
    model.update(trainX, trainy)
    model.evaluate(validX, validy)
    
# 通过观察 learning curve 检视学习过程
utils.plot_loss(model, n_iter)


#%% 查看训练结果
n = 100
x_origin = np.linspace(X.min(), X.max(), n)
_, _, xdata = utils.expand_feature(x_origin.reshape((n, 1)), mu, std)

ydata = model.predict(xdata)
plt.figure()
plt.plot(xdata[:,1], ydata, 'r-')
plt.scatter(trainX[:,1], trainy)
plt.xlabel('normalized hand-size')
plt.ylabel('stature')
plt.legend(['compound model', 'data points'], loc='lower right')
plt.show()


#%% 预测
best_idx = np.argmin(model.validloss)
model.W = model.snapshot[best_idx]
hand_size = float(input('输入你手的尺寸（以毫米为单位）：'))
hand_size = np.array([[hand_size]])
_, _, hand_size = utils.expand_feature(hand_size, mu, std)
print("预测的身高: %.2f mm" % (model.predict(hand_size)))


#%% 目标值和测试值对照检查
target_pred = np.hstack( (testy, model.predict(testX)) )
for i in range(testy.shape[0]):
    print('目标值:{0}, 预测值:{1}'.
          format(target_pred[i,0], target_pred[i,1]))


#%% 载入作业用数据
data = np.load('data.npy')


#%% #%% 定义常量
n_feature = 3
n_sample = data.shape[0]
n_iter = 50
lr = 0.2


#%% 置乱然后划分数据
np.random.seed(1)
np.random.shuffle(data)
X = data[:,0].reshape((n_sample, 1))
y = data[:,1].reshape((n_sample, 1))

# dividing
trainX = X[: int(n_sample * 0.6)]
trainy = y[: int(n_sample * 0.6)]
validX = X[int(n_sample * 0.6) : int(n_sample * 0.8)]           
validy = y[int(n_sample * 0.6) : int(n_sample * 0.8)]
testX = X[int(n_sample * 0.8) :]           
testy = y[int(n_sample * 0.8) :]


#%% 任务2：特征工程
# 实现 utils 中的 expand_feature2 函数。以便对作业数据进行建模
mu, std, trainX = utils.expand_feature2(trainX)
_, _, validX = utils.expand_feature2(validX, mu, std)
_, _, testX = utils.expand_feature2(testX, mu, std)


#%% 训练模型
# 创建模型对象
model = LinearRegression(n_feature, lr)

for i in range(n_iter):
    model.update(trainX, trainy)
    model.evaluate(validX, validy)
    
utils.plot_loss(model, n_iter)


#%% 查看训练结果
n = 100
x_origin = np.linspace(X.min(), X.max(), n)
_, _, xdata = utils.expand_feature2(x_origin.reshape((n, 1)), mu, std)

ydata = model.predict(xdata)
plt.figure()
plt.plot(xdata[:,1], ydata, 'r-')
plt.scatter(trainX[:,1], trainy)
plt.xlabel('normalized x')
plt.ylabel('y')
plt.legend(['compound model', 'data points'], loc='lower right')
plt.show()


#%% 用高斯核函数转换数据
# 划分数据
rawX = X[: int(n_sample * 0.6)]
trainy = y[: int(n_sample * 0.6)]
validX = X[int(n_sample * 0.6) : int(n_sample * 0.8)]           
validy = y[int(n_sample * 0.6) : int(n_sample * 0.8)]
testX = X[int(n_sample * 0.8) :]           
testy = y[int(n_sample * 0.8) :]

n_feature = rawX.shape[0]
width = 4

trainX = np.hstack(utils.expand_feature3(rawX, rawX, width))
validX = np.hstack(utils.expand_feature3(validX, rawX, width))
testX = np.hstack(utils.expand_feature3(testX, rawX, width))


#%% 训练模型
# 创建模型对象
n_iter = 100
lr = 0.2  # learning rate
model = LinearRegression(n_feature, lr)

# 用梯度下降方法进行迭代优化
for i in range(n_iter):
    model.update(trainX, trainy)
    model.evaluate(validX, validy)
    
# 通过观察 learning curve 检视学习过程
utils.plot_loss(model, n_iter)


#%% 观察模型
x_origin = np.linspace(X.min(), X.max(), 100)
xdata = np.hstack(utils.expand_feature3(x_origin[:, np.newaxis], rawX, width))
ydata = model.predict(xdata)

plt.figure()
plt.plot(x_origin, ydata, 'r-')
plt.scatter(rawX[:,0], trainy)
plt.xlabel('hand-size')
plt.ylabel('stature')
plt.legend(['gausian kernel model', 'data points'], loc='lower right')
plt.show()




#%% 载入葡萄酒数据（多元线性回归）
data = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)


#%% define constant
n_sample = data.shape[0]
n_feature = 12 # 通过数据的元信息已经提前知道
n_iter = 100
lr = 0.2 # 学习率是通过多次尝试得到的，你可以尝试其它值，观察训练过程


#%% divide dataset
np.random.seed(1)
np.random.shuffle(data)
X = data[:,:11]
y = data[:,-1].reshape((n_sample, 1)) 

trainX = X[: int(n_sample * 0.6)]
trainy = y[: int(n_sample * 0.6)]
validX = X[int(n_sample * 0.6) : int(n_sample * 0.8)]           
validy = y[int(n_sample * 0.6) : int(n_sample * 0.8)]
testX = X[int(n_sample * 0.8) :]           
testy = y[int(n_sample * 0.8) :]


#%% visualize data
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, init='pca', random_state=1)
Xtsne = tsne.fit_transform(trainX)

plt.figure()
colors = plt.cm.rainbow(np.linspace(0, 1, trainy.max()-trainy.min()+1))
for i in range(trainX.shape[0]):
    color = colors[ int(trainy[i,0]-trainy.min()) ]
    plt.scatter(Xtsne[i,0], Xtsne[i,1], s=2, c=color) # 为每个点分别着色
plt.show()



#%% feature normalization
mu = np.mean(trainX, axis=0)
sigma = np.std(trainX, axis=0)
trainX = (trainX - mu) / sigma
validX = (validX - mu) / sigma
testX = (testX - mu) / sigma

# add '1' column
trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
validX = np.hstack((np.ones((validX.shape[0], 1)), validX))
testX = np.hstack((np.ones((testX.shape[0], 1)), testX))


#%% train
# create model
# 注意：自始至终模型没有改变，变的只是特征
model = LinearRegression(n_feature, lr)

for i in range(n_iter):
    model.update(trainX, trainy)
    model.evaluate(validX, validy)

utils.plot_loss(model, n_iter)


#%% 目标值和测试值对照检查
# choose best model（选择验证误差 validloss 最小的模型）
mdl_idx = np.argmin(model.validloss)
best_mdl = model.snapshot[mdl_idx]
model.W = best_mdl

target_pred = np.hstack( (testy, model.predict(testX)) )
for i in range(testy.shape[0]):
    print('目标值:{0}, 预测值:{1}'.
          format(target_pred[i,0], target_pred[i,1]))
