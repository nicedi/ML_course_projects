# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import utils


#%% 载入葡萄酒数据
data = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)


#%% 定义常量
n_sample = data.shape[0]
n_feature = 12
n_iter = 100
lr = 0.2


#%% 划分数据集
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


#%% 特征归一化
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

tsne = TSNE(n_components=2, init='pca', random_state=1)
Xtsne = tsne.fit_transform(trainX)

plt.figure()
colors = plt.cm.rainbow(np.linspace(0, 1, trainy.max()-trainy.min()+1))
for i in range(trainX.shape[0]):
    color = colors[ int(trainy[i,0]-trainy.min()) ]
    plt.scatter(Xtsne[i,0], Xtsne[i,1], s=2, c=color) # 为每个点分别着色
plt.show()


#%% 用 Gradient Descent 训练
# 创建模型对象
model1 = LinearRegression(n_feature, lr)

for i in range(n_iter):
    model1.update(trainX, trainy)
    model1.evaluate(validX, validy)

utils.plot_loss(model1, n_iter)


#%% 用 Coordinate Descent 训练
# 任务1：实现 LinearRegressionCD 类的 update_CD 方法

# 创建模型对象， 利用坐标下降法训练模型
from linear_regression_CD import LinearRegressionCD
model2 = LinearRegressionCD(n_feature)

counter = 0
while True:
    W_old = model2.W.copy()
    
    for j in range(n_feature):
        model2.update_CD(trainX, trainy, j)
        model2.evaluate(validX, validy)
        counter += 1
    
    change = np.mean((W_old - model2.W)**2)
    print(change)
    
    if change < 1e-5 or counter > 500:
        break
utils.plot_loss(model2, counter)


#%% 将上述代码包装进一个函数，以便在下一个单元中调用
def train_CD():
    counter = 0
    while True:
        W_old = model.W.copy()
        
        for j in range(n_feature):
            model.update_CD(trainX, trainy, j)
            model.evaluate(validX, validy)
            counter += 1
        
        change = np.mean((W_old - model.W)**2)
        if change < 1e-5 or counter > 500:
            break

    
#%% train with l1 norm term (LASSO)
from linear_regression_lasso import LinearRegressionLasso
lambd = 0.1 # try different lambd:0.01, 0.1, 1, 10

# 任务2：实现 LinearRegressionLasso 类的 update_CD 方法
model = LinearRegressionLasso(n_feature, lambd)
train_CD()
utils.plot_loss(model, len(model.trainloss))


#%% 查看lasso回归模型的参数如何随着lambd的变化而变化
result = None
all_lambd = np.linspace(0.01, 0.5, 100)

for lambd in all_lambd:
    model = LinearRegressionLasso(n_feature, lambd)
    train_CD()
    if result is None:
        result = model.W.copy()
    else:
        result = np.hstack((result, model.W.copy()))
result = np.transpose(result)

plt.figure()
plt.plot(all_lambd.reshape((-1,1)), result[:, 1:])
plt.xlabel('lambda')
plt.ylabel('weight')
plt.title('weight shrinkage path of lasso regression')
names = ['fixed acidity','volatile acidity','citric acid',
         'residual sugar','chlorides','free sulfur dioxide',
         'total sulfur dioxide','density','pH','sulphates','alcohol']
plt.legend(names, loc='upper right')
plt.show()



#%% train with l2 norm term (ridge regression)
from linear_regression_ridge import LinearRegressionRidge
lambd = 1

# 任务3：实现 LinearRegressionRidge 类的 update_CD 方法
model = LinearRegressionRidge(n_feature, lambd)
train_CD()
utils.plot_loss(model, len(model.trainloss))



#%% 查看ridge回归模型的参数如何随着lambd的变化而变化
result = None
all_lambd = np.linspace(0.1, 10, 100)

for lambd in all_lambd:
    model = LinearRegressionRidge(n_feature, lambd)
    train_CD()
    if result is None:
        result = model.W.copy()
    else:
        result = np.hstack((result, model.W.copy()))
result = np.transpose(result)

plt.figure()
plt.plot(all_lambd.reshape((-1,1)), result[:, 1:])
plt.xlabel('lambda')
plt.ylabel('weight')
plt.title('weight shrinkage path of ridge regression')
names = ['fixed acidity','volatile acidity','citric acid',
         'residual sugar','chlorides','free sulfur dioxide',
         'total sulfur dioxide','density','pH','sulphates','alcohol']
plt.legend(names, loc='upper right')
plt.show()



#%% 目标值和测试值对照检查
# choose best model
mdl_idx = np.argmin(model.validloss)
best_mdl = model.snapshot[mdl_idx]
model.W = best_mdl

target_pred = np.hstack( (testy, model.predict(testX)) )
for i in range(testy.shape[0]):
    print('目标值:{0}, 预测值:{1}'.
          format(target_pred[i,0], target_pred[i,1]))
