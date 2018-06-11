# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.pyplot as plt
import utils


#%% 载入测试数据
X = np.load('test.npy')
colors = ['r']*16 + ['g']*17 + ['b']*17
plt.figure()
for i in range(50):
    plt.scatter(X[i,0],X[i,1],c=colors[i])
plt.title('original X')
plt.show()


#%% 归一化
X = (X - X.mean(axis=0)) / X.std(axis=0)



#%% 计算协方差矩阵（强调零均值化的必要性）
sigma = np.matmul(np.transpose(X), X) / X.shape[0]


#%% 利用SVD分解计算sigma的特征值与特征向量
# http://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/

U, s, V = np.linalg.svd(sigma)
# 利用linalg.svd而不用linalg.eig分解sigma，是因为svd分解后
# s中的特征值是按从大到小的顺序排列的，便于选择主成分
# U和V互为转置的关系

# 任务1：在console键入 np.linalg.eig? 阅读其文档，利用该函数计算特征向量U和特征值s。
#       特征值按从大到小顺序排列，对应特征向量也排序。
#       与 np.linalg.svd 的结果对比。


#%% 旋转数据
Xrot = np.matmul(X, U)
plt.figure()
for i in range(50):
    plt.scatter(Xrot[i,0],Xrot[i,1],c=colors[i])
plt.title('rotated X')
plt.show()


#%% 选择主成分个数
print(s)
reserved_var = s[0] / s.sum()
print('保留的方差：%.4f' % reserved_var)


#%% 降维：取Xrot前k列（此处 k=1），相当于其它列置零
# 将第二个维度的数据置零，观察第一个维度数据的分布
Xrot[:,1] = 0 
plt.figure()
for i in range(50):
    plt.scatter(Xrot[i,0], Xrot[i,1], c=colors[i])
plt.title('rotated X')
plt.show()


#%% 利用Xrot第一列数据还原数据
Xrec = np.outer(Xrot[:,0], V[0, :])

plt.figure()
for i in range(50):
    plt.scatter(Xrec[i,0], Xrec[i,1], c=colors[i])
plt.title('rotated X')
plt.show()



#%% 载入数据
# Olivetti Faces dataset: 40 people, 10 photos each, 64 by 64 large
faces = np.load('olivettifaces.npy')
n_sample, n_feature = faces.shape


#%% 划分数据
np.random.seed(1)
idxs = np.random.permutation(n_sample)

trainX = faces[idxs[: 300]]
testX = faces[idxs[300 :]]  

# 数据归一化
mu = np.mean(trainX, axis=0)
std = np.std(trainX, axis=0)
trainX = trainX - mu # 零均值化即可
testX = testX - mu



#%% 计算sigma，SVD分解，选择主成分个数
sigma = np.matmul(np.transpose(trainX), trainX) / trainX.shape[0]
U, s, V = np.linalg.svd(sigma)

# 任务2：实现 utils.dimrd_degree 
K = utils.dimrd_degree(s, threshold=0.99) # 保留99%的方差
print('降到%d个维度。' % K)


#%% 展示“特征脸”
im_num = 10
plt.figure()
for i in range(im_num):
    im = U[:,i]
    plt.subplot(im_num/2, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title('eigen-face %d' % i)
    plt.imshow(im.reshape((64,64)), cmap='gray', interpolation='None')

plt.show()


#%% 数据降维(只与前K个特征向量相乘，即投影到特征向量方向，或理解为”特征脸“的线性组合)
# Xdrd 某一行数据是某个图像在前K个“特征脸”上的系数
# 注意：在测试集上降维，观察结果
Xdrd = np.matmul(testX, U[:,:K])


#%% 还原数据，观察还原效果
Xrec = np.matmul(Xdrd, V[:K, :])

# 观察对比前五幅图
im_num = 5
plt.figure()
for row in range(im_num ):
    im0 = testX[row, :]
    im0 = im0 + mu
    plt.subplot(im_num , 2, row*2+1)
    plt.xticks([])
    plt.yticks([])
    plt.title('original')
    plt.imshow(im0.reshape((64,64)), cmap='gray', interpolation='None')
    
    im1 = Xrec[row, :]
    im1 = im1 + mu
    plt.subplot(im_num , 2, row*2+2)
    plt.xticks([])
    plt.yticks([])
    plt.title('recovered')
    plt.imshow(im1.reshape((64, 64)), cmap='gray', interpolation='None')
plt.show()

# MSE
PCA_error = np.mean((testX - Xrec) ** 2)
print("PCA reconstruction error: %.4f" % (PCA_error))


#%% 利用自编码器降维
from nn import NeuralNetwork
from linear import Linear
from relu import Relu

lr = 0.00001

model = NeuralNetwork()
model.layers.append(Linear(n_feature, K, lr))
model.layers.append(Relu())
model.layers.append(Linear(K, n_feature, lr))


#%% 归一化数据
trainX = faces[idxs[: 300]]
testX = faces[idxs[300 :]]  

trainX = (trainX - mu) / std
testX = (testX - mu) / std



#%% 训练自编码器
n_iter = 15
batchsize = 5
trainloss = []
testloss = []
snapshot = []

# import pdb; pdb.set_trace()
for i in range(n_iter):
    # 每一轮迭代前，产生一组新的序号（目的在于置乱数据）
    idxs = np.random.permutation(trainX.shape[0])
    
    for j in range(0, trainX.shape[0], batchsize):
        batchX = trainX[idxs[j : j+batchsize]]
        
        # denoising auto-encoder
        # 可以向训练数据中注入噪声来提高模型性能。本次教学不讨论这方面内容。
        #noise_level = 0.2
        #noise = np.random.normal(0, noise_level, batchX.shape)
        #batchX = batchX + noise
        
        batchy = batchX.copy() # 对于自编码器，其自身是目标值
        
        y_hat = model.forward(batchX)
        # MSE loss
        rec_error = y_hat - batchy
        trainloss.append((rec_error ** 2).mean())
        
        # backprop
        model.backward(rec_error)
        
        # evaluate
        test_error = model.forward(testX) - testX
        testloss.append((test_error ** 2).mean())
        snapshot.append((model.layers[0].W.copy(), model.layers[-1].W.copy()))
        

utils.plot_loss(trainloss, testloss)


#%% 还原数据，观察还原效果
# choose best model
best_idx = np.argmin(testloss)
encdec = snapshot[best_idx]
model.layers[0].W = encdec[0]
model.layers[2].W = encdec[1]

# 利用自编码器压缩再重建testX
AErec = model.forward(testX)

# 观察对比前五幅图
im_num = 5
plt.figure()
for row in range(im_num ):
    im0 = testX[row, :] * std + mu
    plt.subplot(im_num , 2, row*2+1)
    plt.xticks([])
    plt.yticks([])
    plt.title('original')
    plt.imshow(im0.reshape((64,64)), cmap='gray', interpolation='None')
    
    im1 = AErec[row, :] * std + mu
    plt.subplot(im_num , 2, row*2+2)
    plt.xticks([])
    plt.yticks([])
    plt.title('recovered')
    plt.imshow(im1.reshape((64, 64)), cmap='gray', interpolation='None')
plt.show()

# MSE
AE_error = np.mean(((testX - AErec)*std) ** 2)
print("Auto-encoder reconstruction error: %.4f" % (AE_error))


#%% face matching and retrieval in the reduced dimension
# PCA 或 bottlenet 自编码器将数据投射到低维空间，可以认为去除了噪声保留了数据内在的分布规律
# 下面定性检验降维后的特征比原始特征更有利于检索


#%% 用原始特征
new_feature = faces.copy().astype(np.float64)


#%% 用PCA特征
new_feature = np.matmul(faces, U[:,:K])


#%% 用自编码器编码后的特征
norm_faces = (faces - mu) / std
new_feature = model.layers[1].forward(model.layers[0].forward(norm_faces))



#%% 在特征空间查询相似样本
query = new_feature[20, :]

# Euclidean distance
compares, rank = utils.euclidean_rank(new_feature, query)


# Adjusted Cosine Similarity
# 任务3： 实现余弦相似度计算
#compares, rank = utils.cosine_rank(new_feature, query)


# 展示结果
plt.figure()
for i in range(10):
    im = faces[rank[i]].reshape((64,64))
    plt.subplot(5, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Dist:%.2f, Rank:%d' % (compares[rank[i]], i))
    plt.imshow(im, cmap='gray', interpolation='None')
plt.show()


#%% 定量评价
counter = 0
for i in range(0, 400):
    query = new_feature[i, :]
    # 比较两种距离(相似度)度量
    _, rank = utils.euclidean_rank(new_feature, query)
    # _, rank = utils.cosine_rank(new_feature, query)
    
    counter += np.sum(np.logical_and(rank[:10]>int(np.floor(i/10)*10), rank[:10]<int(np.ceil((i+1)/10)*10)))
    
print('在前10个结果中检出同一人图像的数量是：%d' % counter)
