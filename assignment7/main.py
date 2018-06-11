# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from kmeans import KMeans
from gmm import GMM
import utils
from scipy.stats import multivariate_normal
from scipy.misc import imread, imsave


#%% 载入测试数据
X = np.load('x.npy')

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show()


#%% KMeans 聚类
# 任务1：完成 KMeans 类中 fit 方法
model = KMeans(2) # two centroids
model.fit(X)
              
              
#%% 观察KMeans聚类过程
colors = ['r', 'b']
for centroids, tags in model.snapshot:
    plt.figure()
    # plot data points
    for d, t in zip(X, tags):
        plt.scatter(d[0], d[1], c=colors[t])
    # plot centroids
    for c in centroids:
        plt.scatter(c[0], c[1], c='k', marker='+')
    plt.show()


#%% 查询类别
query = np.array([1, 1])
print('{0} 所属类别是:{1}'.format(query, model.predict(query)))





#%% 高斯分布可视化，二维情形
mu = np.array([[0, 0]], dtype=np.float64)
sigma = np.array([[1, 3/5], [3/5, 2]], dtype=np.float64)
n = 50
x = np.linspace(-4, 4, n)
y = x.copy()
X, Y = np.meshgrid(x, y)
data = np.hstack((X.reshape((-1,1)), Y.reshape((-1,1))))
p = utils.gaussian_pdf(data, mu, sigma)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, p.reshape((n,n)), cmap=cm.coolwarm)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


#%% 高斯混合模型。以两个高斯分布的加权和为例
mu1 = np.array([[0, 0]], dtype=np.float64)
sigma1 = np.array([[1, 3/5], [3/5, 2]], dtype=np.float64)
w1 = 0.4

mu2 = np.array([[-1, -2]], dtype=np.float64)
sigma2 = np.array([[1, -3/5], [-3/5, 2]], dtype=np.float64)
w2 = 0.6

n = 50
x = np.linspace(-5, 5, n)
y = x.copy()
X, Y = np.meshgrid(x, y)
data = np.hstack((X.reshape((-1,1)), Y.reshape((-1,1))))
# 注意下面的加权和
p = w1*utils.gaussian_pdf(data, mu1, sigma1) + \
    w2*utils.gaussian_pdf(data, mu2, sigma2)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, p.reshape((n,n)), cmap=cm.coolwarm)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


#%% 载入测试数据
X = np.load('x.npy')

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show()



#%% GMM 聚类
# 任务2：实现 GMM 类的 predict 方法

model = GMM(2) # two clusters
model.fit(X)


#%% 观察GMM聚类过程
n = 100
x = np.linspace(X[:,0].min(), X[:,0].max(), n)
y = np.linspace(X[:,1].min(), X[:,1].max(), n)
x, y = np.meshgrid(x, y)

pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

for _, mu, sigma in model.snapshot:
    plt.figure()
    # plot data points
    plt.scatter(X[:,0], X[:,1])
        
    # plot contour
    for i in range(mu.shape[0]):
        rv = multivariate_normal(mu[i].tolist(), sigma[i].tolist())
        plt.contour(x, y, rv.pdf(pos))
    plt.show()


#%% 可视化每个样本的"soft-label"
W = model.predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=W[:,0]-W[:,1], cmap=cm.coolwarm)
plt.show()


#%% 如何选择聚类个数？
# 任务3：实现 utils 中 weighted_radius 函数功能

ks = [1, 2, 3, 4, 5]
avg_dist = []
for k in ks:
    model = GMM(k)
    model.fit(X)
    W = model.predict(X)
    # 计算样本点到簇（cluster）中心的加权距离，多个距离再取平均。
    dist = 0
    for i in range(model.mu.shape[0]):
        dist += utils.weighted_radius(model.mu[i], X, W[:,i])
    dist /= k
    avg_dist.append(dist)
    
plt.figure()
plt.plot(ks, avg_dist)
plt.title('average radius vs. cluster numbers')
plt.xlabel('cluster number')
plt.ylabel('average weighted radius')
plt.show()


#%% 图像聚类
im = imread('flower.jpg')
h, w, c = im.shape
X = im.reshape((h*w, c))

# GMM 聚类
colors = 3  # 由上一单元的方法确定
model = GMM(colors)
model.fit(X)


#%% 重新上色
W = model.predict(X)
color_idx = W.argmax(axis=1) # 把最大概率值对应的簇作为像素所属的簇
new_im = np.empty_like(X)
for i in range(X.shape[0]):
    new_im[i] = model.mu[color_idx[i]]
    
new_im = new_im.reshape((h, w, c)) # 把数据形状转为图像原始形状
imsave('new_flower.jpg', new_im) # 保存图像


#%% 抠图
for i in range(colors):
    mask = np.zeros_like(X)    # 利用掩码方式隐去不相关的图像部分
    for j in range(X.shape[0]):
        if color_idx[j] == i:
            mask[j] = 1
    new_im = X * mask         # 不是同一簇的像素变成了(0,0,0)（黑色）
    plt.figure()
    plt.imshow(new_im.reshape((h,w,c)))
    plt.show()