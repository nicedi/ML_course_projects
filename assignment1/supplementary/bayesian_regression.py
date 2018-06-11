# -*- coding: utf-8 -*-
import numpy as np
#import scipy as sp
import seaborn as sns
import pymc3 as pm
import matplotlib.pyplot as plt

np.random.seed(101)


#%% 载入数据集
data = np.genfromtxt('dataset.csv', delimiter=',')
np.random.shuffle(data)

stature = data[30:,2]
handsize = data[30:,3]
footsize = data[30:,4]

stature_test = data[:30,2]
handsize_test = data[:30,3]
footsize_test = data[:30,4]


#%% 手的大小与身高的关系
plt.figure()
plt.scatter(handsize, stature)
plt.xlabel('hand size (mm)')
plt.ylabel('stature (mm)')
plt.show()

#%% handsize 直方图
sns.distplot(handsize)


#%% 脚的大小与身高的关系
plt.figure()
plt.scatter(footsize, stature)
plt.xlabel('foot size (mm)')
plt.ylabel('stature (mm)')
plt.show()


#%% footsize 直方图
sns.distplot(footsize)


#%% 均值归一化，十分必要
handsize_mu = handsize.mean()
handsize -= handsize_mu


#%%  定义模型
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=1700, sd=10)
    beta = pm.HalfNormal('beta', sd=5)
    epsilon = pm.HalfCauchy('epsilon', 10)
    nu = pm.Deterministic('nu', pm.Exponential('nu_', 1/29)+1)
    mu = pm.Deterministic('mu', alpha + beta * handsize)
    y_pred = pm.StudentT('y_pred', mu=mu, sd=epsilon, \
                         nu=nu, observed=stature)
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace = pm.sample(2000, step=step, start=start)


#%% 检视结果
trace = trace[100:] # burn-in
# traceplot 得到两幅图，左边是核密度估计图（Kernel Density Estimation， KDE）
# 可理解为平滑后的直方图；右边是采样过程中的采样值，其看起来应该像白噪声，即有很好的
# 混合度（mixing）
pm.traceplot(trace)

# 用forestplot将 R_hat(应小于1.1且在1附近)和参数均值、50% HPD
# (Highest Posterior Density, HPD)、95% HPD 表示出来
pm.forestplot(trace, varnames=['alpha'])
pm.forestplot(trace, varnames=['beta'])

# 理想的采样应该不会是自相关的，用autocorrplot观察自相关程度。
pm.autocorrplot(trace)

# summary提供对后验的文字描述
pm.summary(trace)

# 后验可视化总结：Kruschke图
pm.plot_posterior(trace['alpha'], kde_plot=True)
pm.plot_posterior(trace['beta'], kde_plot=True)


#%% 对后验进行解释和可视化
plt.plot(handsize, stature, 'b.')
alpha_m = trace['alpha'].mean()
beta_m = trace['beta'].mean()
plt.plot(handsize, alpha_m+beta_m*handsize, c='k', \
         label='y={:.2f}+{:.2f}*x'.format(alpha_m, beta_m))
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc=2, fontsize=14)


#plt.plot(handsize, alpha_m+beta_m*handsize, c='k', \
#         label='y={:.2f}+{:.2f}*x'.format(alpha_m, beta_m))
idx = np.argsort(handsize)
x_ord = handsize[idx]
sig = pm.hpd(trace['mu'], alpha=.02)[idx]
plt.fill_between(x_ord, sig[:,0], sig[:,1], color='gray')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)


#%% 后验预测检查（posterior predictive check, PPC）
y_pred = pm.sample_ppc(trace, 100, model)
sns.kdeplot(stature, c='b')

for i in y_pred['y_pred']:
    sns.kdeplot(i, c='r', alpha=0.1)
    
plt.xlim(1300, 2000)
