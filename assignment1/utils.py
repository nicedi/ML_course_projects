# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def show_data_model(model, X, y):
    xdata = np.linspace(X.min(), X.max(), 100)
    ydata = model.k * xdata + model.b
    plt.plot(xdata, ydata, 'r-')
    plt.scatter(X, y)
    plt.legend(['linear model', 'data points'], loc='lower right')
    plt.show()
    
    
    
def show_contour_model(model, X, y):
    l = 100
    k = np.linspace(50, 100, l)
    b = np.linspace(1650, 1700, l)
    K, B = np.meshgrid(k, b)
    k0, b0 = model.k, model.b
    
    J = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            model.k, model.b = k[i], b[j]
            J[i, j] = model.get_loss(X, y)
    plt.contour(K, B, J, 20)
    model.k ,model.b = k0, b0
    plt.scatter(model.k, model.b)
    plt.xlabel('k')
    plt.ylabel('b')
    plt.show()
    
    
def plot_loss(model, n_iter):
    plt.plot(model.trainloss, 'b-', model.testloss, 'r-')
    plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['train loss', 'test loss'])
    plt.show()