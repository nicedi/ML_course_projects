# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(model, n_iter):
    plt.figure()
    plt.plot(model.trainloss, 'b-', model.validloss, 'r-')
    plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training loss', 'validation loss'])
    plt.show()