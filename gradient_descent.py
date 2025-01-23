#!/usr/bin/env python
# coding: utf-8
# In[6]:


import numpy as np
from auxiliary_functions_and_classes import minus


def learning_schedule(t):
    t0 = 20 
    t1 = 1000
    return t0 / (t + t1)


def batch_gradient_descent(X, y, n_iterations=1000, eta=0.1):
    """
    Реализация пакетного градиентного спуска
    
    X * theta = y
    eta - скорость обучения
    n_iterations - число итераций
    @return: theta - решение, полученное методом пакетного градиентного спуска 
    """

    m = len(y)
    theta = np.random.randn(m,1) #random initilization
    
    for i in range(n_iterations):
        gradients = 2/m * X.T.dot(minus(X.dot(theta),y))
        theta = theta - eta * gradients

    return theta


def mini_batch_gradient_descent(X, y, n_iterations=1000, minibatch_size=3, eta=0.1):
    """
    Реализация минипакетного градиентного спуска
    
    X * theta = y
    eta - скорость обучения
    n_iterations - число итераций
    minibatch_size - число экземпляров, использующееся при вычислении градиента 
    @return: theta - решение, полученное методом минипакетного градиентного спуска 
    """

    m = len(y)
    theta = np.random.randn(m,1) #random initilization
    
    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            gradients = 2/minibatch_size * xi.T.dot(minus(xi.dot(theta), yi))
            eta = learning_schedule(t)
            theta = theta - eta * gradients

    return theta

