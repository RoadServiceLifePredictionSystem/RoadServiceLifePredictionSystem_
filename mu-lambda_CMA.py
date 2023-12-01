#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random


def mu_lambda_cmaes(N: int, m: int, M: int, xmean: list, stopfitness: float, stopeval: int, stopdif: float, f):
    """
    Реализация модификации mu-lambda CMA
    
    @param N: int 
        Размерность задачи
    @param m: int 
        Количество векторов, которые выбираются из поколения для генерации следующего.
    @param M: int 
        Количество векторов, которые генерируются в одном поколении. 
    @param xmean: list 
        Начальный средний вектор 
    @param stopfitness: float
        Целевое значение
    @param ftarget: float
        Целевое значение функции
    @param mu: int 
        Количество векторов, которые выбираются из поколения для генерации следующего.
    @param stopeval: int
        Максимальное число вычислений значения f.
    @param stopdif: float
        Разница между последующими полученными значениями не больше stopdif.
    @param f: callable
        Функция, которую минимизируем. Она принимает на вход вектор x и возвращает скалярное значение
    @return: tuple
        (best_value, countiter), где
        best_value - лучшее найденное решение
        countiter - количество итераций, потребовавшееся для получения решения
    """
    lmbda = M
    sigma = 0.5
    eta = []
    for j in range(N):
        eta.append(random.normalvariate(0, 1)) # нормально распрделённый вектор N(0, I)
    eta = np.array(eta)
    t = lmbda // m
    kol = 0

    flag = False
    while (kol < stopeval and  f(xmean[0]) >= stopfitness and not flag):
        x = []
#         if lmbda % m != 0:
#             t += 1 
        for i in range(m):
            for j in range(t):
                x.append(xmean[i] + eta * sigma)
        x.sort(key = f)  
        if abs(f(x[0]) - f(xmean[0])) < stopdif:
            flag = True
        xmean = x[:m]  
        kol += 1

    return  xmean[0], kol


# In[ ]:




