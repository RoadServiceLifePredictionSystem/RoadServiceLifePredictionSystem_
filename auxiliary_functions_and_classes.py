#!/usr/bin/env python
# coding: utf-8

# In[8]:


import math
import time
import numpy as np
from random import normalvariate as random_normalvariate
from typing import Callable, NamedTuple
from CMA import fmin


class PartialSum(NamedTuple):
    func_partial_sum : Callable
    randn : Callable
    N : int
    list_args : tuple
        

def partial_sum_function(*args, N=1, randn=random_normalvariate):
    v = [randn(*args) for i in range(N)]
    return sum(v) / N


def exact_solution(A, B):
    """
    Точное решение плохо обусловленной СЛУ: A * X = B
    """
    return np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), B)


def generator_of_poorly_conditioned_matrices(N: int, sigma=None):
    
    """ 
    Генератор плохо обусловленных матриц
    sigma от -N + 2 до infinity. Чем больше, тем больше число обусловленности, тем более плохо обусловленная матрица
    """
    if sigma is None:
        sigma = 100 - N
    a = []
    for i in range(N):
        k = i / (N - 1 + sigma)
        if k <= 1:
            a.append(1 - k)
        else:
            a.append(0)
    res = [[] for i in range(N)]  
    res[0] = a
    for i in range(1, N):
        for j in range(N):
            res[i].append(res[(i-1) % N][(j-1) % N])
    return res


def average_duration_of_operation_time_and_confidence_intervals(N, func, *args, **kwargs):
    """
    N - число испытаний
    Вычисление среднего времени и доверительных интервалов
    """
    times = []
    for i in range(N):
        start = time.time()
        decision = func(*args, **kwargs)
        times.append(time.time() - start)
        
    av_time = sum(times) / N
    st_dev = math.sqrt(sum(list((x - av_time)**2 for x in times)) / N)
    z_0_475 = 1.96
    conf_int = z_0_475 * st_dev / math.sqrt(N)
    x_beg = av_time - conf_int
    x_end = av_time + conf_int
    
    return av_time, x_beg, x_end


def average_norm_of_the_difference_between_solutions_and_confidence_intervals(N, exact_dec, func, 
                                                                              *args, **kwargs):
    """
    N - число испытаний
    Вычисление средней нормы разности между полученным решением и точным и доверительных интервалов
    """
    norms = []
    for i in range(N):
        decision = func(*args, **kwargs)
        if func == fmin:
            dec = decision[0] - exact_dec
            norms.append(np.linalg.norm(dec))
        else:
            dec = minus(decision, exact_dec)
            norms.append(np.linalg.norm(list(map(lambda x: x[0], dec))))
        
    av_norm = sum(norms) / N
    st_dev = math.sqrt(sum(list((x - av_norm)**2 for x in norms)) / N)
    z_0_475 = 1.96
    conf_int = z_0_475 * st_dev / math.sqrt(N)
    x_beg = av_norm - conf_int
    x_end = av_norm + conf_int
    
    return av_norm, x_beg, x_end


def minus(A, B):
    if len(A) == len(B):
        C  = []
        for i in range(len(B)):
            C.append([])
            C[i].append(A[i][0] - B[i])
        return C
    else:
        return 0
    


# In[ ]:




