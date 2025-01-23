#!/usr/bin/env python
# coding: utf-8

# In[13]:

import math 
import numpy as np
import warnings as _warnings
from sys import stdout as _stdout
from math import log, exp, cos, pi


def fmin(objective_fct, xstart, sigma, randn,
         args=(), popsize='4 + int(3 * log(N))', 
         maxfevals='1e3 * N**2', exact_solution=None, mu=None, ftarget=None,
         verb_disp=100, verb_cmu_decreasing=500):
    """
    Минимизация нелинейных невыпуклых функций. 
    Цель - найти такой x, на котором objective_fct достигает глобального минимума

    @param objective_fct: callable
        Функция, которую минимизируем. Она принимает на вход вектор x и возвращает скалярное значение
    @param xstart: list 
        Начальный вектор x0
    @param sigma: float
        Начальное стандартное отклонение по всем координатам
    @param randn: PartialSum
        Объект хранит информацию о распределении векторов, использующемся внутри CMA_ES
    @param args: tuple
        Дополнительные параметры для objective_fct, если есть
    @param popsize: int or str
        Количество векторов, которые генерируются в одном поколении. Если параметр задан через строку, 
        N - размерность задачи 
    @param maxfevals: int or str
        Максимальное число вычислений значения objective_fct. Если параметр задан через строку, 
        N - размерность задачи 
    @param exact_solution: list
        Точное решение, если оно есть, чтобы сравнить с полученным
    @param mu: int 
        Количество векторов, которые выбираются из поколения для генерации следующего.
    @param ftarget: float
        Целевое значение функции
    @param verb_disp: int
        Отображение в консоли кадждой verb_disp итерации. Если verb_disp = 0, никогда не выводим 
    @param verb_cmu_decreasing: int
        Раз во сколько итераций уменьшается скорость обучения cmu
    @return: tuple
        (xmin: list, es:CMAES), где 
        xmin - лучшее найденное решение
        es - объект класса CMAES
    
    Пример вызова
    =======
    Минимизируем функцию felli:
    
    >>> import CMA

    >>> def felli(x):
    ...     return sum(10**(6 * i / (len(x)-1)) * xi**2
    ...                for i, xi in enumerate(x))
    >>> res = CMA.fmin(felli, 3 * [0.5], 0.3, verb_disp=100)  
    
    Вывод:
    
    evals: ax-ratio max(std)   f-value
        7:     1.0  3.4e-01  240.2716966
       14:     1.0  3.9e-01  2341.50170536
      700:   247.9  2.4e-01  0.629102574062
     1400:  1185.9  5.3e-07  4.83466373808e-13
     1421:  1131.2  2.9e-07  5.50167024417e-14
    termination by {'tolfun': 1e-12}
    best f-value = 2.72976881789e-14
    solution = [5.284564665206811e-08, 2.4608091035303e-09, -1.3582873173543187e-10]
    
    >>> print(res[0])  
    [5.284564665206811e-08, 2.4608091035303e-09, -1.3582873173543187e-10]
    
    >>> res[1].result[1])  
    2.72976881789e-14
    
"""
    
    es = CMAES(xstart, sigma, randn, popsize=popsize, mu=mu, maxfevals=maxfevals, ftarget=ftarget)
    
    while not es.stop():
        X = es.ask()  # получение выборки кандидатов в решения
        fit = [objective_fct(x, *args) for x in X]  # расчёт значений функции для выборки
        es.tell(X, fit)  # обновление параметров распределение

        
        es.disp(verb_disp)
        
#         if verb_cmu_decreasing:
#             if es.counteval / es.params.lam % verb_cmu_decreasing < 1:
#                 print(f'Decreasing cmu: old value = {es.params.cmu}, new value = {es.params.cmu * 0.85}')
#                 print(f'Increasing c1: old value = {es.params.c1}, new value = {es.params.c1 * 1.15}')
#                 es.params.cmu *= 0.85
#                 es.params.c1 *= 1.15

    if verb_disp:  
        es.disp(1)
        print('termination by', es.stop())
        print('best f-value =', es.result[1])
        print('solution =', es.result[0])
        if exact_solution is not None:
#             print('solution - the_exact_solution=', es.result[0] - exact_solution)
            print('The norm of the L2 difference vector(solution - the_exact_solution)=', 
                  np.linalg.norm(es.result[0] - exact_solution))

    return [es.best.x if es.best.f < objective_fct(es.xmean) else
            es.xmean, es]


class CMAESParameters(object):
    """
    Внутренние статические параметры для CMAES, которые создаются в начале 
    работы алгоритма и больше не меняются
    """
    
    default_popsize = '4 + int(3 * log(N))'
    def __init__(self, N, popsize=None, mu=None,
                 RecombinationWeights=None):

        self.dimension = N
        self.chiN = N**0.5 * (1 - 1. / (4 * N) + 1. / (21 * N**2))

        # Параметры, относящиеся к выборке
        self.lam = eval(safe_str(popsize if popsize else
                                 CMAESParameters.default_popsize,
                                 {'int': 'int', 'log': 'log', 'N': N}))
        if mu:
            self.mu = mu
        else:
            self.mu = int(self.lam / 2)  # number of parents/points/solutions for recombination
            
        if RecombinationWeights:
            self.weights = RecombinationWeights(self.lam)
            self.mueff = self.weights.mueff
            
        else:  # ручное управление весами
            _weights = [log(self.mu  + 0.5) - log(i + 1) if i < self.mu else 0
                        for i in range(self.lam)]
            w_sum = sum(_weights[:self.mu])
            self.weights = [w / w_sum for w in _weights]  # сумма стремится к 1
            self.mueff = sum(self.weights[:self.mu])**2 / \
                         sum(w**2 for w in self.weights[:self.mu])  # эффективная дисперсия суммы w_i x_i

        # Параметры, относящиеся к адаптации
        self.cc = (4 + self.mueff/N) / (N+4 + 2 * self.mueff/N)  # временная постоянная для кумуляции C
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)  # временная постоянная для кумуляции контроля sigma
        self.c1 = 2 / ((N + 1.3)**2 + self.mueff)  # скорость обучения rank-one обновления  C

        self.cmu = min([1 - self.c1, 2 * 
                        (self.mueff - 2 + 1/self.mueff) / ((N + 2)**2 + self.mueff)])  # для rank-mu обновления
        self.damps = 2 * self.mueff/self.lam + 0.3 + self.cs  # демпфирование для sigma, обычно близко к 1

        if RecombinationWeights:
            self.weights.finalize_negative_weights(N, self.c1, self.cmu)
            
            
        self.lazy_gap_evals = 0.5 * N * self.lam * (self.c1 + self.cmu)**-1 / N**2

class CMAES():  
    """
    Класс для нелинейной невыпуклой численной минимизации с помощью CMA-ES.
    
    """
    def __init__(self, xstart, sigma, randn, 
                 popsize=CMAESParameters.default_popsize,
                 ftarget=None, mu=None,
                 maxfevals='100 * popsize + '  
                           '150 * (N + 3)**2 * popsize**0.5'):
        """
        Инициализация

        @param xstart: list 
            Начальный вектор x0
        @param sigma: float
            Начальное стандартное отклонение по всем координатам
        @param randn: PartialSum
            Объект хранит информацию о распределении векторов, использующемся внутри CMA_ES
        @param popsize: int or str
            Количество векторов, которые генерируются в одном поколении. Если параметр задан через строку, 
            N - размерность задачи
        @param ftarget: float
            Целевое значение функции
        @param mu: int 
            Количество векторов, которые выбираются из поколения для генерации следующего.
        @param maxfevals: int or str
            Максимальное число вычислений значения objective_fct. Если параметр задан через строку, 
            N - размерность задачи 
        """
        
        N = len(xstart)  # размерность задачи
#         self.params = CMAESParameters(N, popsize, mu=mu, RecombinationWeights=RecombinationWeights)
        self.params = CMAESParameters(N, popsize, mu=mu)
        self.maxfevals = eval(safe_str(maxfevals,
                                       known_words={'N': N, 'popsize': self.params.lam}))
        self.ftarget = ftarget  # остановка, если fitness <= ftarget
        self.randn = randn

        # динамические переменные 
        self.xmean = xstart[:]  
        self.sigma = sigma
        self.pc = N * [0]  # эволюционный путь для C
        self.ps = N * [0]  # и для sigma
        self.C = DecomposingPositiveMatrix(N)  # ковариационная матрица
        self.counteval = 0  # счётчик вычислений. Количество итераций = counteval / lam
        self.fitvals = []   
        self.best = BestSolution()

    def ask(self):
        """
        Получение выборки размера lambda, 
        распределенных соответственно:

            m + sigma * Normal(0,C) = m + sigma * B * D * Normal(0,I)
                                    = m + B * D * sigma * Normal(0,I)
        """
        self.C.update_eigensystem(self.counteval,
                                  self.params.lazy_gap_evals)

        candidate_solutions = []
        for _k in range(self.params.lam):  
            z = [self.sigma * eigenval**0.5 * \
                 self.randn.func_partial_sum(*self.randn.list_args, N=self.randn.N, randn=self.randn.randn)
                 for eigenval in self.C.eigenvalues]

            y = dot(self.C.eigenbasis, z)
            candidate_solutions.append(plus(self.xmean, y))
        return candidate_solutions

    def tell(self, arx, fitvals):
        """
        Обновление эволюционного пути и параметров распределения m,
        sigma, и C.

        @param arx: list 
            Список кандидатов для решения.
        @param fitvals: list
            Соответсвующие значения objective function для кандидатов
        """
        
        self.counteval += len(fitvals)  
        N = len(self.xmean)
        par = self.params
        xold = self.xmean  
        

        # Сортировка по значениям
        arx = [arx[k] for k in argsort(fitvals)]  
        self.fitvals = sorted(fitvals)  
        self.best.update(arx[0], self.fitvals[0], self.counteval)
        
        
        # Вычисление нового вектора среднего значения
        self.xmean = dot(arx[0:par.mu], par.weights[:par.mu], transpose=True)
        

        # Кумуляция: обновление эволюционных путей
        y = minus(self.xmean, xold)
        z = dot(self.C.invsqrt, y)  # == C**(-1/2) * (xmeannew - xold)
        csn = (par.cs * (2 - par.cs) * par.mueff)**0.5 / self.sigma

        for i in range(N):  # обновление пути ps
            self.ps[i] = (1 - par.cs) * self.ps[i] + csn * z[i]
            
        ccn = (par.cc * (2 - par.cc) * par.mueff)**0.5 / self.sigma
        
        # Выключение rank-one аккумуляции, когда sigma быстро увеличивается
        hsig = (sum(x**2 for x in self.ps) / N  # ||ps||^2 / N должно быть близко к 1
                / (1-(1-par.cs)**(2*self.counteval/par.lam))  
                < 2 + 4./(N+1))  
        for i in range(N):  # обновление пути pc
            self.pc[i] = (1 - par.cc) * self.pc[i] + ccn * hsig * y[i]
            
        # Адаптация ковариационной матрицы C
        # Незначительная поправка на потерю дисперсии из-за hsig
        c1a = par.c1 * (1 - (1-hsig**2) * par.cc * (2-par.cc))
        self.C.multiply_with(1 - c1a - par.cmu * sum(par.weights))  # C *= 1 - c1 - cmu * sum(w)
        self.C.addouter(self.pc, par.c1)  # C += c1 * pc * pc^T, rank-one обновление

        for k, wk in enumerate(par.weights):  # rank-mu обновление
            if wk < 0:  # гарантия положительной определённости С
                wk *= N * (self.sigma / self.C.mahalanobis_norm(minus(arx[k], xold)))**2

            self.C.addouter(minus(arx[k], xold),  # C += wk * cmu * dx * dx^T
                            wk * par.cmu / self.sigma**2)

        # Адаптация размера шага sigma
        cn, sum_square_ps = par.cs / par.damps, sum(x**2 for x in self.ps)
        self.sigma *= exp(min(1, cn * (sum_square_ps / N - 1) / 2))

    def stop(self):
        """
        Возвращение причины остановки алгоритма в словаре в формате {'termination_reason':value, ...}
        """
        
        res = {}
        if self.counteval <= 0:
            return res
#         if self.counteval >= self.maxfevals:
#             res['maxfevals'] = self.maxfevals
        if self.ftarget is not None and len(self.fitvals) > 0 \
                and self.fitvals[0] <= self.ftarget:
            res['ftarget'] = self.ftarget
        if self.C.condition_number > 1e14:
            res['condition'] = self.C.condition_number
        if len(self.fitvals) > 1 \
                and self.fitvals[-1] - self.fitvals[0] < 1e-10:
            res['tolfun'] = 1e-10
        if self.sigma * max(self.C.eigenvalues)**0.5 < 1e-11:
            res['tolx'] = 1e-11
        if self.sigma > 1e11:
            res['discrepancy'] = self.sigma
        return res

    @property
    def result(self):
        """
        Возвращает tuple в формате(xbest, f(xbest), evaluations_xbest, evaluations,
        iterations, xmean, stds)
        """
        return (self.best.x,
                self.best.f,
                self.best.evals,
                self.counteval,
                int(self.counteval / self.params.lam),
                self.xmean,
                [self.sigma * C_ii**0.5 for C_ii in self.C.diag])

    def disp(self, verb_modulo=1):
        """
        Вывод информации по итерациям
        """
        if verb_modulo is None:
            verb_modulo = 20
        if not verb_modulo:
            return
        iteration = self.counteval / self.params.lam
        
        if iteration == 1 or iteration % (10 * verb_modulo) < 1:
            print('evals: ax-ratio max(std)   f-value')
        if iteration <= 2 or iteration % verb_modulo < 1:
            print(str(self.counteval).rjust(5) + ': ' +
                  ' %6.1f %8.1e  ' % (self.C.condition_number**0.5,
                                      self.sigma * max(self.C.diag)**0.5) +
                  str(self.fitvals[0]))
            _stdout.flush()


#_____________________________________________________________________
#_________________ Целевые функции _____________________

class ff(object):  
    """
    Набор тестовых функций в статических методах
    """

    @staticmethod  
    def elli(x):
        """
        Эллипсоидная функция
        """
        n = len(x)
        aratio = 1e3
        return sum(x[i]**2 * aratio**(2.*i/(n-1)) for i in range(n))

    @staticmethod
    def sphere(x):
        """
        Сферическая функция 
        """
        return sum(x[i]**2 for i in range(len(x)))

    @staticmethod
    def rosenbrock(x):
        """
        Функция Розенброка
        """
        n = len(x)
        if n < 2:
            raise ValueError('dimension must be greater one')
        return sum(100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i
                   in range(n-1))
    
    @staticmethod
    def rastrigin(x):
        """
        Функция Растригина
        """
        n = len(x)
        if n < 2:
            raise ValueError('dimension must be greater one')
        return 10*n + sum(x[i]**2 - 10*cos(2*pi*x[i]) for i
                   in range(n))
    
    @staticmethod
    def f_ploho_obuslovlennaya_matrica_6(x):
    
        A = [[1.0087, 0.9557, 0.9849, 1.069, 0.9009, 0.9243], [1.0341, 1.0652, 0.9273, 1.015, 1.0783, 0.9418],
        [0.9371, 0.9217, 0.9439, 1.0957, 1.0623, 0.9344], [1.0632, 0.9548, 0.9863, 1.088, 1.0635, 0.9672],
        [0.9351, 0.9746, 0.9011, 0.9505, 1.0591, 0.9031], [1.0198, 1.0208, 0.921, 0.9764, 0.9073, 1.0781]]
    
        B = [20.2011, 21.0487, 20.9129, 21.4044, 20.1037, 20.7351]
        
        n = 6
        summ = 0
        for i in range(n):
            sum_ = 0
            for j in range(n):
                sum_ += A[i][j] * x[j]
            sum_ -= B[i]
            summ += sum_ * sum_

        return summ
    
    @staticmethod
    def f_ploho_obuslovlennaya_matrica_10(x):
        
        A = [[1.0, 0.98989898989899, 0.9797979797979798, 0.9696969696969697, 0.9595959595959596, 
              0.9494949494949495, 0.9393939393939394, 0.9292929292929293, 0.9191919191919192, 0.9090909090909091], 
             [0.9090909090909091, 1.0, 0.98989898989899, 0.9797979797979798, 0.9696969696969697, 0.9595959595959596, 
              0.9494949494949495, 0.9393939393939394, 0.9292929292929293, 0.9191919191919192], [0.9191919191919192, 
              0.9090909090909091, 1.0, 0.98989898989899, 0.9797979797979798, 0.9696969696969697, 0.9595959595959596, 
              0.9494949494949495, 0.9393939393939394, 0.9292929292929293], [0.9292929292929293, 0.9191919191919192, 
              0.9090909090909091, 1.0, 0.98989898989899, 0.9797979797979798, 0.9696969696969697, 0.9595959595959596, 
              0.9494949494949495, 0.9393939393939394], [0.9393939393939394, 0.9292929292929293, 0.9191919191919192, 
              0.9090909090909091, 1.0, 0.98989898989899, 0.9797979797979798, 0.9696969696969697, 0.9595959595959596, 
              0.9494949494949495], [0.9494949494949495, 0.9393939393939394, 0.9292929292929293, 0.9191919191919192, 
              0.9090909090909091, 1.0, 0.98989898989899, 0.9797979797979798, 0.9696969696969697, 0.9595959595959596], 
             [0.9595959595959596, 0.9494949494949495, 0.9393939393939394, 0.9292929292929293, 0.9191919191919192, 
              0.9090909090909091, 1.0, 0.98989898989899, 0.9797979797979798, 0.9696969696969697], [0.9696969696969697, 
              0.9595959595959596, 0.9494949494949495, 0.9393939393939394, 0.9292929292929293, 0.9191919191919192, 
              0.9090909090909091, 1.0, 0.98989898989899, 0.9797979797979798], [0.9797979797979798, 0.9696969696969697, 
              0.9595959595959596, 0.9494949494949495, 0.9393939393939394, 0.9292929292929293, 0.9191919191919192, 
              0.9090909090909091, 1.0, 0.98989898989899], [0.98989898989899, 0.9797979797979798, 0.9696969696969697, 
              0.9595959595959596, 0.9494949494949495, 0.9393939393939394, 0.9292929292929293, 0.9191919191919192, 
              0.9090909090909091, 1.0]]
        
        B = [51.66666667, 52.12121212, 52.47474747, 52.72727273, 52.87878788, 52.92929293,
                52.87878788, 52.72727273, 52.47474747, 52.12121212]
        
        n = 10
        summ = 0
        for i in range(n):
            sum_ = 0
            for j in range(n):
                sum_ += A[i][j] * x[j]
            sum_ -= B[i]
            summ += sum_ * sum_

        return summ
    
    
    @staticmethod
    def f_ploho_obuslovlennaya_matrica_15(x):
        
        A = [[1.0, 0.984375, 0.96875, 0.953125, 0.9375, 0.921875, 0.90625, 0.890625, 0.875, 0.859375, 0.84375, 
              0.828125, 0.8125, 0.796875, 0.78125], [0.78125, 1.0, 0.984375, 0.96875, 0.953125, 0.9375, 0.921875, 
              0.90625, 0.890625, 0.875, 0.859375, 0.84375, 0.828125, 0.8125, 0.796875], [0.796875, 0.78125, 1.0, 
              0.984375, 0.96875, 0.953125, 0.9375, 0.921875, 0.90625, 0.890625, 0.875, 0.859375, 0.84375, 0.828125, 0.8125], 
             [0.8125, 0.796875, 0.78125, 1.0, 0.984375, 0.96875, 0.953125, 0.9375, 0.921875, 0.90625, 0.890625, 0.875, 
              0.859375, 0.84375, 0.828125], [0.828125, 0.8125, 0.796875, 0.78125, 1.0, 0.984375, 0.96875, 0.953125, 0.9375, 
              0.921875, 0.90625, 0.890625, 0.875, 0.859375, 0.84375], [0.84375, 0.828125, 0.8125, 0.796875, 0.78125, 1.0, 
              0.984375, 0.96875, 0.953125, 0.9375, 0.921875, 0.90625, 0.890625, 0.875, 0.859375], [0.859375, 0.84375, 
              0.828125, 0.8125, 0.796875, 0.78125, 1.0, 0.984375, 0.96875, 0.953125, 0.9375, 0.921875, 0.90625, 
              0.890625, 0.875], [0.875, 0.859375, 0.84375, 0.828125, 0.8125, 0.796875, 0.78125, 1.0, 0.984375, 0.96875, 
              0.953125, 0.9375, 0.921875, 0.90625, 0.890625], [0.890625, 0.875, 0.859375, 0.84375, 0.828125, 0.8125, 
              0.796875, 0.78125, 1.0, 0.984375, 0.96875, 0.953125, 0.9375, 0.921875, 0.90625], [0.90625, 0.890625, 0.875, 
              0.859375, 0.84375, 0.828125, 0.8125, 0.796875, 0.78125, 1.0, 0.984375, 0.96875, 0.953125, 0.9375, 0.921875], 
             [0.921875, 0.90625, 0.890625, 0.875, 0.859375, 0.84375, 0.828125, 0.8125, 0.796875, 0.78125, 1.0, 0.984375, 
              0.96875, 0.953125, 0.9375], [0.9375, 0.921875, 0.90625, 0.890625, 0.875, 0.859375, 0.84375, 0.828125, 0.8125, 
              0.796875, 0.78125, 1.0, 0.984375, 0.96875, 0.953125], [0.953125, 0.9375, 0.921875, 0.90625, 0.890625, 0.875, 
              0.859375, 0.84375, 0.828125, 0.8125, 0.796875, 0.78125, 1.0, 0.984375, 0.96875], [0.96875, 0.953125, 0.9375, 
              0.921875, 0.90625, 0.890625, 0.875, 0.859375, 0.84375, 0.828125, 0.8125, 0.796875, 0.78125, 1.0, 0.984375], 
             [0.984375, 0.96875, 0.953125, 0.9375, 0.921875, 0.90625, 0.890625, 0.875, 0.859375, 0.84375, 0.828125, 0.8125, 
              0.796875, 0.78125, 1.0]]
        
        B = [20.2011, 21.0487, 20.9129, 21.4044, 20.1037, 20.7351, 20.567, 20.457, 21.384, 20.985, 
             21.0467, 20.3129, 21.9044, 21.1037, 20.1351]
        
        n = 15
        summ = 0
        for i in range(n):
            sum_ = 0
            for j in range(n):
                sum_ += A[i][j] * x[j]
            sum_ -= B[i]
            summ += sum_ * sum_

        return summ

    
#_______________________ Вспомогательные классы и функции ______________________

class BestSolution(object):
    """
    Лучшее решение
    """
    
    def __init__(self, x=None, f=None, evals=None):
        self.x, self.f, self.evals = x, f, evals

    def update(self, x, f, evals=None):
        """
        Обновление лучшего решения если f < self.f
        """
        if self.f is None or f < self.f:
            self.x = x
            self.f = f
            self.evals = evals
        return self
    @property
    def all(self):
        return self.x, self.f, self.evals

class SquareMatrix(list): 
    """
    Класс квадратных матриц
    """
    def __init__(self, dimension):
        """
        Инициализация единичной матрицей
        """
        for i in range(dimension):
            self.append(dimension * [0])
            self[i][i] = 1

    def multiply_with(self, factor):
        """
        Умножение матрицы на скаляр
        """
        for row in self:
            for j in range(len(row)):
                row[j] *= factor
        return self

    def addouter(self, b, factor=1):
        """
        Прибавление вектора, умноженного на скаляр
        """
        for i, row in enumerate(self):
            for j in range(len(row)):
                row[j] += factor * b[i] * b[j]
        return self
    @property
    def diag(self):
        """
        Диагональ матрицы
        """
        return [self[i][i] for i in range(len(self)) if i < len(self[i])]

class DecomposingPositiveMatrix(SquareMatrix):
    """
    У симмитричной матрицы есть собственное разложение 

    Если isinstance(C, DecomposingPositiveMatrix),
    собственное разложение(значения с.в.) хранится в атрибутах `eigenbasis` и `eigenvalues` таким образом, 
    что i-й собственный вектор равен [row[i] for row in C.eigenbasis] с с.ч. и

    C = C.eigenbasis x diag(C.eigenvalues) x C.eigenbasis^T

    """
    def __init__(self, dimension):
        SquareMatrix.__init__(self, dimension)
        self.eigenbasis = eye(dimension)  # B - матрица из собственных векторов С
        self.eigenvalues = dimension * [1] #  диагональ матрицы D с собственыыми значениями С
        self.condition_number = 1 # число обусловленности
        self.invsqrt = eye(dimension) # C^(-1/2)
        self.updated_eval = 0 # число вычислений для обновлений 

    def update_eigensystem(self, current_eval, lazy_gap_evals):
        """
        Прервать композицию на собственные вектора, если
        current_eval > lazy_gap_evals + last_updated_eval.

        self должно быть положительно определённым
        """
        
        if current_eval <= self.updated_eval + lazy_gap_evals:
#             print('Ничего не изменилось, выходим')
            return self
#       Cимметризация
        self._enforce_symmetry()  
        self.eigenvalues, self.eigenbasis = np.linalg.eigh(self)  # O(N**3)

        if min(self.eigenvalues) <= 0:
            raise RuntimeError(
                "The smallest eigenvalue is <= 0 after %d evaluations!"
                "\neigenvectors:\n%s \neigenvalues:\n%s"
                % (current_eval, str(self.eigenbasis), str(self.eigenvalues)))
        self.condition_number = max(self.eigenvalues) / min(self.eigenvalues)

        # Вычисление  invsqrt(C) = C**(-1/2) = B D**(-1/2) B
        for i in range(len(self)):
            for j in range(i+1):
                self.invsqrt[i][j] = self.invsqrt[j][i] = sum(
                    self.eigenbasis[i][k] * self.eigenbasis[j][k]
                    / self.eigenvalues[k]**0.5 for k in range(len(self)))
                
        self.updated_eval = current_eval
        return self

    def mahalanobis_norm(self, dx):
        """
        dx^T * C^-1 * dx)**0.5
        """
        return sum(xi**2 for xi in dot(self.invsqrt, dx))**0.5

    def _enforce_symmetry(self):
        for i in range(len(self)):
            for j in range(i):
                self[i][j] = self[j][i] = (self[i][j] + self[j][i]) / 2
        return self

def eye(dimension):
    """
    Единичная матрица в формате list of list
    """
    m = [dimension * [0] for i in range(dimension)]
    for i in range(dimension):
        m[i][i] = 1
    return m

def dot(A, b, transpose=False):
    """ usual dot product of "matrix" A with "vector" b.

    ``A[i]`` is the i-th row of A. With ``transpose=True``, A transposed
    is used.
    """
    if not transpose:
        return [sum(A[i][j] * b[j] for j in range(len(b)))
                for i in range(len(A))]
    else:
        return [sum(A[j][i] * b[j] for j in range(len(b)))
                for i in range(len(A[0]))]

def plus(a, b):
    """
    Сложение векторов
    @return: a + b 
    """
    return [a[i] + b[i] for i in range(len(a))]

def minus(a, b):
    """
    Вычитание векторов
    @return: a - b 
    """
    return [a[i] - b[i] for i in range(len(a))]

def argsort(a):
    """
    @return: список индексов такой, чтобы получить отсортированный a, т.е.
    a[argsort(a)[i]] == sorted(a)[i]
    """
    return sorted(range(len(a)), key=a.__getitem__)  # a.__getitem__(i) = a[i]

def safe_str(s, known_words=None):
    """
    Строки в dict `known_words` заменяются их значениями
    для последующего вычисления с помощью eval.
    """
    
    safe_chars = ' 0123456789.,+-*()[]e<>='
    if s != str(s):
        return str(s)
    if not known_words:
        known_words = {}
    stest = s[:]  
    sret = s[:]  
    for word in sorted(known_words.keys(), key=len, reverse=True):
        stest = stest.replace(word, '  ')
        sret = sret.replace(word, " %s " % known_words[word])
    for c in stest:
        if c not in safe_chars:
            raise ValueError('"%s" is not a safe string'
                             ' (known words are %s)' % (s, str(known_words)))
    return sret


class RecombinationWeights(list):
    """
    Список уменьшающихся(рекомбинационных) значений веса w_i
    
    """
    
    def __init__(self, len_, exponent=1):
        """
        Сумма положительных и отрицательных весов сремится к 1 and -1, соответственно.
        Количество положительных весов = self.mu, около len_/2. Веса строго уменьшаются.


        @params len_: количество весов
        @return: list
            список весов
            Общая сумма = 0
            
        """
        weights = len_
        self.exponent = exponent  
        if exponent is None:
            self.exponent = 1 
        try:
            len_ = len(weights)
        except TypeError:
            try:  
                len_ = len(list(weights))
            except TypeError:  
                def signed_power(x, expo):
                    if expo == 1: return x
                    s = (x != 0) * (-1 if x < 0 else 1)
                    return s * math.fabs(x)**expo
                weights = [signed_power(log((len_ + 1) / 2.) - log(i), self.exponent)
                           for i in range(1, len_ + 1)]  # raw shape
        if len_ < 2:
            raise ValueError("number of weights must be >=2, was %d"
                             % (len_))
        self.debug = False

        list.__init__(self, weights)

        self.set_attributes_from_weights(do_asserts=False)
        sum_neg = sum(self[self.mu:])
        if sum_neg != 0:
            for i in range(self.mu, len(self)):
                self[i] /= -sum_neg
        self.do_asserts()
        self.finalized = False

    def __call__(self, lambda_):
        if lambda_ <= self.mu:
            return self[:lambda_]
        if lambda_ < self.lambda_:
            return self[:self.mu] + self[self.mu - lambda_:]
        if lambda_ > self.lambda_:
            return self[:self.mu] + (lambda_ - self.lambda_) * [0] + self[self.mu:]
        return self

    def set_attributes_from_weights(self, weights=None, do_asserts=True):
        if weights is not None:
            if not weights[0] > 0:
                raise ValueError(
                    "the first weight must be >0 but was %f" % weights[0])
            if weights[-1] > 0:
                raise ValueError(
                    "the last weight must be <=0 but was %f" %
                    weights[-1])
            self[:] = weights
        weights = self
        assert all(weights[i] >= weights[i+1]
                        for i in range(len(weights) - 1))
        self.mu = sum(w > 0 for w in weights)
        spos = sum(weights[:self.mu])
        assert spos > 0
        for i in range(len(self)):
            self[i] /= spos
            
        self.mueff = 1**2 / sum(w**2 for w in
                                   weights[:self.mu])
        sneg = sum(weights[self.mu:])
        assert (sneg - sum(w for w in weights if w < 0))**2 < 1e-11
        not do_asserts or self.do_asserts()
        return self

    def finalize_negative_weights(self, dimension, c1, cmu, pos_def=True):
    
        if dimension <= 0:
            raise ValueError("dimension must be larger than zero, was " +
                             str(dimension))
        self._c1 = c1  # for the record
        self._cmu = cmu

        if self[-1] < 0:
            if cmu > 0:
                if c1 > 10 * cmu:
                    print("""WARNING: c1/cmu = %f/%f seems to assume a
                    too large value for negative weights setting"""
                          % (c1, cmu))
                self._negative_weights_set_sum(1 + c1 / cmu)
                if pos_def:
                    self._negative_weights_limit_sum((1 - c1 - cmu) / cmu
                                                     / dimension)
            self._negative_weights_limit_sum(1 + 2 * self.mueffminus
                                             / (self.mueff + 2))
        self.do_asserts()
        self.finalized = True

        if self.debug:
            print("sum w = %.2f (final)" % sum(self))

    def zero_negative_weights(self):
        for k in range(len(self)):
            self[k] *= 0 if self[k] < 0 else 1
        self.finalized = True
        return self

    def _negative_weights_set_sum(self, value):

        weights = self 
        value = abs(value)  
        assert weights[self.mu] <= 0
        if not weights[-1] < 0:
            istart = max((self.mu, int(self.lambda_ / 2)))
            for i in range(istart, self.lambda_):
                weights[i] = -value / (self.lambda_ - istart)
        factor = abs(value / sum(weights[self.mu:]))
        for i in range(self.mu, self.lambda_):
            weights[i] *= factor
        assert 1 - value - 1e-5 < sum(weights) < 1 - value + 1e-5
        if self.debug:
            print("sum w = %.2f, sum w^- = %.2f" %
                  (sum(weights), -sum(weights[self.mu:])))

    def _negative_weights_limit_sum(self, value):
        weights = self  
        value = abs(value)  
        if sum(weights[self.mu:]) >= -value: 
            return  
        assert weights[-1] < 0 and weights[self.mu] <= 0
        factor = abs(value / sum(weights[self.mu:]))
        if factor < 1:
            for i in range(self.mu, self.lambda_):
                weights[i] *= factor
            if self.debug:
                print("sum w = %.2f (with correction %.2f)" %
                      (sum(weights), value))
        assert sum(weights) + 1e-5 >= 1 - value

    def do_asserts(self):
        weights = self
        assert 1 >= weights[0] > 0
        assert weights[-1] <= 0
        assert len(weights) == self.lambda_
        assert all(weights[i] >= weights[i+1]
                        for i in range(len(weights) - 1))  
        assert self.mu > 0 
        assert weights[self.mu-1] > 0 >= weights[self.mu]
        assert 0.999 < sum(w for w in weights[:self.mu]) < 1.001
        assert (self.mueff / 1.001 <
                sum(weights[:self.mu])**2 / sum(w**2 for w in weights[:self.mu]) <
                1.001 * self.mueff)
        assert (self.mueffminus == 0 == sum(weights[self.mu:]) or
                self.mueffminus / 1.001 <
                sum(weights[self.mu:])**2 / sum(w**2 for w in weights[self.mu:]) <
                1.001 * self.mueffminus)

    @property
    def lambda_(self):
        return len(self)
    @property
    def mueffminus(self):
        weights = self
        sneg = sum(weights[self.mu:])
        assert (sneg - sum(w for w in weights if w < 0))**2 < 1e-11
        return (0 if sneg == 0 else
                sneg**2 / sum(w**2 for w in weights[self.mu:]))
    @property
    def positive_weights(self):
        try:
            from numpy import asarray
            return asarray(self[:self.mu])
        except:
            return self[:self.mu]
    @property
    def asarray(self):
        from numpy import asarray
        return asarray(self)


# In[ ]:





# In[ ]:




