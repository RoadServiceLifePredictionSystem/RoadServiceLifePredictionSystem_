#!/usr/bin/env python
# coding: utf-8
# In[117]:


import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, svm
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from auxiliary_functions_and_classes import create_new_columns_according_to_the_specified_combinations

import matplotlib.pyplot as plt


# In[346]:


data = pd.read_csv('test_data.csv') # данные будут лежать в файле test_data.csv


# In[348]:


data.columns = np.arange(len(data.columns))


# In[334]:


# нормализация данных

for col in data.columns:
    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())


# In[ ]:


data_y = list(map(lambda x: [x], data.values[:,3])) # целевые значения y лежат в 3 столбце таблицы
data_x = data[data.columns.values[data.columns.values != 3]].values


# In[1]:


# Модель - случайный лес 

model_1 = ensemble.RandomForestRegressor(n_estimators=90)
cv_results_1 = cross_validate(model_1, data_x, data_y, cv=3)
mean_1 = np.mean(cv_results['test_score'])

print(mean_1)


# In[ ]:


# Модель - метод опорных векторов

model_2 = svm.SVR()
cv_results_2 = cross_validate(model_2, data_x, data_y, cv=3)
mean_2 = np.mean(cv_results['test_score'])

print(mean_2)


# In[ ]:


# Модель - Регрессия Байесовского хребта

model_3 = linear_model.BayesianRidge()
cv_results_3 = cross_validate(model_3, data_x, data_y, cv=3)
mean_3 = np.mean(cv_results['test_score'])

print(mean_3)


# In[ ]:


# Модель - Линейная регрессия

# возможность использовать полиномиальные модели заданной степени

# если хотим иметь все комбинации данной степени
poly_features = PolynomialFeatures(degree=2)
data_x_poly = poly_features.fit_transform(data_x)

# если хотим иметь определённые комбинации
new_columns = create_new_columns_according_to_the_specified_combinations(data_x, ['x1,x2,x3'])
data_x_poly = np.hstack([data_x, new_columns])

model_4 = linear_model.LinearRegression()
cv_results_4 = cross_validate(model_4, data_x_poly, data_y, cv=3)
mean_4 = np.mean(cv_results['test_score'])

print(mean_4)


# In[ ]:


# Модель - Хребет

model_5 = linear_model.Ridge()
cv_results_5 = cross_validate(model_5, data_x, data_y, cv=3)
mean_5 = np.mean(cv_results['test_score'])

print(mean_5)


# In[315]:


# model_1.fit(data_x_train, data_y_train)

