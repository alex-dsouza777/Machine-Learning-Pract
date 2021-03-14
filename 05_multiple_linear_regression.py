# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:59:44 2021

@author: Alex D'souza
"""

#Ass-5 Write a python program to implement Multiple Linear Regression for given dataset 

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Importing the dataset
dataset = pd.read_csv("salary_data1.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
#split the data in to training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)
x_new = [[5],[2],[1],[2]]
y_pred = regressor.predict(np.array(x_new).reshape(1, 4))
print(y_pred)
accuracy = (regressor.score(x_test,y_test))
print(accuracy)
