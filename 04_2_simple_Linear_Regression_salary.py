# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:56:07 2021

@author: Alex D'souza
"""

#Write a python program to implement simple Linear Regression for predicting Employee Salary.

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('salary_data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# Visualizing the Training set results
viz_train = plt
viz_train.scatter(x_train, y_train, color='red')
viz_train.plot(x_train, regressor.predict(x_train), color='blue')
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()
# Visualizing the Test set results
viz_test = plt
viz_test.scatter(x_test, y_test, color='red')
viz_test.plot(x_train, regressor.predict(x_train), color='blue')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()


# Predicting the result of 5 Years Experience
y_pred=regressor.predict(np.array([5]).reshape(1, 1))
# Predicting the Test set results
y_pred = regressor.predict(x_test)

 