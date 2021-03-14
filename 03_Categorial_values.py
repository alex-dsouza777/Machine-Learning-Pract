# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:47:23 2021

@author: Alex D'souza
"""

#Ass-3. Write a python program to make Categorial values in numeric format for a given dataset 

#Importing Liabraries
import numpy as np            
import pandas as pd            
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler	
from sklearn.compose import ColumnTransformer	
#Importing DataSet
dataset = pd.read_csv('Data1.csv') 	
X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, -1].values 
# Encode categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
rg = ColumnTransformer([("Region", OneHotEncoder(), [0])], remainder = 'passthrough')
X = rg.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
#Splitting the data into Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
