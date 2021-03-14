# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:13:22 2021

@author: Alex D'souza
"""

#Ass-8 Write a python program to implement Decision Tree whether or not to play Tennis

#Importing Libraries
import numpy as np
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv("play_tennis.csv")

#Encode Categorial Data
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()

dataset['outlook'] = Le.fit_transform(dataset['outlook'])
dataset['temp'] = Le.fit_transform(dataset['temp'])
dataset['humidity'] = Le.fit_transform(dataset['humidity'])
dataset['windy'] = Le.fit_transform(dataset['windy'])
dataset['play'] = Le.fit_transform(dataset['play'])

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Fitting Decision Tree Model
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, y)

# We can visualize the tree using tree.plot_tree
tree.plot_tree(clf)

# The predictions are stored in X_pred
X_pred = clf.predict(X)

# verifying if the model has predicted it all right.
X_pred == y
