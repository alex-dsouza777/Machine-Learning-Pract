# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:40:38 2021

@author: Alex D'souza
"""

#Ass-1.Python program to prepare Scatter Plot for Iris Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris=pd.read_csv("iris.csv")
iris.plot(kind='scatter',x='sepal_length',y='sepal_width')
iris.plot(kind='scatter',x='petal_length',y='petal_width')


sns.set_style("whitegrid");
sns.FacetGrid(iris,hue="species",size=4).map(plt.scatter,"sepal_length","sepal_width").add_legend()
sns.FacetGrid(iris,hue="species",size=4).map(plt.scatter,"petal_length","petal_width").add_legend()
