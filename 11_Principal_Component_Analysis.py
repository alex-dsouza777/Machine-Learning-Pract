# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:21:31 2021

@author: Alex D'souza
"""

#Ass-11 Write a python Program to transform data with Principal Component Analysis(PCA)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Here we are using inbuilt dataset of scikit learn 
from sklearn.datasets import load_breast_cancer

# instantiating
cancer = load_breast_cancer()

cancer.keys()

# creating dataframe
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

# checking head of dataframe
df.head()

# Importing standardscalar module 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fitting
scaler.fit(df)

scaled_data = scaler.transform(df)

# Importing PCA 
from sklearn.decomposition import PCA

# Let's say, components = 2
pca = PCA(n_components=2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

scaled_data.shape

x_pca.shape

# giving a larger plot
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='rainbow')

# labeling x and y axes 
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

# components
pca.components_

map= pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
# plotting heatmap 
sns.heatmap(map,cmap='twilight')
