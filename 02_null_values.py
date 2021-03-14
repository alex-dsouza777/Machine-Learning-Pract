# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:44:53 2021

@author: Alex D'souza
"""

#Ass-2. Write a python program to find all null values in a given dataset and remove them.(Download dataset from Github.com)


#Importing Liabraries
import pandas as pd  
dataset = pd.read_csv('airquality11.csv')
dataset.isnull()
dataset.isnull().head(10)
dataset.isnull().sum()
dataset.isnull().head().sum()
modifieddataset=dataset.fillna(" ")
modifieddataset.isnull().sum()
#To remove rows with null values 
#dataset=dataset.dropna()
