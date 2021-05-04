#!/usr/bin/env python
# coding: utf-8

# In[20]:


import io
import time
#import graphviz 
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import svc
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets, metrics, preprocessing
from sklearn.model_selection import GridSearchCV, learning_curve
#from keras.models import Sequential
#from keras.optimizers import rmsprop
#from keras.layers import Dense, Activation

import warnings
warnings.filterwarnings("ignore")



data=pd.read_csv('Todas_las_ligas.csv',sep=';')

data.info()


# In[12]:


#preprocessing data



sns.countplot(data['FTR'])


# In[14]:


features = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR']
y_target = 'FTR'


# Drop rows with nan
data = data.dropna()

for col in ['HomeTeam', 'AwayTeam', 'Season', 'HTR']:

    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

x = data[features]
y = data['FTR']

# x = bundesliga.drop('FTR',axis=1)
# y = bundesliga['FTR']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[15]:


#RANDOM FOREST CLASSIFIER
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(x_train, y_train) 
pred_rfc=rfc.predict(x_test)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# In[16]:


#SVM CLASSIFIER
clf=svm.SVC()
clf.fit(x_train, y_train )
pred_clf= clf.predict(x_test)
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))


# In[19]:


#Neural Network
mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(x_train, y_train)
pred_mlpc= mlpc.predict(x_test)
print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))


# In[ ]:





# In[ ]:




