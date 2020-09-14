# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:14:31 2020

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data=pd.read_csv("D:\\karthik\\datasets\\suv_data.csv")
data.head(6)
data.info()

sns.countplot(x='Purchased',hue='Age',data=data)

data.isnull().sum()


x=df
y=df['Purchased']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1)
model=LogisticRegression()
model=model.fit(X_train,y_train)
pred=model.predict(X_test)
accuracy_score(y_test,pred)

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
