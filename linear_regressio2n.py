# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:09:24 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("D:\\karthik\\datasets\\headbrain.csv")
data.info()
data.describe()


X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values


x_mean=np.mean(X)
y_mean=np.mean(Y)

n=len(X)

numer=0
denom=0
for i in range(n):
    numer+=(X[i]-x_mean)*(Y[i]-y_mean)
    denom+=(X[i]-x_mean)**2
    m=numer/denom
    c=y_mean-(m*x_mean)
    
    
print(m,c)

max_x=np.max(X)+100
min_x=np.min(X)-100


x=np.linspace(min_x,max_x,1000)
y=c+m*x

plt.plot(x,y)
plt.scatter(X,Y,color='green',label='points')
plt.show()
total1=0
total2=0

for i in range(n):
    yp=c+m*X[i]
    total1+=(Y[i]-yp)**2
    total2+=(Y[i]-y_mean)**2
r=1-(total2/total1)
print(r)











from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X=X.reshape((n,1))
model=LinearRegression()
model=model.fit(X,Y)
print(model)
y_pre=model.predict(X)
r2=model(X,Y)
print(r2)

plt.plot(X,y_pre)
plt.scatter(X,Y,color='red',label='points')
plt.show()



























