# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 22:56:44 2020

@author: mrank
"""

# -*- coding: utf-8 -*-
"""
@author: TIET
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('Student_selection_dataset.csv')
print(data)

d1 = pd.get_dummies(data['Selection'], drop_first=True)
print(d1)

data = pd.concat([data,d1],axis=1)
print(data)

data = data.drop('Selection',axis=1)
print(data)

X=data.iloc[:,:-1]
Y=data.iloc[:,-1:]
print(X)
print(Y)

model = LinearRegression()
x_train,x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
print(x_train)

print(y_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(pred)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred))
p= model.predict_proba(x_test)[:,:]
print(p)

query=pd.read_csv('QueryStudentSelection.csv')
print(query)
pred = model.predict(query)
print(pred)

