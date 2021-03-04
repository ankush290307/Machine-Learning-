#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
data = pd.read_csv('_salary_predict_dataset1.csv')
data= data.drop('Unnamed: 0',axis=1)
print(data)


# In[36]:


X=data.iloc[:,:-1]
Y=data.iloc[:,-1:]
print(X)
print(Y)




# In[37]:


x_train,x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
print(x_train)
print(y_train)


# In[39]:


model.fit(x_train,y_train)
pred= model.predict(x_test)
print(model.coef_)
print(model.intercept_)


# In[41]:


pred= model.predict(x_test)
print(pred)


# In[42]:


from sklearn.metrics import r2_score
score = r2_score(y_test,pred)
print(score)


# In[47]:


query = pd.read_csv('PredictQuery.csv')
query= query.drop('Salary',axis=1)
print(query)


# In[48]:


pred = model.predict(query)
print(pred)

