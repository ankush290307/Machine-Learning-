#!/usr/bin/env python
# coding: utf-8

# In[30]:


#creating svm classifier 
import pandas as pd
import numpy as np
data=pd.read_csv('C:\\Users\\mr.ank\\Downloads\\data 1.csv')
print(data)
print(data.shape)


# In[36]:


d1=pd.get_dummies(data['diagnosis'],drop_first=True)
print(d1)


# In[37]:


data1=data.drop('diagnosis',axis=1)
print(data1)


# In[39]:


x=data1.iloc[:,2:32].values
y=data1.iloc[:,0:1].values
print(x.shape)
print(y.shape)


# In[40]:


data1.isnull().sum()#checking null value


# In[43]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
print(len(x_train))
print(len(x_test))


# In[57]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train)
print(x_test)


# In[63]:


#fit the svm model to traing dataset
from sklearn.svm import SVC
svm_model=SVC(kernel='rbf')
svm_model.fit(x_train,y_train)


# In[64]:


#prediction on test dataset
y_pred=svm_model.predict(x_test)
print(y_pred)


# In[65]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[66]:


#checking accuracy of the model
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)

