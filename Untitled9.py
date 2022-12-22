#!/usr/bin/env python
# coding: utf-8

# In[1576]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1577]:


dat=pd.read_csv('titanic.csv')


# In[1578]:


dat.head()


# In[1579]:


dat.isnull()


# In[1580]:


dat.corr()


# In[1581]:


sns.heatmap(dat.isnull(),cbar=False,cmap='viridis')


# In[1582]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=dat)


# In[1583]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=dat)


# In[1584]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=dat,palette='flare')


# In[1585]:


sns.distplot(dat['Age'].dropna(),kde=True,color='darkred',bins=20)


# In[1586]:


sns.boxplot(x='Pclass',y='Age',data=dat)


# In[1587]:


def impute_age(cols):
    Age=[0]
    Pclass=[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[1588]:


dat['Age']=dat[['Age','Pclass']].apply(impute_age,axis=1)


# In[1589]:


sns.heatmap(dat.isnull(),cbar=False,cmap='viridis')


# In[1590]:


dat.loc[:"Cabin"]


# In[1591]:


dat.info()


# In[1592]:


embark = pd.get_dummies(dat['Embarked'],drop_first='True')
sex = pd.get_dummies(dat['Sex'],drop_first='True')


# In[1593]:


dat.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[1594]:


dat=pd.concat([dat,sex,embark],axis=1)


# In[1595]:


dat.info()


# In[1596]:


dat['Cabin'].fillna(0,inplace=True)


# In[1597]:


dat.drop(['Cabin'],axis=1,inplace=True)


# In[1598]:


data=pd.get_dummies((dat))


# In[ ]:


train=dat[0:7999]
test=dat[8000:]


# In[ ]:


dat.info()


# In[ ]:


x_train=train.drop(['Age','Fare'],axis=1)


# In[ ]:


y_train=train.drop('Survived',axis=1)


# In[ ]:


x_test=train.drop('Survived',axis=1)


# In[ ]:


true_p=train['Survived']


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lreg=LinearRegression()


# In[ ]:


x_train=pd.get_dummies(x_train)


# In[ ]:


x_test=pd.get_dummies(x_train)


# In[ ]:


x_train.fillna(0,inplace=True)
x_test.fillna(0,inplace=True)


# In[ ]:


lreg.fit(x_train,y_train)


# In[ ]:


pred=lreg.predict(x_test)


# In[ ]:


lreg.score(x_test,true_p)
lreg.score(x_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:




