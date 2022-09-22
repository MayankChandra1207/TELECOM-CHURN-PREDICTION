#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("telecom churn.csv")


# In[3]:


df


# In[4]:


df['International plan'].replace(['No','Yes'],[0,1],inplace=True)


# In[6]:


df['Voice mail plan'].replace(['No','Yes'],[0,1],inplace=True)


# In[7]:


df


# In[18]:


df.info()


# In[19]:


df.describe()


# In[20]:


df['Churn'].value_counts()


# In[21]:


df.describe(include=['O'])


# In[23]:


corrlation= df.corr()


# In[24]:


plt.figure(figsize = (18,6))
sns.heatmap(corrlation,cmap="YlGnBu", annot = True)


# In[33]:


df['Churn'].value_counts(normalize=True)


# In[36]:


plt.figure(figsize=(18,6))
sns.countplot(x = 'State', data = df, hue = 'Churn')


# In[37]:


df.drop("State",axis=1)


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x = df.drop(["Churn"],axis=1)
y = df['Churn']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=2)


# In[46]:


logmodel = LogisticRegression()


# In[44]:


df.drop("State",axis=1)


# In[49]:


df1=df.drop("State",axis=1)


# In[50]:


df1


# In[51]:


x = df1.drop(["Churn"],axis=1)
y = df1['Churn']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=2)


# In[52]:


logmodel = LogisticRegression()


# In[53]:


logmodel.fit(x_train,y_train)


# In[54]:


prediction=logmodel.predict(x_test)


# In[55]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[56]:


confusion_matrix(y_test,prediction)


# In[57]:


accuracy_score(y_test,prediction)


# In[58]:


log_score=accuracy_score(y_test,prediction)*100
log_score


# In[60]:


from sklearn import metrics
logmodel_y_pred = logmodel.predict(x_test)
logmodel_cm = metrics.confusion_matrix(logmodel_y_pred, y_test)
sns.heatmap(logmodel_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')


# # KNN ALGORITHM

# In[61]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[62]:


sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


# In[63]:


knn=KNeighborsClassifier(n_neighbors=4)


# In[64]:


knn.fit(x_train,y_train)


# In[65]:


knn_score=knn.score(x_test,y_test)*100
knn_score


# In[69]:


knn_y_pred = knn.predict(x_test)
knn_cm = metrics.confusion_matrix(knn_y_pred, y_test)
sns.heatmap(knn_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('KNN')


# # COMPARING BOTH THE ALGORITHMS

# In[67]:


models = ['Logistic Regression' , 'KNN Algorithm']
model_data = [log_score , knn_score]
cols = ["accuracy_score"]
compare=pd.DataFrame(data=model_data , index= models , columns= cols)
compare.sort_values(ascending= False , by = ['accuracy_score'])


# In[68]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logmodel.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(x_test)[:,1])
rf_roc_auc = roc_auc_score(y_test, knn.predict(x_test))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, knn.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='KNN (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




