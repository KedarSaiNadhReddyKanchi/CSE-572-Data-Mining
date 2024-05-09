#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import import_ipynb
from sklearn.preprocessing import StandardScaler
import train
from train import features_Glucose
import pickle
from sklearn.decomposition import PCA
import pickle_compat
pickle_compat.patch()


# In[2]:


with open("RF_Model.pkl", 'rb') as file:
        GPC_Model = pickle.load(file) 
        test_df = pd.read_csv('test.csv', header=None)


# In[3]:


Features_CGM=features_Glucose(test_df)


# In[4]:


ss_fit = StandardScaler().fit_transform(Features_CGM)


# In[5]:


pca = PCA(n_components=5)


# In[6]:


pca_fit=pca.fit_transform(ss_fit)


# In[7]:


predictions = GPC_Model.predict(pca_fit)
print(predictions)


# In[8]:


pd.DataFrame(predictions).to_csv("Result.csv", header=None, index=False)

