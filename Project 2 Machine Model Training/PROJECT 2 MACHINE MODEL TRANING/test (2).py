#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft,rfft
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from joblib import dump, load


# In[108]:


data=pd.read_csv('test.csv',header=None)


# In[109]:


def creatematrix(non_meal_data):
    index_to_remove=non_meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    clean_non_meal_data=non_meal_data.drop(non_meal_data.index[index_to_remove]).reset_index().drop(columns='index')
    clean_non_meal_data=clean_non_meal_data.interpolate(method='linear',axis=1)
    index_to_drop_again=clean_non_meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().index
    clean_non_meal_data=clean_non_meal_data.drop(clean_non_meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    non_meal_feature_matrix=pd.DataFrame()
    
    clean_non_meal_data['tau_time']=abs(clean_non_meal_data.iloc[:,0:24].idxmin(axis=1)-clean_non_meal_data.iloc[:,0:24].idxmax(axis=1))*5
    clean_non_meal_data['difference_in_glucose_normalized']=(clean_non_meal_data.iloc[:,0:24].max(axis=1)-clean_non_meal_data.iloc[:,0:24].min(axis=1))/(clean_non_meal_data.iloc[:,0:24].max(axis=1))
    power_first_max,index_first_max,power_second_max,index_second_max=[],[],[],[]
    for i in range(len(clean_non_meal_data)):
        array=abs(rfft(clean_non_meal_data.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(clean_non_meal_data.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    non_meal_feature_matrix['tau_time']=clean_non_meal_data['tau_time']
    non_meal_feature_matrix['difference_in_glucose_normalized']=clean_non_meal_data['difference_in_glucose_normalized']
    non_meal_feature_matrix['power_first_max']=power_first_max
    non_meal_feature_matrix['power_second_max']=power_second_max
    non_meal_feature_matrix['index_first_max']=index_first_max
    non_meal_feature_matrix['index_second_max']=index_second_max
    first_differential_data=[]
    second_differential_data=[]
    for i in range(len(clean_non_meal_data)):
        first_differential_data.append(np.diff(clean_non_meal_data.iloc[:,0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(clean_non_meal_data.iloc[:,0:24].iloc[i].tolist())).max())
    non_meal_feature_matrix['1stDifferential']=first_differential_data
    non_meal_feature_matrix['2ndDifferential']=second_differential_data
    return non_meal_feature_matrix


# In[110]:


dataset=creatematrix(data)


# In[111]:


from joblib import dump, load
with open('RandomForestClassifier.pickle', 'rb') as pre_trained:
    pickle_file = load(pre_trained)
    predict = pickle_file.predict(dataset)    
    pre_trained.close()


# In[1]:


print(predict)


# In[112]:


pd.DataFrame(predict).to_csv('Result.csv',index=False,header=False)

