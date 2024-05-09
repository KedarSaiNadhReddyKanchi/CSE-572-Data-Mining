#!/usr/bin/env python
# coding: utf-8

# In[426]:


import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft,rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from joblib import dump, load
from sklearn import svm


# In[427]:


insulin_patient_df=pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date','Time','BWZ Carb Input (grams)'])
cgm_patient_df=pd.read_csv('CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
insulin_patient_df['date_time_stamp']=pd.to_datetime(insulin_patient_df['Date'] + ' ' + insulin_patient_df['Time'])
cgm_patient_df['date_time_stamp']=pd.to_datetime(cgm_patient_df['Date'] + ' ' + cgm_patient_df['Time'])

insulin_patient_df_1=pd.read_csv('Insulin_patient2.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
cgm_patient_df_1=pd.read_csv('CGM_patient2.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
insulin_patient_df_1['date_time_stamp']=pd.to_datetime(insulin_patient_df_1['Date'] + ' ' + insulin_patient_df_1['Time'])
cgm_patient_df_1['date_time_stamp']=pd.to_datetime(cgm_patient_df_1['Date'] + ' ' + cgm_patient_df_1['Time'])


# In[428]:


def mealdatacreation(insulin_patient_df, cgm_patient_df, dateidentifier):
    insulin_df=insulin_patient_df.copy()
    insulin_df=insulin_df.set_index('date_time_stamp')
    
    # sort by date_time_stamp, 
    _2_5_hours_meal_timestamp_df=insulin_df.sort_values(by='date_time_stamp',ascending=True).dropna()
    _2_5_hours_meal_timestamp_df['BWZ Carb Input (grams)'].replace(0.0,np.nan,inplace=True)
    _2_5_hours_meal_timestamp_df=_2_5_hours_meal_timestamp_df.dropna().reset_index()
    _2_5_hours_meal_timestamp_df=_2_5_hours_meal_timestamp_df.reset_index().drop(columns='index')
    
    valid_list=[]
    time=0
    for idx,i in enumerate(_2_5_hours_meal_timestamp_df['date_time_stamp']):
        try:
            time=(_2_5_hours_meal_timestamp_df['date_time_stamp'][idx+1]-i).seconds / 60.0
            if time >= 120:
                valid_list.append(i)
        except KeyError:
            break
        # if idx == 2:
        #     break
    
    outList=[]
    if dateidentifier==1:
        for idx,i in enumerate(valid_list):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            get_date=i.date().strftime('%#m/%#d/%Y')
            outList.append(cgm_patient_df.loc[cgm_patient_df['Date']==get_date].set_index('date_time_stamp').between_time(start_time=start.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
    else:
        for idx,i in enumerate(valid_list):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            get_date=i.date().strftime('%Y-%m-%d')
            outList.append(cgm_patient_df.loc[cgm_patient_df['Date']==get_date].set_index('date_time_stamp').between_time(start_time=start.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
    return pd.DataFrame(outList)


# In[429]:


patient_meal_data=mealdatacreation(insulin_patient_df, cgm_patient_df, 1)
patient_meal_data=patient_meal_data.iloc[:,0:30]

patient_meal_data1=mealdatacreation(insulin_patient_df_1, cgm_patient_df_1, 2)
patient_meal_data1=patient_meal_data1.iloc[:,0:30]


# In[430]:


def nomealdatacreation(insulin_patient_df,cgm_patient_df):
    insulin_no_meal_df=insulin_patient_df.copy()
    test1_df=insulin_no_meal_df.sort_values(by='date_time_stamp',ascending=True).replace(0.0,np.nan).dropna().copy()
    test1_df=test1_df.reset_index().drop(columns='index')
    valid_timestamp=[]
    for idx,i in enumerate(test1_df['date_time_stamp']):
        try:
            time=(test1_df['date_time_stamp'][idx+1]-i).seconds//3600
            if time >=4:
                valid_timestamp.append(i)
        except KeyError:
            break
    dataset=[]
    for idx, i in enumerate(valid_timestamp):
        iteration_dataset=1
        try:
            all_nomeal=cgm_patient_df.loc[(cgm_patient_df['date_time_stamp']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_patient_df['date_time_stamp']<valid_timestamp[idx+1])]['Sensor Glucose (mg/dL)']
            total_length = len(all_nomeal)
            no_of_24_dataset=total_length//24
            while (iteration_dataset<=no_of_24_dataset):
                if iteration_dataset==1:
                    dataset.append(all_nomeal.iloc[:24].values.tolist())   
                else:
                    dataset.append(all_nomeal.iloc[(iteration_dataset-1)*24:(iteration_dataset)*24].values.tolist())
                iteration_dataset+=1
        except IndexError:
            break
    return pd.DataFrame(dataset)


# In[431]:


patient_nomeal_data=nomealdatacreation(insulin_patient_df, cgm_patient_df)
patient_nomeal_data1=nomealdatacreation(insulin_patient_df_1, cgm_patient_df_1)


# In[432]:


def mealfeaturematrixcreation(meal_data):
    index=meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    cleanedmealdata=meal_data.drop(meal_data.index[index]).reset_index().drop(columns='index')
    cleanedmealdata=cleanedmealdata.interpolate(method='linear',axis=1)
    index_to_drop=cleanedmealdata.isna().sum(axis=1).replace(0,np.nan).dropna().index
    cleanedmealdata=cleanedmealdata.drop(meal_data.index[index_to_drop]).reset_index().drop(columns='index')
    cleanedmealdata['tau_time']=abs(cleanedmealdata.iloc[:,0:30].idxmin(axis=1)-cleanedmealdata.iloc[:,0:30].idxmax(axis=1))*5
    cleanedmealdata['difference_in_glucose_normalized']=(cleanedmealdata.iloc[:,0:30].max(axis=1)-cleanedmealdata.iloc[:,0:25].min(axis=1))/(cleanedmealdata.iloc[:,0:30].max(axis=1))
    cleanedmealdata=cleanedmealdata.dropna().reset_index().drop(columns='index')
    power_first_max=[]
    index_first_max=[]
    power_second_max=[]
    index_second_max=[]
    for i in range(len(cleanedmealdata)):
        array=abs(rfft(cleanedmealdata.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(cleanedmealdata.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    meal_feature_matrix=pd.DataFrame()
    meal_feature_matrix['tau_time']=cleanedmealdata['tau_time']
    meal_feature_matrix['difference_in_glucose_normalized']=cleanedmealdata['difference_in_glucose_normalized']
    meal_feature_matrix['power_first_max']=power_first_max
    meal_feature_matrix['power_second_max']=power_second_max
    meal_feature_matrix['index_first_max']=index_first_max
    meal_feature_matrix['index_second_max']=index_second_max
    tm=cleanedmealdata.iloc[:,22:25].idxmin(axis=1)
    maximum=cleanedmealdata.iloc[:,5:19].idxmax(axis=1)
    list1=[]
    second_differential_data=[]
    standard_deviation=[]
    for i in range(len(cleanedmealdata)):
        list1.append(np.diff(cleanedmealdata.iloc[:,maximum[i]:tm[i]].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(cleanedmealdata.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(cleanedmealdata.iloc[i]))
    meal_feature_matrix['1stDifferential']=list1
    meal_feature_matrix['2ndDifferential']=second_differential_data
    return meal_feature_matrix


# In[433]:


meal_feature_matrix=mealfeaturematrixcreation(patient_meal_data)
meal_feature_matrix1=mealfeaturematrixcreation(patient_meal_data1)
meal_feature_matrix=pd.concat([meal_feature_matrix,meal_feature_matrix1]).reset_index().drop(columns='index')


# In[434]:


def createnomealfeaturematrix(non_meal_data):
    index_to_remove_non_meal=non_meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    non_meal_data_cleaned=non_meal_data.drop(non_meal_data.index[index_to_remove_non_meal]).reset_index().drop(columns='index')
    non_meal_data_cleaned=non_meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop=non_meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    non_meal_data_cleaned=non_meal_data_cleaned.drop(non_meal_data_cleaned.index[index_to_drop]).reset_index().drop(columns='index')
    non_meal_feature_matrix=pd.DataFrame()
    
    non_meal_data_cleaned['tau_time']=abs(non_meal_data_cleaned.iloc[:,0:24].idxmin(axis=1)-non_meal_data_cleaned.iloc[:,0:24].idxmax(axis=1))*5
    non_meal_data_cleaned['difference_in_glucose_normalized']=(non_meal_data_cleaned.iloc[:,0:24].max(axis=1)-non_meal_data_cleaned.iloc[:,0:24].min(axis=1))/(non_meal_data_cleaned.iloc[:,0:24].max(axis=1))
    power_first_max,index_first_max,power_second_max,index_second_max=[],[],[],[]
    for i in range(len(non_meal_data_cleaned)):
        array=abs(rfft(non_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(non_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    non_meal_feature_matrix['tau_time']=non_meal_data_cleaned['tau_time']
    non_meal_feature_matrix['difference_in_glucose_normalized']=non_meal_data_cleaned['difference_in_glucose_normalized']
    non_meal_feature_matrix['power_first_max']=power_first_max
    non_meal_feature_matrix['power_second_max']=power_second_max
    non_meal_feature_matrix['index_first_max']=index_first_max
    non_meal_feature_matrix['index_second_max']=index_second_max
    first_differential_data=[]
    second_differential_data=[]
    for i in range(len(non_meal_data_cleaned)):
        first_differential_data.append(np.diff(non_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(non_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist())).max())
    non_meal_feature_matrix['1stDifferential']=first_differential_data
    non_meal_feature_matrix['2ndDifferential']=second_differential_data
    return non_meal_feature_matrix


# In[435]:



non_meal_feature_matrix=createnomealfeaturematrix(patient_nomeal_data)
non_meal_feature_matrix1=createnomealfeaturematrix(patient_nomeal_data1)
non_meal_feature_matrix=pd.concat([non_meal_feature_matrix,non_meal_feature_matrix1]).reset_index().drop(columns='index')


# In[436]:


meal_feature_matrix['label']=1
non_meal_feature_matrix['label']=0
total_data=pd.concat([meal_feature_matrix,non_meal_feature_matrix]).reset_index().drop(columns='index')
dataset=shuffle(total_data,random_state=1).reset_index().drop(columns='index')
kfold = KFold(n_splits=10,shuffle=False)
principaldata=dataset.drop(columns='label')
scores_rf = []
model=DecisionTreeClassifier(criterion="entropy")
# model=svm.SVC(kernel="linear")
for train_index, test_index in kfold.split(principaldata):
    X_train,X_test,y_train,y_test = principaldata.loc[train_index],principaldata.loc[test_index],    dataset.label.loc[train_index],dataset.label.loc[test_index]
    model.fit(X_train,y_train)
    scores_rf.append(model.score(X_test,y_test))


# In[437]:


print('Prediction score is',np.mean(scores_rf)*100)


# In[438]:


classifier=DecisionTreeClassifier(criterion='entropy')
X, y= principaldata, dataset['label']
classifier.fit(X,y)
print(scores_rf)
dump(classifier, 'RandomForestClassifier.pickle')

