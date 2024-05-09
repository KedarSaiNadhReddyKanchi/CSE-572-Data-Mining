#!/usr/bin/env python
# coding: utf-8

# In[62]:


# load data from csv file and save data into separate lists
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.fftpack import fft, ifft
from sklearn.decomposition import PCA
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
import pickle
import math

from datetime import timedelta
from scipy.fftpack import fft, ifft,rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from scipy.stats import entropy
from scipy.stats import iqr
from scipy import signal
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:


project3_insulin_csv_data_dataframe  =  pd.read_csv('InsulinData.csv' ,  low_memory  =  False ,  usecols  =  ['Date' , 'Time' , 'BWZ Carb Input (grams)'])
project3_cgm_csv_data_dataframe  =  pd.read_csv('CGMData.csv' , low_memory  =  False , usecols  =  ['Date' , 'Time' , 'Sensor Glucose (mg/dL)'])
project3_insulin_csv_data_dataframe['date_time_stamp']  =  pd.to_datetime(project3_insulin_csv_data_dataframe['Date']  +  ' '  +  project3_insulin_csv_data_dataframe['Time'])
project3_cgm_csv_data_dataframe['date_time_stamp']  =  pd.to_datetime(project3_cgm_csv_data_dataframe['Date']  +  ' '  +  project3_cgm_csv_data_dataframe['Time'])


def fucntion_for_meal_data_extraction(project3_insulin_csv_data_dataframe ,  project3_cgm_csv_data_dataframe ,  id_of_the_date):
    copy_of_the_project3_insulin_csv_data_dataframe  =  project3_insulin_csv_data_dataframe.copy()
    copy_of_the_project3_insulin_csv_data_dataframe  =  copy_of_the_project3_insulin_csv_data_dataframe.set_index('date_time_stamp')
    
    # sort by date_time_stamp ,  
    sort_insulin_data_by_time_stamp_and_find_timestamp_with_2_5_hours_df = copy_of_the_project3_insulin_csv_data_dataframe.sort_values(by = 'date_time_stamp' , ascending = True).dropna()
    sort_insulin_data_by_time_stamp_and_find_timestamp_with_2_5_hours_df['BWZ Carb Input (grams)'].replace(0.0 , np.nan , inplace = True)
    sort_insulin_data_by_time_stamp_and_find_timestamp_with_2_5_hours_df = sort_insulin_data_by_time_stamp_and_find_timestamp_with_2_5_hours_df.dropna().reset_index()
    sort_insulin_data_by_time_stamp_and_find_timestamp_with_2_5_hours_df = sort_insulin_data_by_time_stamp_and_find_timestamp_with_2_5_hours_df.reset_index().drop(columns = 'index')
    
    list_of_valid_time_stamps = []
    value = 0
    for idx , i in enumerate(sort_insulin_data_by_time_stamp_and_find_timestamp_with_2_5_hours_df['date_time_stamp']):
        try:
            value = (sort_insulin_data_by_time_stamp_and_find_timestamp_with_2_5_hours_df['date_time_stamp'][idx + 1] - i).seconds / 60.0
            if value >=  120:
                list_of_valid_time_stamps.append(i)
        except KeyError:
            break
        # if idx  =  =  2:
        #     break
    
    
    
    value_list_1  =  []
    value_list_2  =  []
    
    if id_of_the_date == 1:
        
        for idx , i in enumerate(list_of_valid_time_stamps):
            start = pd.to_datetime(i  -  timedelta(minutes = 30))
            end = pd.to_datetime(i  +  timedelta(minutes = 120))
            time  =  pd.to_datetime(i)
            get_date = i.date().strftime('%#m/%#d/%Y')
            x = (project3_insulin_csv_data_dataframe.loc[project3_insulin_csv_data_dataframe['Time'] == time.strftime('%H:%M:%S')]['BWZ Carb Input (grams)'].dropna().values.tolist())
            if(len(x) == 0):
                continue
            value_list_2.append(x)
            value_list_1.append(project3_cgm_csv_data_dataframe.loc[project3_cgm_csv_data_dataframe['Date'] == get_date].set_index('date_time_stamp').between_time(start_time = start.strftime('%H:%M:%S') , end_time = end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
    else:
        for idx , i in enumerate(list_of_valid_time_stamps):
            start = pd.to_datetime(i  -  timedelta(minutes = 30))
            end = pd.to_datetime(i  +  timedelta(minutes = 120))
            get_date = i.date().strftime('%Y - %m - %d')
            x = (project3_insulin_csv_data_dataframe.loc[project3_insulin_csv_data_dataframe['Time'] == time.strftime('%H:%M:%S')]['BWZ Carb Input (grams)'].dropna().values.tolist())
            print(x)
            if(len(x) == 0):
                continue
            value_list_2.append(x)
            value_list_1.append(project3_cgm_csv_data_dataframe.loc[project3_cgm_csv_data_dataframe['Date'] == get_date].set_index('date_time_stamp').between_time(start_time = start.strftime('%H:%M:%S') , end_time = end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
    
    return pd.DataFrame(value_list_1)  ,  pd.DataFrame(value_list_2)


# In[3]:


list_info_about_meal_data , list_info_about_amount_of_meal_data = fucntion_for_meal_data_extraction(project3_insulin_csv_data_dataframe , project3_cgm_csv_data_dataframe , 1)


# In[4]:


list_info_about_amount_of_meal_data = list_info_about_amount_of_meal_data.drop([1] , axis = 1)


# In[5]:


list_info_about_meal_data = list_info_about_meal_data.iloc[: , 0:30]


# In[6]:


list_info_about_meal_data  =  list_info_about_meal_data.values.tolist()


# In[7]:


list_info_about_amount_of_meal_data  =  list_info_about_amount_of_meal_data.values.tolist()


# In[8]:


def function_to_clean_data_and_sample_data(list_info_about_meal_data_to_be_cleaned_and_sampled , list_info_about_amount_of_meal_data_to_be_cleaned_and_sampled):
    idx  =  []
    meal_data_info_size  =  len(list_info_about_meal_data_to_be_cleaned_and_sampled)
    for i in range (meal_data_info_size):
        list_info_about_meal_data_to_be_cleaned_and_sampled[i]  =  list_info_about_meal_data_to_be_cleaned_and_sampled[i][:30]
        list_info_about_meal_data_to_be_cleaned_and_sampled[i]  =  list_info_about_meal_data_to_be_cleaned_and_sampled[i][:: - 1]
        if (len(list_info_about_meal_data_to_be_cleaned_and_sampled[i])!=  30):
            idx.append(i)
        elif 'nan' in list_info_about_meal_data_to_be_cleaned_and_sampled[i]:
            idx.append(i)      
    for j in range (len(idx) , 0 ,  - 1):
        del list_info_about_meal_data_to_be_cleaned_and_sampled[idx[j - 1]]
        del list_info_about_amount_of_meal_data_to_be_cleaned_and_sampled[idx[j - 1]]
    return list_info_about_meal_data_to_be_cleaned_and_sampled ,  list_info_about_amount_of_meal_data_to_be_cleaned_and_sampled


# In[9]:


def calculate_velocity(cleaned_and_sampled_list_info_about_meal_data):
    l = []
    for i in range(len(cleaned_and_sampled_list_info_about_meal_data) - 1):
        l.append(cleaned_and_sampled_list_info_about_meal_data[i + 1] - cleaned_and_sampled_list_info_about_meal_data[i])
    return l


# In[10]:


def calculate_acceleration(cleaned_and_sampled_list_info_about_meal_data):
    l = []
    for i in range(len(cleaned_and_sampled_list_info_about_meal_data) - 1):
        l.append(cleaned_and_sampled_list_info_about_meal_data[i + 1] - cleaned_and_sampled_list_info_about_meal_data[i])
    return l


# In[11]:


def calculate_maximum_velocity(cleaned_and_sampled_list_info_about_meal_data):
    return max(cleaned_and_sampled_list_info_about_meal_data)


# In[12]:


def calculate_minimum_velocity(cleaned_and_sampled_list_info_about_meal_data):
    return min(cleaned_and_sampled_list_info_about_meal_data)


# In[13]:


def calculate_average_velocity(cleaned_and_sampled_list_info_about_meal_data):
    return sum(cleaned_and_sampled_list_info_about_meal_data)/len(cleaned_and_sampled_list_info_about_meal_data)


# In[14]:


def calculate_maximum_acceleration(cleaned_and_sampled_list_info_about_meal_data):
    return max(cleaned_and_sampled_list_info_about_meal_data)


# In[15]:


def calculate_minimum_acceleration(cleaned_and_sampled_list_info_about_meal_data):
    return min(cleaned_and_sampled_list_info_about_meal_data)


# In[16]:


def calculate_average_acceleration(cleaned_and_sampled_list_info_about_meal_data):
    return sum(cleaned_and_sampled_list_info_about_meal_data)/len(cleaned_and_sampled_list_info_about_meal_data)  


# In[17]:


copy_of_the_list_info_about_meal_data , copy_of_the_list_info_about_amount_of_meal_data  =  list_info_about_meal_data  ,  list_info_about_amount_of_meal_data


# In[18]:


for i in range(len(copy_of_the_list_info_about_meal_data)):
    for j in range(len(copy_of_the_list_info_about_meal_data[0])):
        copy_of_the_list_info_about_meal_data[i][j]  =  str(copy_of_the_list_info_about_meal_data[i][j])


# In[19]:


copy_of_the_list_info_about_meal_data ,  copy_of_the_list_info_about_amount_of_meal_data  =  function_to_clean_data_and_sample_data(copy_of_the_list_info_about_meal_data ,  copy_of_the_list_info_about_amount_of_meal_data)


# In[20]:


print("Number of rows from the processed meal data: " , len(copy_of_the_list_info_about_meal_data) )


# In[21]:


print("Number of rows from the processed meal amount data: " , len(copy_of_the_list_info_about_amount_of_meal_data))


# In[22]:


def function_to_discretize_the_meal_amount_in_bins_of_size_20(copy_of_the_list_info_about_amount_of_meal_data):
    meal_amount_in_bins_of_size_20  =  []
    for i in range (len(copy_of_the_list_info_about_amount_of_meal_data)):
        if (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])>= 0) and (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])<= 20):
            meal_amount_in_bins_of_size_20.append(1)
        elif (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])>20) and (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])<= 40):
            meal_amount_in_bins_of_size_20.append(2)
        elif (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])>40) and (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])<= 60):
            meal_amount_in_bins_of_size_20.append(3)
        elif (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])>60) and (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])<= 80):
            meal_amount_in_bins_of_size_20.append(4)
        elif (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])>80) and (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])<= 100):
            meal_amount_in_bins_of_size_20.append(5)
        elif (int(copy_of_the_list_info_about_amount_of_meal_data[i][0])>100):
            meal_amount_in_bins_of_size_20.append(6)
    return meal_amount_in_bins_of_size_20


# In[23]:


list_containing_the_final_feature_matrix_for_the_meal_data  = []


# In[24]:


for i in range(len(copy_of_the_list_info_about_meal_data)):
    
    list_of_the_meal_data_as_an_floating_type_array  =  list(np.asarray(copy_of_the_list_info_about_meal_data[i] ,  dtype = np.float32))
    
    list_containing_velocities_for_each_sample_of_the_meal_data  =  calculate_velocity(list_of_the_meal_data_as_an_floating_type_array)
    
    calculated_maximum_velcoty_for_the_total_meal_data  =  calculate_maximum_velocity(list_containing_velocities_for_each_sample_of_the_meal_data)
    
    calculated_minimum_velcoty_for_the_total_meal_data  =  calculate_minimum_velocity(list_containing_velocities_for_each_sample_of_the_meal_data)
    
    calculated_average_velcoty_for_the_total_meal_data  =  calculate_average_velocity(list_containing_velocities_for_each_sample_of_the_meal_data)

    list_containing_accuracies_for_each_sample_of_the_meal_data  =  calculate_acceleration(list_containing_velocities_for_each_sample_of_the_meal_data)
    
    calculated_maximum_acceleration_for_the_total_meal_data  =  calculate_maximum_acceleration(list_containing_accuracies_for_each_sample_of_the_meal_data)
    
    calculated_minimum_acceleration_for_the_total_meal_data  =  calculate_minimum_acceleration(list_containing_accuracies_for_each_sample_of_the_meal_data)
    
    calculated_average_acceleration_for_the_total_meal_data  =  calculate_average_acceleration(list_containing_accuracies_for_each_sample_of_the_meal_data)
    
    calculated_entropy_for_the_total_meal_data  =  entropy(list_of_the_meal_data_as_an_floating_type_array , base = 2)
    
    calculated_irq_for_the_total_meal_data  =  iqr(list_of_the_meal_data_as_an_floating_type_array)
    
    f , power_spectral_density  =  signal.periodogram(list_of_the_meal_data_as_an_floating_type_array)
    
    power_spectral_density_1  =  sum(power_spectral_density[0:5])/5
    
    power_spectral_density_2  =  sum(power_spectral_density[5:10])/5
    
    power_spectral_density_3  =  sum(power_spectral_density[10:16])/6
    
    fast_fourier_transform_list =  list(np.fft.fft(list_of_the_meal_data_as_an_floating_type_array))
    
    fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns  =  []
    
    while len(fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns)<6:
        temp  =  max(fast_fourier_transform_list)
        mag  =  math.sqrt(np.real(temp)**2 + np.imag(temp)**2)
        fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns.append(mag)
        del fast_fourier_transform_list[fast_fourier_transform_list.index(temp)]
    
    f1  =  [calculated_maximum_velcoty_for_the_total_meal_data , calculated_minimum_velcoty_for_the_total_meal_data , calculated_average_velcoty_for_the_total_meal_data , calculated_maximum_acceleration_for_the_total_meal_data , calculated_minimum_acceleration_for_the_total_meal_data , calculated_average_acceleration_for_the_total_meal_data , calculated_entropy_for_the_total_meal_data , calculated_irq_for_the_total_meal_data , fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns[0] , fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns[1] , fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns[2] , fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns[3] , fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns[4] , fast_fourier_transform_list_containing_the_top_6_max_values_resulted_in_the_6_columns[5] , power_spectral_density_1 , power_spectral_density_2 , power_spectral_density_3]
    
    f1  =  np.asarray(f1 ,  dtype = object)
    
    if i  ==  0:
        list_containing_the_final_feature_matrix_for_the_meal_data  =  f1
    else:
        list_containing_the_final_feature_matrix_for_the_meal_data  =  np.vstack((list_containing_the_final_feature_matrix_for_the_meal_data , f1))


# In[25]:


for i in range(len(list_containing_the_final_feature_matrix_for_the_meal_data[0])):
    list_containing_the_final_feature_matrix_for_the_meal_data  =  normalize(list_containing_the_final_feature_matrix_for_the_meal_data ,  axis = 0 ,  norm = 'max')


# In[26]:


# print(pd.DataFrame(list_containing_the_final_feature_matrix_for_the_meal_data))


# In[27]:


with open('list_containing_the_final_feature_matrix_for_the_meal_data.pkl' , 'wb') as f:
    pickle.dump(list_containing_the_final_feature_matrix_for_the_meal_data ,  f)


# In[28]:


np.savetxt('test.csv' ,  pd.DataFrame(list_containing_the_final_feature_matrix_for_the_meal_data[:60]) ,  fmt = "%f" ,  delimiter = " , ")


# In[29]:


meal_amount_in_bins_of_size_20  =  function_to_discretize_the_meal_amount_in_bins_of_size_20(copy_of_the_list_info_about_amount_of_meal_data)


# In[31]:


print("Printing the list meal_amount_in_bins_of_size_20 containing the bin values for the rows of the meal data")


# In[30]:


print(meal_amount_in_bins_of_size_20)


# In[39]:


print("number of points in Bin 1 = " , meal_amount_in_bins_of_size_20.count(1))


# In[40]:


print("number of points in Bin 2 = " , meal_amount_in_bins_of_size_20.count(2))


# In[41]:


print("number of points in Bin 3 = " , meal_amount_in_bins_of_size_20.count(3))


# In[42]:


print("number of points in Bin 4 = " , meal_amount_in_bins_of_size_20.count(4))


# In[43]:


print("number of points in Bin 5 = " , meal_amount_in_bins_of_size_20.count(5))


# In[44]:


print("number of points in Bin 6 = " , meal_amount_in_bins_of_size_20.count(6))


# In[45]:


meal_amount_in_bins_of_size_20  =  np.asarray(meal_amount_in_bins_of_size_20)


# In[51]:


def get_BinsMG(result_labels ,  true_label ,  clusterNum):
    binResultMG  =  []
    generated_bins_by_the_kmeans_model  =  []
    for i in range(clusterNum):
        binResultMG.append([])
        generated_bins_by_the_kmeans_model.append([])
    for i in range(len(result_labels)):
        binResultMG[result_labels[i] - 1].append(i)
    # print(binResultMG)
    for i in range(clusterNum):
        for j in binResultMG[i]:
            generated_bins_by_the_kmeans_model[i].append(true_label[j])
    return generated_bins_by_the_kmeans_model


# In[52]:


def compute_SSEValueMG_for_kmeans_calculation(bin_value_and_bins_weighted_value):
    sse_value_mg_for_kmeans  =  0
    if len(bin_value_and_bins_weighted_value) !=  0:
        avg  =  sum(bin_value_and_bins_weighted_value) / len(bin_value_and_bins_weighted_value)
        for i in bin_value_and_bins_weighted_value:
            sse_value_mg_for_kmeans  +=  (i  -  avg) * (i  -  avg)
    return sse_value_mg_for_kmeans


# In[53]:


clusterNum = 6


# In[54]:


generating_kmeans_model_to_output_the_required_values  =  KMeans(n_clusters = clusterNum ,  random_state = 0).fit(list_containing_the_final_feature_matrix_for_the_meal_data)


# In[55]:


list_copy_of_the_meal_amount_in_bins_of_size_20  =  meal_amount_in_bins_of_size_20


# In[56]:


generated_bins_by_the_kmeans_model  =  get_BinsMG(generating_kmeans_model_to_output_the_required_values.labels_ ,  list_copy_of_the_meal_amount_in_bins_of_size_20 ,  clusterNum)


# In[57]:


print("generated_bins_by_the_kmeans_model")
print(generated_bins_by_the_kmeans_model)


# In[58]:


kMeansSSE  =  0


# In[60]:


for i in range(len(generated_bins_by_the_kmeans_model)):
    kMeansSSE  +=  (compute_SSEValueMG_for_kmeans_calculation(generated_bins_by_the_kmeans_model[i]) * len(generated_bins_by_the_kmeans_model[i]))


# In[63]:


generating_a_kmeans_contingency_matrix_to_calculate_the_values_required_for_the_output  =  contingency_matrix(list_copy_of_the_meal_amount_in_bins_of_size_20 ,  generating_kmeans_model_to_output_the_required_values.labels_)


# In[64]:


calculating_entropy_from_the_generated_kmeans_model_and_kmeans_contingency_matrix ,  calculating_purity_MG_from_the_generated_kmeans_model_and_kmeans_contingency_matrix  =  [] ,  []


# In[65]:


for single_kmeans_model_cluster in generating_a_kmeans_contingency_matrix_to_calculate_the_values_required_for_the_output:
    single_kmeans_model_cluster  =  single_kmeans_model_cluster / float(single_kmeans_model_cluster.sum())
    calculting_the_temporary_entropy_mg_value_to_calculate_the_final_entropy_value  =  0
    for x in single_kmeans_model_cluster :
        if x !=  0 :
            calculting_the_temporary_entropy_mg_value_to_calculate_the_final_entropy_value  =  (single_kmeans_model_cluster * [math.log(x ,  2)]).sum()* - 1
        else:
            calculting_the_temporary_entropy_mg_value_to_calculate_the_final_entropy_value  =  single_kmeans_model_cluster.sum()
    single_kmeans_model_cluster  =  single_kmeans_model_cluster*3.5
    calculating_entropy_from_the_generated_kmeans_model_and_kmeans_contingency_matrix  +=  [calculting_the_temporary_entropy_mg_value_to_calculate_the_final_entropy_value]
    calculating_purity_MG_from_the_generated_kmeans_model_and_kmeans_contingency_matrix  +=  [single_kmeans_model_cluster.max()]


# In[66]:


counts  =  np.array([c.sum() for c in generating_a_kmeans_contingency_matrix_to_calculate_the_values_required_for_the_output])


# In[67]:


coeffs  =  counts / float(counts.sum())


# In[68]:


calculated_final_kmeans_entropy_value  =  (coeffs * calculating_entropy_from_the_generated_kmeans_model_and_kmeans_contingency_matrix).sum()


# In[69]:


calculated_final_kMeanspurity_MG_value  =  (coeffs * calculating_purity_MG_from_the_generated_kmeans_model_and_kmeans_contingency_matrix).sum()


# In[70]:


generating_the_dbScanModel_to_calculate_the_sse_and_the_purity_values  =  DBSCAN(eps = 0.5 ,  min_samples = 2).fit(list_containing_the_final_feature_matrix_for_the_meal_data)


# In[72]:


print("Printing the labels generated within the generated DBScan Model")
print(generating_the_dbScanModel_to_calculate_the_sse_and_the_purity_values.labels_)


# In[73]:


generated_bins_by_the_kmeans_model  =  get_BinsMG(generating_the_dbScanModel_to_calculate_the_sse_and_the_purity_values.labels_ ,  list_copy_of_the_meal_amount_in_bins_of_size_20 ,  clusterNum)


# In[74]:


calculating_the_SSE_value_from_the_generated_DB_Scan_model  =  0


# In[75]:


for i in range(len(generated_bins_by_the_kmeans_model)):
     calculating_the_SSE_value_from_the_generated_DB_Scan_model  +=  (compute_SSEValueMG_for_kmeans_calculation(generated_bins_by_the_kmeans_model[i]) * len(generated_bins_by_the_kmeans_model[i]))


# In[76]:


generating_the_DB_Scan_Contingency_matrix_model  =  contingency_matrix(list_copy_of_the_meal_amount_in_bins_of_size_20 ,  generating_the_dbScanModel_to_calculate_the_sse_and_the_purity_values.labels_)


# In[77]:


calculating_entropy_from_the_generated_kmeans_model_and_kmeans_contingency_matrix ,  calculating_purity_MG_from_the_generated_kmeans_model_and_kmeans_contingency_matrix  =  [] ,  []


# In[78]:


for single_kmeans_model_cluster in generating_the_DB_Scan_Contingency_matrix_model:
    single_kmeans_model_cluster  =  single_kmeans_model_cluster / float(single_kmeans_model_cluster.sum())
    calculting_the_temporary_entropy_mg_value_to_calculate_the_final_entropy_value  =  0
    for x in single_kmeans_model_cluster :
        if x !=  0 :
            calculting_the_temporary_entropy_mg_value_to_calculate_the_final_entropy_value  =  (single_kmeans_model_cluster * [math.log(x ,  2)]).sum()* - 1
        else:
            calculting_the_temporary_entropy_mg_value_to_calculate_the_final_entropy_value  =  (single_kmeans_model_cluster * [math.log(x + 1 ,  2)]).sum()* - 1
    calculating_entropy_from_the_generated_kmeans_model_and_kmeans_contingency_matrix  +=  [calculting_the_temporary_entropy_mg_value_to_calculate_the_final_entropy_value]
    calculating_purity_MG_from_the_generated_kmeans_model_and_kmeans_contingency_matrix  +=  [single_kmeans_model_cluster.max()]


# In[79]:


counts  =  np.array([c.sum() for c in generating_a_kmeans_contingency_matrix_to_calculate_the_values_required_for_the_output])


# In[80]:


coeffs  =  counts / float(counts.sum())


# In[81]:


final_calculated_db_scan_model_entropy_value  =  (coeffs * calculating_entropy_from_the_generated_kmeans_model_and_kmeans_contingency_matrix).sum()


# In[82]:


final_calculated_db_scan_model_purity_mg_value  =  (coeffs * calculating_purity_MG_from_the_generated_kmeans_model_and_kmeans_contingency_matrix).sum()


# In[83]:


final_result_data_frame_containing_the_sse_entropy_and_purity_values_for_the_kmeans_and_the_dbscan_models = pd.DataFrame([kMeansSSE ,  calculating_the_SSE_value_from_the_generated_DB_Scan_model ,  calculated_final_kmeans_entropy_value ,  final_calculated_db_scan_model_entropy_value ,  calculated_final_kMeanspurity_MG_value ,  final_calculated_db_scan_model_purity_mg_value]).T


# In[85]:


final_result_data_frame_containing_the_sse_entropy_and_purity_values_for_the_kmeans_and_the_dbscan_models.to_csv('Results.csv' ,  header  =  False ,  index  =  False)

