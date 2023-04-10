#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv('C:\\Users\\Neo Zhan\\Desktop\\ashrae-energy-prediction\\train.csv')
building_metadata = pd.read_csv('C:\\Users\\Neo Zhan\\Desktop\\ashrae-energy-prediction\\building_metadata.csv')


# In[3]:


weather_train = pd.read_csv('C:\\Users\\Neo Zhan\\Desktop\\ashrae-energy-prediction\\weather_train.csv')


# In[4]:


merged_df = pd.merge(weather_train, building_metadata, on='site_id', how='inner')


# In[5]:


Final_merged_df = pd.merge(merged_df, train, on=['building_id', 'timestamp'], how='inner')


# In[6]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[7]:


Final_merged_df['timestamp'] = pd.to_datetime(Final_merged_df['timestamp'])


# In[8]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[9]:


encoder.fit(Final_merged_df['primary_use'])


# In[10]:


encoded_col = encoder.transform(Final_merged_df['primary_use'])


# In[11]:


Final_merged_df['primary_use'] = encoded_col


# In[12]:


imp = IterativeImputer(max_iter=10, random_state=0)


# In[13]:


import numpy as np
epoch = np.datetime64('1970-01-01T00:00:00')


# In[14]:


float_array = (Final_merged_df['timestamp'] - epoch) / np.timedelta64(1, 's')


# In[15]:


Final_merged_df['timestamp']=float_array


# In[16]:


Final_merged_df_sample = Final_merged_df.sample(frac=0.2)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[18]:


X = Final_merged_df_sample[['timestamp', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'primary_use', 'square_feet','year_built', 'floor_count', 'meter']]


# In[19]:


y = Final_merged_df_sample['meter_reading']


# In[20]:


# Initialize the imputer MICE
imputer = IterativeImputer()

# Impute the missing values
X_imputed = pd.DataFrame(imputer.fit_transform(X))


# In[21]:


X_imputed.columns = X.columns


# In[22]:


X_imputed['primary_use'] = X_imputed['primary_use'].astype(int)
X_imputed['square_feet'] = X_imputed['square_feet'].astype('int64')
X_imputed['meter'] = X_imputed['meter'].astype('int64')


# In[23]:


X_imputed_train, X_imputed_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
