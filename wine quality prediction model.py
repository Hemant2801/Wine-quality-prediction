#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# # Data-collection

# In[2]:


#loading the datset
df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Wine qualiy prediction/winequality-red.csv')


# In[3]:


#check the first 5 row
df.head()


# In[4]:


#to check the info of the dataset
df.info()


# In[5]:


#shape of the dataset
df.shape


# In[6]:


#statistical measure of the dataset
df.describe()


# In[7]:


#checking for missing values in the dataset
df.isnull().sum()


# # Data visualization

# In[8]:


df['quality'].value_counts()


# In[9]:


sns.catplot(x = 'quality', data = df, kind = 'count')


# In[10]:


# volatile acidity and quality
plot = plt.figure(figsize = (5, 5))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df )


# In[11]:


# citric acid and quality
plot = plt.figure(figsize = (5, 5))
sns.barplot(x = 'quality', y = 'citric acid', data = df )


# correlation between the data columns

# In[12]:


correlation = df.corr()


# In[13]:


#constructing a heatmap to undersatnd the correlation
plt.figure(figsize = (10, 10))
sns.heatmap(correlation, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws = {'size':8}, cmap = 'Blues')


# # Data preprocessing

# In[14]:


#separating the data and label
X = df.drop('quality', axis = 1)


# In[15]:


#label binarization
Y = df['quality'].apply(lambda y_values : 1 if y_values >= 7 else 0)


# In[16]:


Y.value_counts()


# In[17]:


#splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .3, stratify = Y, random_state = 45)


# Model training
# 
# Random tree classifier

# In[18]:


model = RandomForestClassifier()


# In[19]:


model.fit(x_train, y_train)


# Model evaluation
# 
# accuracy score

# In[20]:


#model evaluatio on tarining data
training_prediction = model.predict(x_train)

training_accuracy = accuracy_score(training_prediction, y_train)

print('THE ACCURACY OF THE TRAINING MODEL IS :', training_accuracy)


# In[21]:


#model evaluation on testing data
testing_prediction = model.predict(x_test)

testing_accuracy = accuracy_score(testing_prediction, y_test)

print('THE ACCURACY OF THE TRAINING MODEL IS :', testing_accuracy)


# # Building a predictive system

# In[24]:


input_data = input()
input_list = [float(i) for i in input_data.split(',')]

input_array = np.asarray(input_list)
reshaped_array = input_array.reshape(1, -1)

prediction = model.predict(reshaped_array)
print('THE PREDICTION IS :', prediction)
if prediction == 1:
    print('THE WINE IS BEST QUALITY')
else:
    print('THE WINE IS SO SO')

