
# coding: utf-8

# # https://we.tl/t-oVH1BBORhA

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


movies_data=pd.read_csv('E:/ML_Codes/ml-1m/ml-1m/movies.dat',
                       sep='::',
                       header=None)


# In[3]:


ratings_data=pd.read_csv('E:/ML_Codes/ml-1m/ml-1m/ratings.dat',
                       sep='::',
                       header=None)


# In[4]:


movies_data.shape


# In[5]:


ratings_data.shape


# In[6]:


ratings_data.head()


# In[7]:


movies_data.head()


# In[8]:


movies_data.columns=['movie_id','title','genre']


# In[9]:


ratings_data.columns=['userid','movie_id','ratings','contact']


# In[17]:


movies_data['index']=movies_data.index


# In[23]:


movies_data.head()


# In[18]:


ratings_data.head()


# In[19]:


total=movies_data.merge(ratings_data,on='movie_id')


# In[20]:


total.drop(['contact','title','genre'],
           axis=1,inplace=True)


# In[21]:


usergroup=total.groupby('userid')


# In[22]:


#Format your data for RBM
#Amount of users to be used for trainings
noofUsers=1500
#training list
trz=[]
for userid,curuser in usergroup:
    #create a variable to store movies ratings
    rate=[0]*len(movies_data)
    for num,movie in curuser.iterrows():
        #store the normalized ratings
        rate[movie['index']]=movie['ratings']/5.0
        
    #Add this list of ratings into training list
    trz.append(rate)
    #check if all users data stored
    if noofUsers==0:
        break
    noofUsers -= 1


# In[29]:


len(trz[3])


# In[ ]:




