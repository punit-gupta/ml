
# coding: utf-8

# In[1]:


import cv2
import os
from random import shuffle
from tqdm import tqdm
import numpy as np


# In[2]:


#pip install tflearn
TRAIN_DIR = 'E:/ML_Codes/all1/train'
TEST_DIR = 'E:/ML_Codes/all1/test'


# In[3]:


def label_img(img):
    word_label = img.split('.')[0]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [0,1]
    #                             [no cat, very dog]
    elif word_label == 'dog': return [1,0]


# In[4]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50,50))
        training_data.append([np.array(img),np.array(label)])
        
    shuffle(training_data)
    #np.save('train_data.npy', training_data)
    return training_data


# In[5]:


train_data = create_train_data()


# In[6]:


#conda install tflearn or pip install tflearn
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,fully_connected,dropout
from tflearn.layers.estimator import regression


# In[7]:


model=input_data(shape=[None,50,50,1], name='input')


# In[8]:


#first convolution Layer
model=conv_2d(model,46,5,activation='relu')
model=max_pool_2d(model,5)

#second Convolution Layer
model=conv_2d(model,42,5,activation='relu')
model=max_pool_2d(model,5)

#third Convolution Layer
model=conv_2d(model,38,5,activation='relu')
model=max_pool_2d(model,5)

#fully Connected Layer
model=fully_connected(model,1024,activation='relu')
model=dropout(model,0.7)

model=fully_connected(model,2,activation='softmax')


# In[9]:


model=regression(model,
                optimizer='adam',
                loss='categorical_crossentropy',
                learning_rate=0.003,
                name='output')


# In[10]:


cnnmodel=tflearn.DNN(model)


# In[11]:


train=train_data[:-500]  #24500 for training
test=train_data[-500:]   #500 for testing


# In[12]:


X=np.array([i[0] for i in train]).reshape(-1,50,50,1)
Y=[i[1] for i in train] 


# In[13]:


test_x=np.array([i[0] for i in test]).reshape(-1,50,50,1)
test_y=[i[1] for i in test] 


# In[ ]:


cnnmodel.fit({'input':X},{'output':Y}, n_epoch=10,
            validation_set=({'input':test_x},{'output':test_y}),
            show_metric=True)


# In[ ]:




