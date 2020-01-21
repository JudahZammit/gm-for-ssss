#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.losses import CategoricalCrossentropy

import numpy as np

import albumentations as A
from PIL import Image

import os
import random


# In[2]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[3]:


SHAPE = 64
RGB = 3
CLASSES = 21
BS = 32
NUM_UNLABELED = 14212
NUM_LABELED = 1456
NUM_VALIDAITON = 1457


# In[4]:


#Defines the augmentitations
def get_training_augmentation():
    train_transform = [
        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),     
        A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1)
    ]
    return A.Compose(test_transform)


# In[5]:


# Build U-Net model
inputs = tf.keras.layers.Input((SHAPE, SHAPE, RGB))

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(inputs)
#c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
 
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p1)
#c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p2)
#c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p3)
#c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p4)
#c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c5)
 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u6)
#c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u7)
#c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u8)
#c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u9)
#c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(21, (1, 1), activation='softmax')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])


# In[6]:


# A function that calcuates the intersection over union couf.
def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_pred * y_true), axis=[1,2,3])
    union = tf.keras.backend.sum(y_pred,[1,2,3])+tf.keras.backend.sum(y_true,[1,2,3])-intersection
    iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

model.compile('Adam',
             loss = 'categorical_crossentropy',
             metrics = [iou_coef])


# In[7]:


def train_generator(batch_size = 64,shape = (SHAPE,SHAPE)):

    # Loads in unlabeled images(file paths) and repeats the labeled images until they're
    # are more labeled ones then unlabeled ones     
    image_path_list = os.listdir('./VOCdevkit/VOC2012/train_frames/')
    unsupervised_path_list = os.listdir('./VOCdevkit/VOC2012/JPEGImages/')

    random.shuffle(image_path_list)
    random.shuffle(unsupervised_path_list)

    lis = os.listdir('./VOCdevkit/VOC2012/train_frames/')
    while len(image_path_list) <= len(unsupervised_path_list):
        random.shuffle(lis)
        image_path_list.extend(lis)

    X_s = np.zeros((batch_size, shape[1], shape[0], RGB), dtype='float32')
    Y_s = np.zeros((batch_size, shape[1], shape[0],CLASSES), dtype='float32')
    X_un = np.zeros((batch_size, shape[1], shape[0], RGB), dtype='float32')
        
    def getitem(i):
        n = 0
        
        for x in image_path_list[i*batch_size:(i+1)*batch_size]:
            
            image = np.array(Image.open('./VOCdevkit/VOC2012/train_frames/' + x))
            label = np.array(Image.open('./VOCdevkit/VOC2012/train_masks/' + x.replace('.jpg','.png')))

            sample = get_training_augmentation()(image=image, mask=label)
            image, label = sample['image']/255,sample['mask']
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)
            categorical_label = tf.keras.utils.to_categorical(label)

            X_s[n] = image
            #cat_label -> image
            Y_s[n] = categorical_label[:,:,0:CLASSES]
            n = n + 1
        
        n = 0
            
        for x in unsupervised_path_list[i*batch_size:(i+1)*batch_size]:
    
            image = np.array(Image.open('./VOCdevkit/VOC2012/JPEGImages/' + x))

            sample = get_training_augmentation()(image=image)
            image= sample['image']/255
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)

            X_un[n] = image
            n = n + 1

        #return [self.X_s,self.Y_s,self.X_un] , self.Y_s
        return [X_s],[Y_s]
    
    def on_epoch_end():
        random.shuffle(unsupervised_path_list)
        image_path_list = os.listdir('./VOCdevkit/VOC2012/train_frames/')
        lis = os.listdir('./VOCdevkit/VOC2012/train_frames/')
        while len(image_path_list) <= len(unsupervised_path_list):
            random.shuffle(lis)
            image_path_list.extend(lis) 
        
    i = -1 
    while True :
        if i < len(unsupervised_path_list) // batch_size:
            i = i + 1
        else: 
            on_epoch_end()
            i = 0
            
        yield getitem(i)
            


# In[8]:


def val_generator(batch_size = 64,shape = (SHAPE,SHAPE)):

    image_path_list = os.listdir('./VOCdevkit/VOC2012/val_frames/')
    random.shuffle(image_path_list)

    X_s = np.zeros((batch_size, shape[1], shape[0], 3), dtype='float32')
    Y_s = np.zeros((batch_size, shape[1], shape[0],21), dtype='float32')
    X_un = np.zeros((batch_size, shape[1], shape[0], 3), dtype='float32')
        
    def getitem(i):
        n = 0
        
        for x in image_path_list[i*batch_size:(i+1)*batch_size]:
            
            image = np.array(Image.open('./VOCdevkit/VOC2012/val_frames/' + x))
            label = np.array(Image.open('./VOCdevkit/VOC2012/val_masks/' + x.replace('.jpg','.png')))

            sample = get_validation_augmentation()(image=image, mask=label)
            image, label = sample['image']/255, sample['mask']
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)

            categorical_label = tf.keras.utils.to_categorical(label)

            X_s[n] = image
            #cat_label -> image
            Y_s[n] = categorical_label[:,:,0:CLASSES]
            n = n + 1

        #return [self.X_s, self.Y_s, self.X_un]
        return [X_s],[Y_s]
        
    def on_epoch_end():
        random.shuffle(image_path_list)    
    i = -1 
    while True:
        if i  < len(image_path_list) // batch_size :
            i = i + 1
        else:
            on_epoch_end()
            i = 0
        yield getitem(i)


# In[9]:


train_gen = train_generator(batch_size = BS)
val_gen = val_generator(batch_size = BS)


# In[ ]:


model.fit_generator(generator = train_gen,
                    steps_per_epoch = NUM_LABELED//BS,
                    epochs=100,
                   validation_data = val_gen,
                   validation_steps = NUM_VALIDAITON//BS)


# In[ ]:




