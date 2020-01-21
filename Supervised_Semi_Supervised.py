#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow_probability as tfp

import numpy as np

import albumentations as A
from PIL import Image

import os
import random
import math


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
LATENT_DIM = 1
BS = 8
TEMPERATURE = .1
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
x_s = tf.keras.layers.Input((SHAPE, SHAPE, RGB))
y_s = tf.keras.layers.Input((SHAPE,SHAPE, CLASSES))


# In[6]:


# Calculates the log liklihood of a point x under a gaussian distribution parameterized by mu and log_var
def gaussian_ll(args):
    x , mu, log_var = args
    x = tf.keras.layers.Flatten()(x)
    mu = tf.keras.layers.Flatten()(mu)
    log_var = tf.keras.layers.Flatten()(log_var)

    c = -.5 * math.log(2*math.pi)
    density = c - log_var/2 - ((x - mu)/(2*tf.keras.backend.exp(log_var) + 1e-8))*(x - mu)

    return tf.keras.backend.sum(density,axis = -1)

# Calculates the log liklihood of a point x under a unit gaussian distribution
def unit_gaussian_ll(args):
    x = args
    x = tf.keras.layers.Flatten()(x)

    c = -.5 * math.log(2*math.pi)
    density = c - x**2/2

    return tf.keras.backend.sum(density,axis = -1)


# In[7]:


# A function that generates samples from a set of categorical distributions 
# in a way that the gradient can propagate through.
def gumbel_softmax(args):
    ind_multinomial = args
    gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(TEMPERATURE, probs=ind_multinomial)
    return gumbel_dist.sample()

# A function for sampling from a gaussian distrubution
def gaussian_sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(z_mean))
    z_sample = z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
    return z_sample


# In[8]:


# q_s(y|x)

q_y__x_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(x_s)
#q_y__x_s_c1 = tf.keras.layers.Dropout(0.1)(q_y__x_s_c1)
q_y__x_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c1)
q_y__x_s_p1 = tf.keras.layers.MaxPooling2D((2, 2))(q_y__x_s_c1)
 
q_y__x_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_p1)
#q_y__x_s_c2 = tf.keras.layers.Dropout(0.1)(q_y__x_s_c2)
q_y__x_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c2)
q_y__x_s_p2 = tf.keras.layers.MaxPooling2D((2, 2))(q_y__x_s_c2)
 
q_y__x_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_p2)
#q_y__x_s_c3 = tf.keras.layers.Dropout(0.2)(q_y__x_s_c3)
q_y__x_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c3)
q_y__x_s_p3 = tf.keras.layers.MaxPooling2D((2, 2))(q_y__x_s_c3)
 
q_y__x_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_p3)
#q_y__x_c4 = tf.keras.layers.Dropout(0.2)(q_y__x_s_c4)
q_y__x_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c4)
q_y__x_s_p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(q_y__x_s_c4)
 
q_y__x_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_p4)
#q_y__x_s_c5 = tf.keras.layers.Dropout(0.3)(q_y__x_s_c5)
q_y__x_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c5)
 
q_y__x_s_u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(q_y__x_s_c5)
q_y__x_s_u6 = tf.keras.layers.concatenate([q_y__x_s_u6, q_y__x_s_c4])
q_y__x_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_u6)
#q_y__x_s_c6 = tf.keras.layers.Dropout(0.2)(q_y__x_s_c6)
q_y__x_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c6)
 
q_y__x_s_u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(q_y__x_s_c6)
q_y__x_s_u7 = tf.keras.layers.concatenate([q_y__x_s_u7, q_y__x_s_c3])
q_y__x_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_u7)
#q_y__x_s_c7 = tf.keras.layers.Dropout(0.2)(q_y__x_s_c7)
q_y__x_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c7)
 
q_y__x_s_u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(q_y__x_s_c7)
q_y__x_s_u8 = tf.keras.layers.concatenate([q_y__x_s_u8, q_y__x_s_c2])
q_y__x_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_u8)
#q_y__x_s_c8 = tf.keras.layers.Dropout(0.1)(q_y__x_s_c8)
q_y__x_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c8)
 
q_y__x_s_u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(q_y__x_s_c8)
q_y__x_s_u9 = tf.keras.layers.concatenate([q_y__x_s_u9, q_y__x_s_c1], axis=3)
q_y__x_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_u9)
#q_y__x_s_c9 = tf.keras.layers.Dropout(0.1)(q_y__x_s_c9)
q_y__x_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_y__x_s_c9)
 
q_y__x_s_parameters = tf.keras.layers.Conv2D(CLASSES, (1, 1), activation='softmax',name = 'q_y__x_s')(q_y__x_s_c9) 

def logq_y__x(y_true,y_pred):
    return -tf.keras.backend.categorical_crossentropy(y_s, q_y__x_s_parameters)


# In[9]:


# q_s(k|y)

q_k__y_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(y_s)
#q_k__y_s_c1 = tf.keras.layers.Dropout(0.1)(q_k__y_s_c1)
q_k__y_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c1)
q_k__y_s_p1 = tf.keras.layers.MaxPooling2D((2, 2))(q_k__y_s_c1)
 
q_k__y_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_p1)
#q_k__y_s_c2 = tf.keras.layers.Dropout(0.1)(q_k__y_s_c2)
q_k__y_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c2)
q_k__y_s_p2 = tf.keras.layers.MaxPooling2D((2, 2))(q_k__y_s_c2)
 
q_k__y_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_p2)
#q_k__y_s_c3 = tf.keras.layers.Dropout(0.2)(q_k__y_s_c3)
q_k__y_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c3)
q_k__y_s_p3 = tf.keras.layers.MaxPooling2D((2, 2))(q_k__y_s_c3)
 
q_k__y_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_p3)
#q_k__y_c4 = tf.keras.layers.Dropout(0.2)(q_k__y_s_c4)
q_k__y_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c4)
q_k__y_s_p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(q_k__y_s_c4)
 
q_k__y_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_p4)
#q_k__y_s_c5 = tf.keras.layers.Dropout(0.3)(q_k__y_s_c5)
q_k__y_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c5)
 
q_k__y_s_u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(q_k__y_s_c5)
q_k__y_s_u6 = tf.keras.layers.concatenate([q_k__y_s_u6, q_k__y_s_c4])
q_k__y_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_u6)
#q_k__y_s_c6 = tf.keras.layers.Dropout(0.2)(q_k__y_s_c6)
q_k__y_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c6)
 
q_k__y_s_u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(q_k__y_s_c6)
q_k__y_s_u7 = tf.keras.layers.concatenate([q_k__y_s_u7, q_k__y_s_c3])
q_k__y_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_u7)
#q_k__y_s_c7 = tf.keras.layers.Dropout(0.2)(q_k__y_s_c7)
q_k__y_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c7)
 
q_k__y_s_u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(q_k__y_s_c7)
q_k__y_s_u8 = tf.keras.layers.concatenate([q_k__y_s_u8, q_k__y_s_c2])
q_k__y_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_u8)
#q_k__y_s_c8 = tf.keras.layers.Dropout(0.1)(q_k__y_s_c8)
q_k__y_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c8)
 
q_k__y_s_u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(q_k__y_s_c8)
q_k__y_s_u9 = tf.keras.layers.concatenate([q_k__y_s_u9, q_k__y_s_c1], axis=3)
q_k__y_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_u9)
#q_k__y_s_c9 = tf.keras.layers.Dropout(0.1)(q_k__y_s_c9)
q_k__y_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_k__y_s_c9)
 
q_k__y_s_mean = tf.keras.layers.Conv2D(LATENT_DIM, (1, 1))(q_k__y_s_c9) 
q_k__y_s_log_var = tf.keras.layers.Conv2D(LATENT_DIM, (1, 1))(q_k__y_s_c9)

k_s_sample = tf.keras.layers.Lambda(gaussian_sampling,name = 'k_s_sample')([q_k__y_s_mean,q_k__y_s_log_var])

def n_logp_k(y_true,y_pred):
    return -unit_gaussian_ll(k_s_sample)

def logq_k__y(y_true,y_pred):
    return gaussian_ll([k_s_sample,q_k__y_s_mean,q_k__y_s_log_var])

def k_loss(y_true,y_pred):
    return n_logp_k(y_true,y_pred) + logq_k__y(y_true,y_pred)


# In[10]:


# p_s(y|k)

p_y__k_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(k_s_sample)
#p_y__k_s_c1 = tf.keras.layers.Dropout(0.1)(p_y__k_s_c1)
p_y__k_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c1)
p_y__k_s_p1 = tf.keras.layers.MaxPooling2D((2, 2))(p_y__k_s_c1)
 
p_y__k_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_p1)
#p_y__k_s_c2 = tf.keras.layers.Dropout(0.1)(p_y__k_s_c2)
p_y__k_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c2)
p_y__k_s_p2 = tf.keras.layers.MaxPooling2D((2, 2))(p_y__k_s_c2)
 
p_y__k_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_p2)
#p_y__k_s_c3 = tf.keras.layers.Dropout(0.2)(p_y__k_s_c3)
p_y__k_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c3)
p_y__k_s_p3 = tf.keras.layers.MaxPooling2D((2, 2))(p_y__k_s_c3)
 
p_y__k_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_p3)
#q_k__y_c4 = tf.keras.layers.Dropout(0.2)(p_y__k_s_c4)
p_y__k_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c4)
p_y__k_s_p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(p_y__k_s_c4)
 
p_y__k_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_p4)
#p_y__k_s_c5 = tf.keras.layers.Dropout(0.3)(p_y__k_s_c5)
p_y__k_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c5)
 
p_y__k_s_u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(p_y__k_s_c5)
p_y__k_s_u6 = tf.keras.layers.concatenate([p_y__k_s_u6, p_y__k_s_c4])
p_y__k_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_u6)
#p_y__k_s_c6 = tf.keras.layers.Dropout(0.2)(p_y__k_s_c6)
p_y__k_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c6)
 
p_y__k_s_u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(p_y__k_s_c6)
p_y__k_s_u7 = tf.keras.layers.concatenate([p_y__k_s_u7, p_y__k_s_c3])
p_y__k_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_u7)
#p_y__k_s_c7 = tf.keras.layers.Dropout(0.2)(p_y__k_s_c7)
p_y__k_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c7)
 
p_y__k_s_u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(p_y__k_s_c7)
p_y__k_s_u8 = tf.keras.layers.concatenate([p_y__k_s_u8, p_y__k_s_c2])
p_y__k_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_u8)
#p_y__k_s_c8 = tf.keras.layers.Dropout(0.1)(p_y__k_s_c8)
p_y__k_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c8)
 
p_y__k_s_u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(p_y__k_s_c8)
p_y__k_s_u9 = tf.keras.layers.concatenate([p_y__k_s_u9, p_y__k_s_c1], axis=3)
p_y__k_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_u9)
#p_y__k_s_c9 = tf.keras.layers.Dropout(0.1)(p_y__k_s_c9)
p_y__k_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_y__k_s_c9)
 
p_y__k_s_parameters = tf.keras.layers.Conv2D(CLASSES, (1, 1),activation = 'softmax',name = 'p_y__k_s')(p_y__k_s_c9) 

def n_logp_y__k(y_true,y_pred):
    return tf.keras.backend.categorical_crossentropy(y_s, p_y__k_s_parameters)


# In[11]:


# q_s(z|y,x)

y_s_x_s_concat = tf.keras.layers.Concatenate()([y_s,x_s])

q_z__y_x_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(y_s_x_s_concat)
#q_z__y_x_s_c1 = tf.keras.layers.Dropout(0.1)(q_z__y_x_s_c1)
q_z__y_x_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c1)
q_z__y_x_s_p1 = tf.keras.layers.MaxPooling2D((2, 2))(q_z__y_x_s_c1)
 
q_z__y_x_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_p1)
#q_z__y_x_s_c2 = tf.keras.layers.Dropout(0.1)(q_z__y_x_s_c2)
q_z__y_x_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c2)
q_z__y_x_s_p2 = tf.keras.layers.MaxPooling2D((2, 2))(q_z__y_x_s_c2)
 
q_z__y_x_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_p2)
#q_z__y_x_s_c3 = tf.keras.layers.Dropout(0.2)(q_z__y_x_s_c3)
q_z__y_x_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c3)
q_z__y_x_s_p3 = tf.keras.layers.MaxPooling2D((2, 2))(q_z__y_x_s_c3)
 
q_z__y_x_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_p3)
#q_k__y_c4 = tf.keras.layers.Dropout(0.2)(q_z__y_x_s_c4)
q_z__y_x_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c4)
q_z__y_x_s_p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(q_z__y_x_s_c4)
 
q_z__y_x_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_p4)
#q_z__y_x_s_c5 = tf.keras.layers.Dropout(0.3)(q_z__y_x_s_c5)
q_z__y_x_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c5)
 
q_z__y_x_s_u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(q_z__y_x_s_c5)
q_z__y_x_s_u6 = tf.keras.layers.concatenate([q_z__y_x_s_u6, q_z__y_x_s_c4])
q_z__y_x_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_u6)
#q_z__y_x_s_c6 = tf.keras.layers.Dropout(0.2)(q_z__y_x_s_c6)
q_z__y_x_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c6)
 
q_z__y_x_s_u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(q_z__y_x_s_c6)
q_z__y_x_s_u7 = tf.keras.layers.concatenate([q_z__y_x_s_u7, q_z__y_x_s_c3])
q_z__y_x_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_u7)
#q_z__y_x_s_c7 = tf.keras.layers.Dropout(0.2)(q_z__y_x_s_c7)
q_z__y_x_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c7)
 
q_z__y_x_s_u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(q_z__y_x_s_c7)
q_z__y_x_s_u8 = tf.keras.layers.concatenate([q_z__y_x_s_u8, q_z__y_x_s_c2])
q_z__y_x_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_u8)
#q_z__y_x_s_c8 = tf.keras.layers.Dropout(0.1)(q_z__y_x_s_c8)
q_z__y_x_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c8)
 
q_z__y_x_s_u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(q_z__y_x_s_c8)
q_z__y_x_s_u9 = tf.keras.layers.concatenate([q_z__y_x_s_u9, q_z__y_x_s_c1], axis=3)
q_z__y_x_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_u9)
#q_z__y_x_s_c9 = tf.keras.layers.Dropout(0.1)(q_z__y_x_s_c9)
q_z__y_x_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(q_z__y_x_s_c9)
 
q_z__y_x_s_mean = tf.keras.layers.Conv2D(LATENT_DIM, (1, 1))(q_z__y_x_s_c9) 
q_z__y_x_s_log_var = tf.keras.layers.Conv2D(LATENT_DIM, (1, 1))(q_z__y_x_s_c9)

z_s_sample = tf.keras.layers.Lambda(gaussian_sampling,name = 'z_s_sample')([q_z__y_x_s_mean,q_z__y_x_s_log_var])

def n_logp_z(y_true,y_pred):
    return -unit_gaussian_ll(z_s_sample)

def logq_z__x_y(y_true,y_pred):
    return gaussian_ll([z_s_sample,q_z__y_x_s_mean,q_z__y_x_s_log_var])

def z_loss(y_true,y_pred):
    return n_logp_z(y_true,y_pred) + logq_z__x_y(y_true,y_pred)


# In[12]:


# p_s(x|y,x)

y_s_z_s_sample_concat = tf.keras.layers.Concatenate()([y_s,z_s_sample])

p_x__y_z_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(y_s_z_s_sample_concat)
#p_x__y_z_s_c1 = tf.keras.layers.Dropout(0.1)(p_x__y_z_s_c1)
p_x__y_z_s_c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c1)
p_x__y_z_s_p1 = tf.keras.layers.MaxPooling2D((2, 2))(p_x__y_z_s_c1)
 
p_x__y_z_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_p1)
#p_x__y_z_s_c2 = tf.keras.layers.Dropout(0.1)(p_x__y_z_s_c2)
p_x__y_z_s_c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c2)
p_x__y_z_s_p2 = tf.keras.layers.MaxPooling2D((2, 2))(p_x__y_z_s_c2)
 
p_x__y_z_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_p2)
#p_x__y_z_s_c3 = tf.keras.layers.Dropout(0.2)(p_x__y_z_s_c3)
p_x__y_z_s_c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c3)
p_x__y_z_s_p3 = tf.keras.layers.MaxPooling2D((2, 2))(p_x__y_z_s_c3)
 
p_x__y_z_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_p3)
#q_k__y_c4 = tf.keras.layers.Dropout(0.2)(p_x__y_z_s_c4)
p_x__y_z_s_c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c4)
p_x__y_z_s_p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(p_x__y_z_s_c4)
 
p_x__y_z_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_p4)
#p_x__y_z_s_c5 = tf.keras.layers.Dropout(0.3)(p_x__y_z_s_c5)
p_x__y_z_s_c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c5)
 
p_x__y_z_s_u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(p_x__y_z_s_c5)
p_x__y_z_s_u6 = tf.keras.layers.concatenate([p_x__y_z_s_u6, p_x__y_z_s_c4])
p_x__y_z_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_u6)
#p_x__y_z_s_c6 = tf.keras.layers.Dropout(0.2)(p_x__y_z_s_c6)
p_x__y_z_s_c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c6)
 
p_x__y_z_s_u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(p_x__y_z_s_c6)
p_x__y_z_s_u7 = tf.keras.layers.concatenate([p_x__y_z_s_u7, p_x__y_z_s_c3])
p_x__y_z_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_u7)
#p_x__y_z_s_c7 = tf.keras.layers.Dropout(0.2)(p_x__y_z_s_c7)
p_x__y_z_s_c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c7)
 
p_x__y_z_s_u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(p_x__y_z_s_c7)
p_x__y_z_s_u8 = tf.keras.layers.concatenate([p_x__y_z_s_u8, p_x__y_z_s_c2])
p_x__y_z_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_u8)
#p_x__y_z_s_c8 = tf.keras.layers.Dropout(0.1)(p_x__y_z_s_c8)
p_x__y_z_s_c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c8)
 
p_x__y_z_s_u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(p_x__y_z_s_c8)
p_x__y_z_s_u9 = tf.keras.layers.concatenate([p_x__y_z_s_u9, p_x__y_z_s_c1], axis=3)
p_x__y_z_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_u9)
#p_x__y_z_s_c9 = tf.keras.layers.Dropout(0.1)(p_x__y_z_s_c9)
p_x__y_z_s_c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p_x__y_z_s_c9)
 
p_x__y_z_s_param = tf.keras.layers.Conv2D(RGB, (1, 1),activation = 'sigmoid',name = 'p_x__y_z_s')(p_x__y_z_s_c9) 

def n_logp_x__y_z(y_true,y_pred):
    return tf.keras.backend.binary_crossentropy(x_s, p_x__y_z_s_param)


# In[13]:


model = tf.keras.Model(inputs=[x_s,y_s], 
                       outputs=[z_s_sample,p_x__y_z_s_param,p_y__k_s_parameters,
                                k_s_sample,q_y__x_s_parameters])


# In[14]:


def dummy_bce1(y_true,y_pred):
    return tf.keras.losses.binary_crossentropy(y_s,q_y__x_s_parameters)

def dummy_bce2(y_true,y_pred):
    return 0.

def dummy_bce3(y_true,y_pred):
    return 0.

def dummy_bce4(y_true,y_pred):
    return 0.

def dummy_bce5(y_true,y_pred):
    return 0.


# In[15]:


# A function that calcuates the intersection over union couf.
def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_pred * y_true), axis=[1,2,3])
    union = tf.keras.backend.sum(y_pred,[1,2,3])+tf.keras.backend.sum(y_true,[1,2,3])-intersection
    iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

losses = {'z_s_sample': z_loss,
          'p_x__y_z_s':n_logp_x__y_z,
         'p_y__k_s': n_logp_y__k,
          'k_s_sample':k_loss,
          'q_y__x_s':logq_y__x}
dummy_boy = {'z_s_sample': dummy_bce1,
          'p_x__y_z_s':dummy_bce2,
         'p_y__k_s':dummy_bce3,
          'k_s_sample':dummy_bce4,
          'q_y__x_s':dummy_bce5}
model.compile('Adam',
             loss = dummy_boy)
model.summary()


# In[16]:


model = tf.keras.Model(inputs=[x_s,y_s], 
                       outputs=q_y__x_s_parameters)
model.compile('Adam',
             loss = dummy_bce1)
model.summary()


# In[22]:


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
        #return [X_s,Y_s],{'p_x__y_z_s':Y_s,'z_s_sample': Y_s,'p_y__k_s': Y_s,'k_s_sample':Y_s,'q_y__x_s':Y_s}
        return [X_s,Y_s], [Y_s]
    
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
            


# In[23]:


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
        return [X_s,Y_s],[Y_s,Y_s,Y_s,Y_s,Y_s]
        
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


# In[24]:


BS = 8


# In[25]:


train_gen = train_generator(batch_size = BS)
val_gen = val_generator(batch_size = BS)


# In[26]:


model.fit_generator(generator = train_gen,
                    steps_per_epoch = NUM_LABELED//BS,
                    epochs=100,
                   validation_data = val_gen,
                   validation_steps = NUM_UNLABELED//BS)


# In[ ]:




