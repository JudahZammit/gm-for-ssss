from param import CLASSES,LATENT_DIM,RGB,TEMPERATURE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import losses

import numpy as np

import os
import math


# Fixes fatal error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class EncoderUnetModule(layers.Layer):
    
    def __init__(self,filterDim,dropout = 0.0,Batch_Norm=False):
        super(EncoderUnetModule,self).__init__()
        self.conv1 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                    padding='same')
        self.dropout = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')

        self.bn = layers.BatchNormalization()
        self.Batch_Norm = Batch_Norm     
    def call(self,inputs,training = False):
        x = self.conv1(inputs)
        if self.Batch_Norm:
            x = self.bn(x)
        if training:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.Batch_Norm:
            x = self.bn(x)
        return x




class DecoderUnetModule(layers.Layer):
    
    def __init__(self,filterDim,dropout = 0.0,Batch_Norm=False):
        super(DecoderUnetModule,self).__init__()
        
        self.deconv = layers.Conv2DTranspose(filterDim, (2, 2), strides=(2, 2), padding='same')
        self.concat = layers.Concatenate()
        self.conv1 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')
        self.dropout = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')
        
        self.bn = layers.BatchNormalization()
        self.Batch_Norm = Batch_Norm     
    
    def call(self,inputs,training = False):
        x = inputs[0]
        skip = inputs[1]
        x = self.deconv(x)
        x = self.concat([x,skip])
        x = self.conv1(x)
        if self.Batch_Norm:
            x = self.bn(x)
        if training:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.Batch_Norm:
            x = self.bn(x)
        return x


# In[9]:


class Unet(layers.Layer):
    
    def __init__(self,dropout = 0.0,Batch_Norm=False):
        super(Unet,self).__init__()
        
        self.pool = layers.MaxPooling2D((2, 2))
        
        self.encoder16 = EncoderUnetModule(16,dropout = dropout,Batch_Norm=Batch_Norm)
        self.encoder32 = EncoderUnetModule(32,dropout = dropout,Batch_Norm=Batch_Norm)
        self.encoder64 = EncoderUnetModule(64,dropout = dropout,Batch_Norm=Batch_Norm)
        self.encoder128 = EncoderUnetModule(128,dropout = dropout,Batch_Norm=Batch_Norm)
        self.encoder256 = EncoderUnetModule(256,dropout = dropout,Batch_Norm=Batch_Norm)
        
        self.decoder128 = DecoderUnetModule(128,dropout = dropout,Batch_Norm=Batch_Norm)
        self.decoder64 = DecoderUnetModule(64,dropout = dropout,Batch_Norm=Batch_Norm)
        self.decoder32 = DecoderUnetModule(32,dropout = dropout,Batch_Norm=Batch_Norm)
        self.decoder16 = DecoderUnetModule(16,dropout = dropout,Batch_Norm=Batch_Norm)
        
        
    def call(self,inputs):
        e16 = self.encoder16(inputs)
        p16 = self.pool(e16)
        e32 = self.encoder32(p16)
        p32 = self.pool(e32)
        e64 = self.encoder64(p32)
        p64 = self.pool(e64)
        e128 = self.encoder128(p64)
        p128 = self.pool(e128)
        e256 = self.encoder256(p128)
        
        d128 = self.decoder128([e256,e128])
        d64 = self.decoder64([d128,e64])
        d32 = self.decoder32([d64,e32])
        d16 = self.decoder16([d32,e16])
        
        return d16     


# In[10]:


class GaussianSampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(z_mean))
    z_sample = z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
    return z_sample


# In[11]:


class GaussianLL(layers.Layer):
    
    def call(self,inputs):
        x , mu, log_var = inputs
        x = tf.keras.layers.Flatten()(x)
        mu = tf.keras.layers.Flatten()(mu)
        log_var = tf.keras.layers.Flatten()(log_var)

        c = -.5 * math.log(2*math.pi)
        density = c - log_var/2 - ((x - mu)/(2*tf.keras.backend.exp(log_var) + 1e-8))*(x - mu)

        return tf.keras.backend.sum(density,axis = -1)


# In[12]:


class UnitGaussianLL(layers.Layer):
    
    def call(self,inputs):
        x = inputs
        x = tf.keras.layers.Flatten()(x)

        c = -.5 * math.log(2*math.pi)
        density = c - x**2/2

        return tf.keras.backend.sum(density,axis = -1)


# In[13]:


class IouCoef(layers.Layer):
    def call(self,inputs):
        y_true,y_pred = inputs
        smooth = 1
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_pred * y_true), axis=[1,2,3])
        union = tf.keras.backend.sum(y_pred,[1,2,3])+tf.keras.backend.sum(y_true,[1,2,3])-intersection
        iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou


# In[14]:

# This layer has no loss, it is required for you to add one after calling it
class q_y__x(layers.Layer):
    
    def __init__(self,Batch_Norm = False):
        super(q_y__x,self).__init__()
        self.unet = Unet(Batch_Norm = Batch_Norm)
        self.out = layers.Conv2D(CLASSES, (1, 1), activation='softmax')
        
    def call(self,inputs):
        supervised_image = inputs

        
        unet_output = self.unet(supervised_image)
        cat_parameters = self.out(unet_output)
         
        return cat_parameters


# In[15]:


class q_k__y(layers.Layer):
    
    def __init__(self,Batch_Norm = False):
        super(q_k__y,self).__init__()
        self.unet = Unet(Batch_Norm =Batch_Norm)
        self.log_var_out = layers.Conv2D(LATENT_DIM, (1, 1))
        self.mean_out = layers.Conv2D(LATENT_DIM, (1, 1))
        
        self.sampling = GaussianSampling()
        self.gaussianLL = GaussianLL()
        self.unitGaussianLL = UnitGaussianLL() 
        
    def call(self,inputs):
        supervised_mask = inputs
        
        unet_output = self.unet(supervised_mask)
        log_var = self.log_var_out(unet_output)
        mean = self.mean_out(unet_output)
        
        k_sample = self.sampling((mean,log_var))
        
        n_logp_k =  tf.reduce_mean(-self.unitGaussianLL(k_sample))
        self.add_loss(n_logp_k)
        
        logq_k__y = tf.reduce_mean(self.gaussianLL((k_sample,mean,log_var)))
        self.add_loss(logq_k__y)
        
        return k_sample


# In[16]:


class p_y__k(layers.Layer):
    
    def __init__(self,Batch_Norm = False):
        super(p_y__k,self).__init__()
        self.unet = Unet(Batch_Norm =Batch_Norm)
        self.out = layers.Conv2D(CLASSES, (1, 1), activation='softmax')
        self.cce = losses.CategoricalCrossentropy()
        
    def call(self,inputs):
        supervised_mask,k_sample = inputs

        unet_output = self.unet(k_sample)
        cat_parameters = self.out(unet_output)
       
        n_logp_y__k = tf.reduce_mean(-self.cce(supervised_mask, cat_parameters))
        self.add_loss(n_logp_y__k)
        
        return cat_parameters


# In[17]:


class q_z__y_x(layers.Layer):
    
    def __init__(self,Batch_Norm = False):
        super(q_z__y_x,self).__init__()
        self.unet = Unet(Batch_Norm =Batch_Norm)
        self.log_var_out = layers.Conv2D(LATENT_DIM, (1, 1))
        self.mean_out = layers.Conv2D(LATENT_DIM, (1, 1))
        
        self.sampling = GaussianSampling()
        self.gaussianLL = GaussianLL()
        self.unitGaussianLL = UnitGaussianLL()
        self.concat = layers.Concatenate()
        
    def call(self,inputs):
        supervised_mask,supervised_image = inputs
        
        concat = self.concat([supervised_mask,supervised_image])
        unet_output = self.unet(concat)
        log_var = self.log_var_out(unet_output)
        mean = self.mean_out(unet_output)
        
        z_sample = self.sampling((mean,log_var))
        
        n_logp_z =  tf.reduce_mean(-self.unitGaussianLL(z_sample))
        self.add_loss(n_logp_z)
        
        logq_z__x_y = tf.reduce_mean(self.gaussianLL((z_sample,mean,log_var)))
        self.add_loss(logq_z__x_y)
        
        return z_sample


# In[18]:


class p_x__y_z(layers.Layer):
    
    def __init__(self,Batch_Norm = False):
        super(p_x__y_z,self).__init__()
        self.unet = Unet(Batch_Norm =Batch_Norm)
        self.out = layers.Conv2D(RGB, (1, 1), activation='sigmoid')
        self.concat = layers.Concatenate()
        self.bce = losses.BinaryCrossentropy()
        
    def call(self,inputs):
        supervised_image,supervised_mask,z_sample = inputs

        concat = self.concat([supervised_image,supervised_mask,z_sample])
        unet_output = self.unet(concat)
        bern_parameters = self.out(unet_output)
       
        n_logp_x__y_z = tf.reduce_mean(self.bce(supervised_image, bern_parameters))
        self.add_loss(n_logp_x__y_z)
        
        return bern_parameters

# A function that generates samples from a set of categorical distributions 
# in a way that the gradient can propagate through.
class Gumbel(layers.Layer):
    def call(self,inputs):
        categorical = inputs
        gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(TEMPERATURE, probs=categorical)
        return gumbel_dist.sample()
