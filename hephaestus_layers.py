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
likelihoodfrom tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# This layer has no loss, it is required for you to add one after calling it
class q_y__x_a(layers.Layer):
    
    def __init__(self,Batch_Norm = False):
        super(q_y__x_a,self).__init__()
        self.unet = Unet(Batch_Norm = Batch_Norm)
        self.out = layers.Conv2D(CLASSES, (1, 1), activation='softmax')
        self.concat = layers.Concatenate()
        
    def call(self,inputs):
        supervised_image,a_sample = inputs
        concat = self.concat([supervised_image,a_sample])
        
        unet_output = self.unet(concat)
        cat_parameters = self.out(unet_output)
         
        return cat_parameters


class q_k__y(layers.Layer):
    
    def __init__(self,Batch_Norm = False,sampling = "gaussian"):
        super(q_k__y,self).__init__()
        self.unet = Unet(Batch_Norm =Batch_Norm)
        self.log_var_out = layers.Conv2D(LATENT_DIM, (1, 1))
        self.mean_out = layers.Conv2D(LATENT_DIM, (1, 1))
        
        if sampling == "gaussian":
            self.sampling = GaussianSampling()

        if sampling == "point":
            self.sampling = PointSampling()


        self.gaussianLL = GaussianLL()
        self.unitGaussianLL = UnitGaussianLL() 
        
    def call(self,inputs):
        supervised_mask = inputs
        
        unet_output = self.unet(supervised_mask)
        log_var = self.log_var_out(unet_output)
        mean = self.mean_out(unet_output)
        
        k_sample = self.sampling((mean,log_var))
        
        if sampling == "point":
            n_logp_k =  tf.reduce_mean(-self.unitGaussianLL(k_sample))
            self.add_loss(n_logp_k)
        
        logq_k__y = tf.reduce_mean(self.gaussianLL((k_sample,mean,log_var)))
        if sampling == "gaussian":
            kl_loss =  - 0.5 * tf.reduce_mean(
                log_var - tf.square(mean) - tf.exp(log_var) + 1) 
            self.add_loss(kl_loss)

        return k_sample

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

class q_z__y_x_a(layers.Layer):
    
    def __init__(self,Batch_Norm = False,sampling = "gaussian"):
        super(q_z__y_x_a,self).__init__()
        self.unet = Unet(Batch_Norm =Batch_Norm)
        self.log_var_out = layers.Conv2D(LATENT_DIM, (1, 1))
        self.mean_out = layers.Conv2D(LATENT_DIM, (1, 1))
        
        if sampling == "gaussian":
            self.sampling = GaussianSampling()

        if sampling == "point":
            self.sampling = PointSampling()
        
        self.gaussianLL = GaussianLL()
        self.unitGaussianLL = UnitGaussianLL()
        self.concat = layers.Concatenate()
        
    def call(self,inputs):
        supervised_mask,supervised_image,a_sample = inputs
        
        concat = self.concat([supervised_mask,supervised_image,a_sample])
        unet_output = self.unet(concat)
        log_var = self.log_var_out(unet_output)
        mean = self.mean_out(unet_output)
        
        z_sample = self.sampling((mean,log_var))
        
        #n_logp_z =  tf.reduce_mean(-self.unitGaussianLL(z_sample))
        #self.add_loss(n_logp_z)
        
        #logq_z__x_y = tf.reduce_mean(self.gaussianLL((z_sample,mean,log_var)))
        #self.add_loss(logq_z__x_y)
        
        kl_loss =  - 0.5 * tf.reduce_mean(
            log_var - tf.square(mean) - tf.exp(log_var) + 1)
     
        self.add_loss(kl_loss)
        
        return z_sample

class p_x__y_z_a(layers.Layer):
    
    def __init__(self,Batch_Norm = False):
        super(p_x__y_z_a,self).__init__()
        self.unet = Unet(Batch_Norm =Batch_Norm)
        self.out = layers.Conv2D(RGB, (1, 1), activation='sigmoid')
        self.concat = layers.Concatenate()
        self.bce = losses.BinaryCrossentropy()
        
    def call(self,inputs):
        supervised_image,supervised_mask,z_sample,a_sample = inputs

        concat = self.concat([supervised_mask,z_sample,a_sample])
        unet_output = self.unet(concat)
        bern_parameters = self.out(unet_output)
       
        n_logp_x__y_z_a = tf.reduce_mean(self.bce(supervised_image, bern_parameters))
        self.add_loss(n_logp_x__y_z_a)
        
        return bern_parameters

class q_a__x(layers.Layer):

    def __init__(self,Batch_Norm = False,sampling = "gaussian"):
        super(q_a__x,self).__init__()
        self.unet = Unet(Batch_Norm = Batch_Norm)
        self.log_var_out = layers.Conv2D(LATENT_DIM, (1, 1))
        self.mean_out = layers.Conv2D(LATENT_DIM, (1, 1))
        
        if sampling == "gaussian":
            self.sampling = GaussianSampling()

        if sampling == "point":
            self.sampling = PointSampling()
        self.gaussianLL = GaussianLL()
        
    def call(self,inputs):
        image = inputs
        unet_output = self.unet(image)
        log_var = self.log_var_out(unet_output)
        mean = self.mean_out(unet_output)
        
        a_sample = self.sampling((mean,log_var))
        
        logq_a__x = tf.reduce_mean(self.gaussianLL((a_sample,mean,log_var)))
        self.add_loss(logq_a__x)

        return a_sample

class p_a__x_y_z(layers.Layer):

    def __init__(self,Batch_Norm = False,sampling = "gaussian"):
        super(p_a__x_y_z,self).__init__()
        self.unet = Unet(Batch_Norm = Batch_Norm)
        self.log_var_out = layers.Conv2D(LATENT_DIM, (1, 1))
        self.mean_out = layers.Conv2D(LATENT_DIM, (1, 1))
        
        if sampling == "gaussian":
            self.sampling = GaussianSampling()

        if sampling == "point":
            self.sampling = PointSampling()

        self.gaussianLL = GaussianLL()
        self.concat = layers.Concatenate()

    def call(self,inputs):
        a_sample,image,mask,z_sample = inputs

        concat = self.concat([image,mask,z_sample])
        unet_output = self.unet(concat)
        log_var = self.log_var_out(unet_output)
        mean = self.mean_out(unet_output)
        
        a_dummy = self.sampling((mean,log_var))
        
        n_logp_a__x_y_z = tf.reduce_mean(-self.gaussianLL((a_sample,mean,log_var)))
        self.add_loss(n_logp_a__x_y_z)

        return a_dummy
