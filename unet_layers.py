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

class EncoderUnetModule(layers.Layer):
    
    def __init__(self,filterDim,dropout = 0.0,Batch_Norm=False):
        super(EncoderUnetModule,self).__init__()
        self.conv1 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                    padding='same')
        self.dropout = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.Batch_Norm = Batch_Norm     
    def call(self,inputs,training = False):
        x = self.conv1(inputs)
        if self.Batch_Norm:
            x = self.bn1(x)
        if training:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.Batch_Norm:
            x = self.bn2(x)
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
        
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.Batch_Norm = Batch_Norm     
    
    def call(self,inputs,training = False):
        x = inputs[0]
        skip = inputs[1]
        x = self.deconv(x)
        x = self.concat([x,skip])
        x = self.conv1(x)
        if self.Batch_Norm:
            x = self.bn1(x)
        if training:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.Batch_Norm:
            x = self.bn2(x)
        return x

class Unet(layers.Layer):
    
    def __init__(self,dropout = 0.0,Batch_Norm = False):
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

