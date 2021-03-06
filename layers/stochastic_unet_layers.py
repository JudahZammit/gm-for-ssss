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

class PointEncoderLayer(layers.Layer):

    def __init__(self,filterDim,dropout = 0.0,Batch_Norm=False,name = ''):
        super(PointEncoderLayer,self).__init__(name = name)
        self.conv1 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                    padding='same')
        self.dropout = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.Batch_Norm = Batch_Norm
        self.concat = layers.Concatenate()
        self.pool = layers.MaxPooling2D((2, 2))
        self.filterDim = filterDim

    def call(self,inputs,training = False):
        x = inputs 
        if self.filterDim > 16:
            x = self.pool(x)
        x = self.conv1(x)
        if self.Batch_Norm:
            x = self.bn1(x)
        if training:
            x = self.dropout(x)
        x = self.conv2(x)
        point_param = x #Identity
        return point_param


class PointDecoderLayer(layers.Layer):

    def __init__(self,filterDim,dropout = 0.0,Batch_Norm=False,Skip = True,name = ''):
        super(PointDecoderLayer,self).__init__(name = name)

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
        self.skip = Skip

    def call(self,inputs,training = False):
        x,skip = inputs
        
        if len(x) > 1:
            x = self.concat(x)
        elif len(x) == 1:
            x = x[0]

        if(self.skip):
            if len(skip) > 1:
                skip = self.concat(skip)
            elif len(skip) == 1:
                skip = skip[0]
        
        x = self.deconv(x)
        if(self.skip):
            x = self.concat([x,skip])
        x = self.conv1(x)
        if self.Batch_Norm:
            x = self.bn1(x)
        if training:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


class GaussianEncoderLayer(layers.Layer):

    def __init__(self,filterDim,dropout = 0.0,Batch_Norm=False,name = ''):
        super(GaussianEncoderLayer,self).__init__(name = name)
        self.conv1 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                    padding='same')
        self.dropout = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.Batch_Norm = Batch_Norm
        self.concat = layers.Concatenate()
        self.pool = layers.MaxPooling2D((2, 2))
        self.filterDim = filterDim

    def call(self,inputs,training = False):
        x = inputs
        if self.filterDim > 16:
            x = self.pool(x)
        x = self.conv1(x)
        if self.Batch_Norm:
            x = self.bn1(x)
        if training:
            x = self.dropout(x)
        mean = self.conv2(x)
        logvar = self.conv2(x)
        return mean,logvar


class GaussianDecoderLayer(layers.Layer):

    def __init__(self,filterDim,dropout = 0.0,Batch_Norm=False,Skip = True,name = ''):
        super(GaussianDecoderLayer,self).__init__(name = name)

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
        self.skip = Skip

    def call(self,inputs,training = False):
        x,skip = inputs

        if len(x) > 1:
            x = self.concat(x)
        elif len(x) == 1:
            x = x[0]

        if(self.skip):
            if len(skip) > 1:
                skip = self.concat(skip)
            elif len(skip) == 1:
                skip = skip[0]
        x = self.deconv(x)
        if(self.skip):
            x = self.concat([x,skip])
     
        x = self.conv1(x)
        if self.Batch_Norm:
            x = self.bn1(x)
        if training:
            x = self.dropout(x)
        mean = self.conv2(x)
        logvar = self.conv2(x)
        return mean,logvar
