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


class GaussianLL(layers.Layer):

    def call(self,inputs):
        x , mu, log_var = inputs
        x = tf.keras.layers.Flatten()(x)
        mu = tf.keras.layers.Flatten()(mu)
        log_var = tf.keras.layers.Flatten()(log_var)

        c = -.5 * math.log(2*math.pi)
        density = c - log_var/2 - ((x - mu)/(2*tf.keras.backend.exp(log_var) + 1e-8))*(x - mu)

        return tf.keras.backend.sum(density,axis = -1)

class UnitGaussianLL(layers.Layer):

    def call(self,inputs):
        x = inputs
        x = tf.keras.layers.Flatten()(x)

        c = -.5 * math.log(2*math.pi)
        density = c - x**2/2

        return tf.keras.backend.sum(density,axis = -1)

