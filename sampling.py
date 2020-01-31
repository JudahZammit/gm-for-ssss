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

class GaussianSampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(z_mean))
    z_sample = z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
    return z_sample
