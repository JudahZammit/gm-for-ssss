import tensorflow as tf
import math
from tensorflow.keras import layers

class GaussianLL(layers.Layer):
    def __init__(self,Kull):
        super(GaussianLL,self).__init__()

        self.KL = Kull

    def call(self,inputs):
        x , mu, log_var = inputs
        x = tf.keras.layers.Flatten()(x)
        mu = tf.keras.layers.Flatten()(mu)
        log_var = tf.keras.layers.Flatten()(log_var)

        c = -.5 * math.log(2*math.pi)
        density = c - log_var/2 - ((x - mu)/(2*tf.keras.backend.exp(log_var) + 1e-8))*(x - mu)

        density = self.KL * density
        return tf.keras.backend.mean(tf.keras.backend.sum(density,axis = -1))

class UnitGaussianLL(layers.Layer):
    def __init__(self,Kull):
        super(UnitGaussianLL,self).__init__()

        self.KL = Kull

    def call(self,inputs):
        x = inputs
        x = tf.keras.layers.Flatten()(x)

        c = -.5 * math.log(2*math.pi)
        density = c - x**2/2

        density = self.KL * density
        return tf.keras.backend.mean(tf.keras.backend.sum(density,axis = -1))

