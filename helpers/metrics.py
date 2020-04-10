import tensorflow as tf
from tensorflow.keras import layers

class IouCoef(layers.Layer):
    def call(self,inputs):
        y_true,y_pred = inputs
        y_true = tf.slice(y_true,[0,0,0,1],[-1,-1,-1,-1])
        y_pred = tf.slice(y_pred,[0,0,0,1],[-1,-1,-1,-1])
        smooth = 1
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_pred * y_true), axis=[1,2,3])
        union = tf.keras.backend.sum(y_pred,[1,2,3])+tf.keras.backend.sum(y_true,[1,2,3])-intersection
        
        
        iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

