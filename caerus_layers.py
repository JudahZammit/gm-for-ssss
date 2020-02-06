from param import CLASSES,LATENT_DIM,RGB,TEMPERATURE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import losses
from unet_layers import EncoderUnetModule,DecoderUnetModule
from sampling import GaussianSampling,Gumbel
from likelihoods import GaussianLL,UnitGaussianLL


# self,filterDim
#16->32->64->128->256
#128->64->32->16

class p_y__k1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_y__k1,self).__init__()

        self.out = layers.Conv2D(CLASSES,(1,1),
                activation='softmax')
    
    def call(self,inputs):
        mask,k1 = inputs

        cat_parameters = self.out(unet_output)

        # add CCE loss with mask

        return cat_parameters

class p_k1__k2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_y__k1,self).__init__()

        self.decoder = DecoderUnetModule(
        self.out = layers.Conv2D(CLASSES,(1,1),
                activation='softmax')
    
    def call(self,inputs):
        mask,k1 = inputs

        cat_parameters = self.out(unet_output)

        # add CCE loss with mask

        return cat_parameters

class p_x__z1_k1_t1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_x__z1_k1_t1,self).__init__()

        self.out = layers.Conv2D(RGB,(1,1),
                activation='sigmoid')

        self.concat = layers.Concatenate()
    
    def call(self,inputs):
        image,z1_sample,k1_sample,t1 = inputs
    
        concat = self.concat([z1_sample,k1_sample,t1])
        bern_parameters = self.out(concat)

        # add CE loss with image

        return bern_parameters
