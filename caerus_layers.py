from param import CLASSES,LATENT_DIM,RGB,TEMPERATURE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras import layers
from tensorflow.keras import losses
from stochastic_unet_layers import (PointEncoderLayer,PointDecoderLayer,
        GaussianEncoderLayer,GaussianDecoderLayer)
from sampling import GaussianSampling
from likelihoods import GaussianLL,UnitGaussianLL
from metrics import IouCoef

class p_x__e1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_x__e1,self).__init__()

        self.conv = layers.Conv2D(RGB,(1,1),
                activation='sigmoid')
        self.bce = BinaryCrossEntropy()
            
    def call(self,inputs):
        image,e1 = inputs
    
        bern_param = self.conv(e1)
        
        n_log_p_x__e1 = self.bce(image,bern_param)
        
        return bern_param

class f_d1__d2_z1_k1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_d1__d2_z1_k1,self).__init__()

        self.decoder = PointDecoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)

    def call(self,inputs):
        d2, z1_sample,k1_sample = inputs

        d1 = self.decoder([[e2],[z1_sample,k1_sample]])
        
        return d1

class f_d2__d3_z2_k2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_d2__d3_z2_k2,self).__init__()

        self.decoder = PointDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)

    def call(self,inputs):
        d3, z2_sample,k2_sample = inputs

        d2 = self.decoder([[d3],[z2_sample,k2_sample]])
        
        return d2

class f_d3__d4_z3_k3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_d3__d4_z3_k3,self).__init__()

        self.decoder = PointDecoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)

    def call(self,inputs):
        d4, z3_sample,k1_sample = inputs

        d3 = self.decoder([[d4],[z3_sample,k3_sample]])
        
        return d3

class f_d4__z4_k4_z5_k5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_d4__z4_k4_z5_k5,self).__init__()

        self.decoder = PointDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)

    def call(self,inputs):
        z4_sample,k4_sample,z5_sample,k5_sample = inputs

        d4 = self.decoder([[z5_sample,k5_sample],[z4_sample,k4_sample]])
        
        return d4

class p_y__k1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_y__k1,self).__init__()

        self.conv = layers.Conv2D(CLASSES,(1,1),
                activation='softmax')
    
        self.cce = CategoricalCrossentropy()
        self.iou = IouCoef()


    def call(self,inputs):
        mask,k1_sample = inputs

        cat_param = self.conv(k1_sample)

        n_log_p_y__k1 = self.cce(mask,cat_param)
        self.add_loss(n_log_p_y__k1)
        
        iou = self.iou((mask,cat_param))
        self.add_metric(iou,name= 'IOU', aggregation= 'mean')

        return cat_param

class p_k1__k2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k1__k2,self).__init__()

        self.decoder = GaussianDecoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        k1_sample, k2_sample = inputs

        mean,logvar = self.decoder([[k1_sample],[]])
        
        n_log_p_k1__k2 = -self.ll(k1_sample,mean,logvar)
        self.add_loss(n_log_p_k1__k2)

        # return won't get used
        return mean,logvar

class p_k2__k3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k2__k3,self).__init__()

        self.decoder = GaussianDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        k2_sample, k3_sample = inputs

        mean,logvar = self.decoder([[k2_sample],[]])
        
        n_log_p_k2__k3 = -self.ll(k2_sample,mean,logvar)
        self.add_loss(n_log_p_k2__k3)

        # return won't get used
        return mean,logvar

class p_k3__k4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k3__k4,self).__init__()

        self.decoder = GaussianDecoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        k3_sample, k4_sample = inputs

        mean,logvar = self.decoder([[k3_sample],[]])
        
        n_log_p_k3__k4 = -self.ll(k3_sample,mean,logvar)
        self.add_loss(n_log_p_k3__k4)

        # return won't get used
        return mean,logvar

class p_k4__k5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k4__k5,self).__init__()

        self.decoder = GaussianDecoderLayer(128,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        k4_sample, k5_sample = inputs

        mean,logvar = self.decoder([[k4_sample],[]])
        
        n_log_p_k4__k5 = -self.ll(k4_sample,mean,logvar)
        self.add_loss(n_log_p_k4__k5)

        # return won't get used
        return mean,logvar

class p_k5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k5,self).__init__()

        self.ll = UnitGaussianLL()

    def call(self,inputs):
        k5_sample = inputs
 
        n_log_k5 = -self.ll(k4_sample)
        self.add_loss(n_log_p_k5)

        # return won't get used
        return mean,logvar

class p_z1__z2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z1__z2,self).__init__()

        self.decoder = GaussianDecoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        z1_sample, z2_sample = inputs

        mean,logvar = self.decoder([[z1_sample],[]])
        
        n_log_p_z1__z2 = -self.ll(z1_sample,mean,logvar)
        self.add_loss(n_log_p_z1__z2)

        # return won't get used
        return mean,logvar

class p_z2__z3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z2__z3,self).__init__()

        self.decoder = GaussianDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        z2_sample, z3_sample = inputs

        mean,logvar = self.decoder([[z2_sample],[]])
        
        n_log_p_z2__z3 = -self.ll(z2_sample,mean,logvar)
        self.add_loss(n_log_p_z2__z3)

        # return won't get used
        return mean,logvar

class p_z3__z4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z3__z4,self).__init__()

        self.decoder = GaussianDecoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        z3_sample, z4_sample = inputs

        mean,logvar = self.decoder([[z3_sample],[]])
        
        n_log_p_z3__z4 = -self.ll(z3_sample,mean,logvar)
        self.add_loss(n_log_p_z3__z4)

        # return won't get used
        return mean,logvar

class p_z4__z5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z4__z5,self).__init__()

        self.decoder = GaussianDecoderLayer(128,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        z4_sample, z5_sample = inputs

        mean,logvar = self.decoder([[z4_sample],[]])
        
        n_log_p_z4__z5 = -self.ll(z4_sample,mean,logvar)
        self.add_loss(n_log_p_z4__z5)

        # return won't get used
        return mean,logvar

class p_z5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z5,self).__init__()

        self.ll = UnitGaussianLL()

    def call(self,inputs):
        z5_sample = inputs
 
        n_log_z5 = -self.ll(z5_sample)
        self.add_loss(n_log_p_z5)

        # return won't get used
        return mean,logvar

class q(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q,self).__init__()


    def call(self,inputs):
         = inputs
 
        return
