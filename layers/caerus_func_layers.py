from param import CLASSES,LATENT_DIM,RGB,TEMPERATURE
from layers.stochastic_unet_layers import (PointEncoderLayer,PointDecoderLayer,
        GaussianEncoderLayer,GaussianDecoderLayer)
from helpers.sampling import GaussianSampling
from helpers.likelihoods import GaussianLL,UnitGaussianLL
from helpers.metrics import IouCoef

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import backend as K

class p_x__d1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_x__d1,self).__init__(name = 'd1--x')

        self.conv = layers.Conv2D(RGB,(1,1),
                activation='sigmoid')
            
    def call(self,inputs):
        d1 = inputs
    
        bern_param = self.conv(d1)
         
        return bern_param

class f_d1__d2_z1_k1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_d1__d2_z1_k1,self).__init__(name = 'z1-k1-d2--d1')

        self.decoder = PointDecoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)

    def call(self,inputs):
        d2, z1_sample,k1_sample = inputs

        d1 = self.decoder([[d2],[z1_sample,k1_sample]])
        
        return d1

class f_d2__d3_z2_k2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_d2__d3_z2_k2,self).__init__(name = 'z2-k2-d3--d2')

        self.decoder = PointDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)

    def call(self,inputs):
        d3, z2_sample,k2_sample = inputs

        d2 = self.decoder([[d3],[z2_sample,k2_sample]])
        
        return d2

class f_d3__d4_z3_k3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_d3__d4_z3_k3,self).__init__(name = 'z3-k3-d4--d3')

        self.decoder = PointDecoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)

    def call(self,inputs):
        d4, z3_sample,k3_sample = inputs

        d3 = self.decoder([[d4],[z3_sample,k3_sample]])
        
        return d3

class f_d4__z4_k4_z5_k5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_d4__z4_k4_z5_k5,self).__init__(name = 'z4-k4-z5-k5--d4')

        self.decoder = PointDecoderLayer(128,dropout = dropout,
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

    def call(self,inputs):
        k1_sample = inputs

        cat_param = self.conv(k1_sample)

        return cat_param

class p_k1__k2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k1__k2,self).__init__()

        self.decoder = GaussianDecoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        k1_sample, k2_sample = inputs

        mean,logvar = self.decoder([[k2_sample],[]])
        
        n_log_p_k1__k2 = -self.ll((k1_sample,mean,logvar))
        self.add_loss(n_log_p_k1__k2)

        # return won't get used
        return n_log_p_k1__k2 

class p_k2__k3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k2__k3,self).__init__()

        self.decoder = GaussianDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        k2_sample, k3_sample = inputs

        mean,logvar = self.decoder([[k3_sample],[]])
        
        n_log_p_k2__k3 = -self.ll((k2_sample,mean,logvar))
        self.add_loss(n_log_p_k2__k3)

        # return won't get used
        return  n_log_p_k2__k3

class p_k3__k4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k3__k4,self).__init__()

        self.decoder = GaussianDecoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        k3_sample, k4_sample = inputs

        mean,logvar = self.decoder([[k4_sample],[]])
        
        n_log_p_k3__k4 = -self.ll((k3_sample,mean,logvar))
        self.add_loss(n_log_p_k3__k4)

        # return won't get used
        return   n_log_p_k3__k4

class p_k4__k5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_k4__k5,self).__init__()

        self.decoder = GaussianDecoderLayer(128,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        k4_sample, k5_sample = inputs

        mean,logvar = self.decoder([[k5_sample],[]])
        
        n_log_p_k4__k5 = -self.ll((k4_sample,mean,logvar))
        self.add_loss(n_log_p_k4__k5)

        # return won't get used
        return n_log_p_k4__k5

class p_k5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False,Kull = 0):
        super(p_k5,self).__init__(name = 'k5')

        self.ll = UnitGaussianLL(Kull)

    def call(self,inputs):
        k5_sample = inputs
 
        n_log_p_k5 = -self.ll(k5_sample)
        self.add_loss(n_log_p_k5)

        # return won't get used
        return n_log_p_k5 

class p_z1__z2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z1__z2,self).__init__()

        self.decoder = GaussianDecoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        z1_sample, z2_sample = inputs

        mean,logvar = self.decoder([[z2_sample],[]])
        
        n_log_p_z1__z2 = -self.ll((z1_sample,mean,logvar))
        self.add_loss(n_log_p_z1__z2)

        # return won't get used
        return n_log_p_z1__z2

class p_z2__z3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z2__z3,self).__init__()

        self.decoder = GaussianDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        z2_sample, z3_sample = inputs

        mean,logvar = self.decoder([[z3_sample],[]])
        
        n_log_p_z2__z3 = -self.ll((z2_sample,mean,logvar))
        self.add_loss(n_log_p_z2__z3)

        # return won't get used
        return n_log_p_z2__z3

class p_z3__z4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z3__z4,self).__init__()

        self.decoder = GaussianDecoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        z3_sample, z4_sample = inputs

        mean,logvar = self.decoder([[z4_sample],[]])
        
        n_log_p_z3__z4 = -self.ll((z3_sample,mean,logvar))
        self.add_loss(n_log_p_z3__z4)

        # return won't get used
        return n_log_p_z3__z4

class p_z4__z5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(p_z4__z5,self).__init__()

        self.decoder = GaussianDecoderLayer(128,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = False)
        self.ll = GaussianLL()

    def call(self,inputs):
        z4_sample, z5_sample = inputs

        mean,logvar = self.decoder([[z5_sample],[]])
        
        n_log_p_z4__z5 = -self.ll((z4_sample,mean,logvar))
        self.add_loss(n_log_p_z4__z5)

        # return won't get used
        return n_log_p_z4__z5

class p_z5(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False,Kull=0):
        super(p_z5,self).__init__(name = 'z5')

        self.ll = UnitGaussianLL(Kull)

    def call(self,inputs):
        z5_sample = inputs
 
        n_log_p_z5 = -self.ll(z5_sample)
        self.add_loss(n_log_p_z5)

        # return won't get used
        return n_log_p_z5

class f_e1__x(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_e1__x,self).__init__()
        
        self.encoder = PointEncoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm)
        
    def call(self,inputs):
        image = inputs
    
        e1 = self.encoder(image)
        return e1

class f_e2__e1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_e2__e1,self).__init__()
        
        self.encoder = PointEncoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm)
        
    def call(self,inputs):
        e1 = inputs
    
        e2 = self.encoder(e1)
        return e2

class f_e3__e2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_e3__e2,self).__init__()
        
        self.encoder = PointEncoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm)
        
    def call(self,inputs):
        e2 = inputs
    
        e3 = self.encoder(e2)
        return e3

class f_e4__e3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(f_e4__e3,self).__init__()
        
        self.encoder = PointEncoderLayer(128,dropout = dropout,
                Batch_Norm = Batch_Norm)
        
    def call(self,inputs):
        e3 = inputs
    
        e4 = self.encoder(e3)
        return e4


class q_z1__z2_e1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_z1__z2_e1,self).__init__()

        self.decoder = GaussianDecoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()

    def call(self,inputs):
        z2_sample,e1 = inputs

        mean,logvar = self.decoder([[z2_sample],[e1]])
        
        z1_sample = self.sample((mean,logvar))

        log_q_z1__z2_e1 = self.ll((z1_sample,mean,logvar))
        self.add_loss(log_q_z1__z2_e1)

        return z1_sample

class q_z2__z3_e2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_z2__z3_e2,self).__init__()

        self.decoder = GaussianDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()

    def call(self,inputs):
        z3_sample,e2 = inputs

        mean,logvar = self.decoder([[z3_sample],[e2]])
        
        z2_sample = self.sample((mean,logvar))

        log_q_z2__z3_e2 = self.ll((z2_sample,mean,logvar))
        self.add_loss(log_q_z2__z3_e2)

        return z2_sample

class q_z3__z4_e3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_z3__z4_e3,self).__init__()

        self.decoder = GaussianDecoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()

    def call(self,inputs):
        z4_sample,e3 = inputs

        mean,logvar = self.decoder([[z4_sample],[e3]])
        
        z3_sample = self.sample((mean,logvar))

        log_q_z3__z4_e3 = self.ll((z3_sample,mean,logvar))
        self.add_loss(log_q_z3__z4_e3)

        return z3_sample

class q_z4__z5_e4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_z4__z5_e4,self).__init__()

        self.decoder = GaussianDecoderLayer(128,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()

    def call(self,inputs):
        z5_sample,e4 = inputs

        mean,logvar = self.decoder([[z5_sample],[e4]])
        
        z4_sample = self.sample((mean,logvar))

        log_q_z4__z5_e4 = self.ll((z4_sample,mean,logvar))
        self.add_loss(log_q_z4__z5_e4)

        return z4_sample

class q_z5__e4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False,Kull = 0,name = 'z5--e4'):
        super(q_z5__e4,self).__init__(name = name)

        self.decoder = GaussianEncoderLayer(256,dropout = dropout,
                Batch_Norm = Batch_Norm)
        self.ll = GaussianLL(Kull)
        self.sample = GaussianSampling()
        self.KL = Kull

    def call(self,inputs):
        e4 = inputs

        mean,logvar = self.decoder(e4)
        z5_sample = self.sample((mean,logvar))

        log_q_z5__e4 = self.ll((z5_sample,mean,logvar))
        #self.add_loss(log_q_z5__e4)


        return mean

class q_k1__k2_e1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_k1__k2_e1,self).__init__()

        self.decoder = GaussianDecoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()

    def call(self,inputs):
        k2_sample,e1 = inputs

        mean,logvar = self.decoder([[k2_sample],[e1]])
        
        k1_sample = self.sample((mean,logvar))

        log_q_k1__k2_e1 = self.ll((k1_sample,mean,logvar))
        self.add_loss(log_q_k1__k2_e1)

        return k1_sample

class q_k2__k3_e2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_k2__k3_e2,self).__init__()

        self.decoder = GaussianDecoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()

    def call(self,inputs):
        k3_sample,e2 = inputs

        mean,logvar = self.decoder([[k3_sample],[e2]])
        
        k2_sample = self.sample((mean,logvar))

        log_q_k2__k3_e2 = self.ll((k2_sample,mean,logvar))
        self.add_loss(log_q_k2__k3_e2)

        return k2_sample

class q_k3__k4_e3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_k3__k4_e3,self).__init__()

        self.decoder = GaussianDecoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()

    def call(self,inputs):
        k4_sample,e3 = inputs

        mean,logvar = self.decoder([[k4_sample],[e3]])
        
        k3_sample = self.sample((mean,logvar))

        log_q_k3__k4_e3 = self.ll((k3_sample,mean,logvar))
        self.add_loss(log_q_k3__k4_e3)

        return k3_sample

class q_k4__k5_e4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_k4__k5_e4,self).__init__()

        self.decoder = GaussianDecoderLayer(128,dropout = dropout,
                Batch_Norm = Batch_Norm,Skip = True)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()

    def call(self,inputs):
        k5_sample,e4 = inputs

        mean,logvar = self.decoder([[k5_sample],[e4]])
        
        k4_sample = self.sample((mean,logvar))

        log_q_k4__k5_e4 = self.ll((k4_sample,mean,logvar))
        self.add_loss(log_q_k4__k5_e4)

        return k4_sample

class q_k5__e4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False,Kull = 0,name = 'k5--e4'):
        super(q_k5__e4,self).__init__(name = name)

        self.decoder = GaussianEncoderLayer(256,dropout = dropout,
                Batch_Norm = Batch_Norm)
        self.ll = GaussianLL(Kull)
        self.sample = GaussianSampling()
        self.KL = Kull
        
    def call(self,inputs):
        e4 = inputs

        mean,logvar = self.decoder(e4)
        k5_sample = self.sample((mean,logvar))

        log_q_k5__k4 = self.ll((k5_sample,mean,logvar))
        #self.add_loss(log_q_k5__k4)
        
        return mean


class q_k1__y(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_k1__y,self).__init__()
        
        self.encoder = GaussianEncoderLayer(16,dropout = dropout,
                Batch_Norm = Batch_Norm)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()
    
    def call(self,inputs):
        mask = inputs
    
        mean,logvar = self.encoder(mask)
        
        k1_sample = self.sample((mean,logvar))

        log_q_k1__y = self.ll((k1_sample,mean,logvar))
        self.add_loss(log_q_k1__y)

        return k1_sample

class q_k2__k1(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_k2__k1,self).__init__()
        
        self.encoder = GaussianEncoderLayer(32,dropout = dropout,
                Batch_Norm = Batch_Norm)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()
    
    def call(self,inputs):
        k1_sample = inputs
    
        mean,logvar = self.encoder(k1_sample)
        
        k2_sample = self.sample((mean,logvar))

        log_q_k2__k1 = self.ll((k2_sample,mean,logvar))
        self.add_loss(log_q_k2__k1)

        return k2_sample

class q_k3__k2(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_k3__k2,self).__init__()
        
        self.encoder = GaussianEncoderLayer(64,dropout = dropout,
                Batch_Norm = Batch_Norm)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()
    
    def call(self,inputs):
        k2_sample = inputs
    
        mean,logvar = self.encoder(k2_sample)
        
        k3_sample = self.sample((mean,logvar))

        log_q_k3__k2 = self.ll((k3_sample,mean,logvar))
        self.add_loss(log_q_k3__k2)

        return k3_sample

class q_k4__k3(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(q_k4__k3,self).__init__()
        
        self.encoder = GaussianEncoderLayer(128,dropout = dropout,
                Batch_Norm = Batch_Norm)
        self.ll = GaussianLL()
        self.sample = GaussianSampling()
    
    def call(self,inputs):
        k3_sample = inputs
    
        mean,logvar = self.encoder(k3_sample)
        
        k4_sample = self.sample((mean,logvar))

        log_q_k4__k3 = self.ll((k4_sample,mean,logvar))
        self.add_loss(log_q_k4__k3)

        return k4_sample

class q_k5__k4(layers.Layer):

    def __init__(self,dropout = 0.0,Batch_Norm = False,Kull = 0,name = 'k5--k4'):
        super(q_k5__k4,self).__init__(name = name)
        
        self.encoder = GaussianEncoderLayer(256,dropout = dropout,
                Batch_Norm = Batch_Norm)
        self.ll = GaussianLL(Kull)
        self.sample = GaussianSampling()
        self.KL = Kull

    def call(self,inputs):
        k4_sample = inputs
    
        mean,logvar = self.encoder(k4_sample)
        k5_sample = self.sample((mean,logvar))

        log_q_k5__k4 = self.ll((k5_sample,mean,logvar))
        #self.add_loss(log_q_k5__k4)

        return mean

