#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import losses

import numpy as np

import albumentations as A
from PIL import Image

import os
import random
import math


# In[2]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[3]:


SHAPE = 64
RGB = 3
CLASSES = 21
LATENT_DIM = 1
TEMPERATURE = .1
NUM_UNLABELED = 14212
NUM_LABELED = 1456
NUM_VALIDAITON = 1457


# In[4]:


#Defines the augmentitations
def get_training_augmentation():
    train_transform = [
        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),     
        A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1)
    ]
    return A.Compose(test_transform)


# In[5]:


def train_generator(batch_size = 64,shape = (SHAPE,SHAPE)):

    # Loads in unlabeled images(file paths) and repeats the labeled images until they're
    # are more labeled ones then unlabeled ones     
    image_path_list = os.listdir('./VOCdevkit/VOC2012/train_frames/')
    unsupervised_path_list = os.listdir('./VOCdevkit/VOC2012/JPEGImages/')

    random.shuffle(image_path_list)
    random.shuffle(unsupervised_path_list)

    lis = os.listdir('./VOCdevkit/VOC2012/train_frames/')
    while len(image_path_list) <= len(unsupervised_path_list):
        random.shuffle(lis)
        image_path_list.extend(lis)

    X_s = np.zeros((batch_size, shape[1], shape[0], RGB), dtype='float32')
    Y_s = np.zeros((batch_size, shape[1], shape[0],CLASSES), dtype='float32')
    X_un = np.zeros((batch_size, shape[1], shape[0], RGB), dtype='float32')
        
    def getitem(i):
        n = 0
        
        for x in image_path_list[i*batch_size:(i+1)*batch_size]:
            
            image = np.array(Image.open('./VOCdevkit/VOC2012/train_frames/' + x))
            label = np.array(Image.open('./VOCdevkit/VOC2012/train_masks/' + x.replace('.jpg','.png')))

            sample = get_training_augmentation()(image=image, mask=label)
            image, label = sample['image']/255,sample['mask']
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)
            categorical_label = tf.keras.utils.to_categorical(label)

            X_s[n] = image
            #cat_label -> image
            Y_s[n] = categorical_label[:,:,0:CLASSES]
            n = n + 1
        
        n = 0
            
        for x in unsupervised_path_list[i*batch_size:(i+1)*batch_size]:
    
            image = np.array(Image.open('./VOCdevkit/VOC2012/JPEGImages/' + x))

            sample = get_training_augmentation()(image=image)
            image= sample['image']/255
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)

            X_un[n] = image
            n = n + 1

        #return [self.X_s,self.Y_s,self.X_un] , self.Y_s
        #return [X_s,Y_s],{'p_x__y_z_s':Y_s,'z_s_sample': Y_s,'p_y__k_s': Y_s,'k_s_sample':Y_s,'q_y__x_s':Y_s}
        #return ([X_s, Y_s])
        return ((X_s,Y_s),())
        

    def on_epoch_end():
        random.shuffle(unsupervised_path_list)
        image_path_list = os.listdir('./VOCdevkit/VOC2012/train_frames/')
        lis = os.listdir('./VOCdevkit/VOC2012/train_frames/')
        while len(image_path_list) <= len(unsupervised_path_list):
            random.shuffle(lis)
            image_path_list.extend(lis) 
        
    i = -1 
    while True :
        if i < len(unsupervised_path_list) // batch_size:
            i = i + 1
        else: 
            on_epoch_end()
            i = 0
            
        yield getitem(i)
            


# In[6]:


def val_generator(batch_size = 64,shape = (SHAPE,SHAPE)):

    image_path_list = os.listdir('./VOCdevkit/VOC2012/val_frames/')
    random.shuffle(image_path_list)

    X_s = np.zeros((batch_size, shape[1], shape[0], 3), dtype='float32')
    Y_s = np.zeros((batch_size, shape[1], shape[0],21), dtype='float32')
    X_un = np.zeros((batch_size, shape[1], shape[0], 3), dtype='float32')
        
    def getitem(i):
        n = 0
        
        for x in image_path_list[i*batch_size:(i+1)*batch_size]:
            
            image = np.array(Image.open('./VOCdevkit/VOC2012/val_frames/' + x))
            label = np.array(Image.open('./VOCdevkit/VOC2012/val_masks/' + x.replace('.jpg','.png')))

            sample = get_validation_augmentation()(image=image, mask=label)
            image, label = sample['image']/255, sample['mask']
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)

            categorical_label = tf.keras.utils.to_categorical(label)

            X_s[n] = image
            #cat_label -> image
            Y_s[n] = categorical_label[:,:,0:CLASSES]
            n = n + 1

        #return [self.X_s, self.Y_s, self.X_un]
        #return [X_s,Y_s],[Y_s,Y_s,Y_s,Y_s,Y_s]
        return ((X_s,Y_s),())
        
    def on_epoch_end():
        random.shuffle(image_path_list)    
    i = -1 
    while True:
        if i  < len(image_path_list) // batch_size :
            i = i + 1
        else:
            on_epoch_end()
            i = 0
        yield getitem(i)


# In[7]:


class EncoderUnetModule(layers.Layer):
    
    def __init__(self,filterDim,dropout = 0.0):
        super(EncoderUnetModule,self).__init__()
        self.conv1 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                    padding='same')
        self.dropout = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')
        
    def call(self,inputs,training = False):
        x = self.conv1(inputs)
        if training:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


# In[8]:


class DecoderUnetModule(layers.Layer):
    
    def __init__(self,filterDim,dropout = 0.0):
        super(DecoderUnetModule,self).__init__()
        
        self.deconv = layers.Conv2DTranspose(filterDim, (2, 2), strides=(2, 2), padding='same')
        self.concat = layers.Concatenate()
        self.conv1 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')
        self.dropout = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(filterDim, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')
        
    def call(self,inputs,training = False):
        x = inputs[0]
        skip = inputs[1]
        x = self.deconv(x)
        x = self.concat([x,skip])
        x = self.conv1(x)
        if training:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


# In[9]:


class Unet(layers.Layer):
    
    def __init__(self):
        super(Unet,self).__init__()
        
        self.pool = layers.MaxPooling2D((2, 2))
        
        self.encoder16 = EncoderUnetModule(16)
        self.encoder32 = EncoderUnetModule(32)
        self.encoder64 = EncoderUnetModule(64)
        self.encoder128 = EncoderUnetModule(128)
        self.encoder256 = EncoderUnetModule(256)
        
        self.decoder128 = DecoderUnetModule(128)
        self.decoder64 = DecoderUnetModule(64)
        self.decoder32 = DecoderUnetModule(32)
        self.decoder16 = DecoderUnetModule(16)
        
        
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


# In[10]:


class GaussianSampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(z_mean))
    z_sample = z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
    return z_sample


# In[11]:


class GaussianLL(layers.Layer):
    
    def call(self,inputs):
        x , mu, log_var = inputs
        x = tf.keras.layers.Flatten()(x)
        mu = tf.keras.layers.Flatten()(mu)
        log_var = tf.keras.layers.Flatten()(log_var)

        c = -.5 * math.log(2*math.pi)
        density = c - log_var/2 - ((x - mu)/(2*tf.keras.backend.exp(log_var) + 1e-8))*(x - mu)

        return tf.keras.backend.sum(density,axis = -1)


# In[12]:


class UnitGaussianLL(layers.Layer):
    
    def call(self,inputs):
        x = inputs
        x = tf.keras.layers.Flatten()(x)

        c = -.5 * math.log(2*math.pi)
        density = c - x**2/2

        return tf.keras.backend.sum(density,axis = -1)


# In[13]:


class IouCoef(layers.Layer):
    def call(self,inputs):
        y_true,y_pred = inputs
        smooth = 1
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_pred * y_true), axis=[1,2,3])
        union = tf.keras.backend.sum(y_pred,[1,2,3])+tf.keras.backend.sum(y_true,[1,2,3])-intersection
        iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou


# In[14]:


class q_y__x_s(layers.Layer):
    
    def __init__(self):
        super(q_y__x_s,self).__init__()
        self.unet = Unet()
        self.out = layers.Conv2D(CLASSES, (1, 1), activation='softmax')
        self.cce = losses.CategoricalCrossentropy()
        self.iou = IouCoef()
        
    def call(self,inputs):
        supervised_mask,supervised_image = inputs

        
        unet_output = self.unet(supervised_image)
        cat_parameters = self.out(unet_output)
        
        logq_y__x = tf.reduce_mean(2000*self.cce(supervised_mask, cat_parameters))
        self.add_loss(logq_y__x)
        
        iou = self.iou((supervised_mask,cat_parameters))
        self.add_metric(iou,name='iou',aggregation='mean')
        
        return cat_parameters


# In[15]:


class q_k__y_s(layers.Layer):
    
    def __init__(self):
        super(q_k__y_s,self).__init__()
        self.unet = Unet()
        self.log_var_out = layers.Conv2D(LATENT_DIM, (1, 1))
        self.mean_out = layers.Conv2D(LATENT_DIM, (1, 1))
        
        self.sampling = GaussianSampling()
        self.gaussianLL = GaussianLL()
        self.unitGaussianLL = UnitGaussianLL() 
        
    def call(self,inputs):
        supervised_mask = inputs
        
        unet_output = self.unet(supervised_mask)
        log_var = self.log_var_out(unet_output)
        mean = self.mean_out(unet_output)
        
        k_sample = self.sampling((mean,log_var))
        
        n_logp_k =  tf.reduce_mean(-self.unitGaussianLL(k_sample))
        self.add_loss(n_logp_k)
        
        logq_k__y = tf.reduce_mean(self.gaussianLL((k_sample,mean,log_var)))
        self.add_loss(logq_k__y)
        
        return k_sample


# In[16]:


class p_y__k_s(layers.Layer):
    
    def __init__(self):
        super(p_y__k_s,self).__init__()
        self.unet = Unet()
        self.out = layers.Conv2D(CLASSES, (1, 1), activation='softmax')
        self.cce = losses.CategoricalCrossentropy()
        
    def call(self,inputs):
        supervised_mask,k_sample = inputs

        unet_output = self.unet(k_sample)
        cat_parameters = self.out(unet_output)
       
        n_logp_y__k = tf.reduce_mean(-self.cce(supervised_mask, cat_parameters))
        self.add_loss(n_logp_y__k)
        
        return cat_parameters


# In[17]:


class q_z__y_x_s(layers.Layer):
    
    def __init__(self):
        super(q_z__y_x_s,self).__init__()
        self.unet = Unet()
        self.log_var_out = layers.Conv2D(LATENT_DIM, (1, 1))
        self.mean_out = layers.Conv2D(LATENT_DIM, (1, 1))
        
        self.sampling = GaussianSampling()
        self.gaussianLL = GaussianLL()
        self.unitGaussianLL = UnitGaussianLL()
        self.concat = layers.Concatenate()
        
    def call(self,inputs):
        supervised_mask,supervised_image = inputs
        
        concat = self.concat([supervised_mask,supervised_image])
        unet_output = self.unet(concat)
        log_var = self.log_var_out(unet_output)
        mean = self.mean_out(unet_output)
        
        z_sample = self.sampling((mean,log_var))
        
        n_logp_z =  tf.reduce_mean(-self.unitGaussianLL(z_sample))
        self.add_loss(n_logp_z)
        
        logq_z__x_y = tf.reduce_mean(self.gaussianLL((z_sample,mean,log_var)))
        self.add_loss(logq_z__x_y)
        
        return z_sample


# In[18]:


class p_x__y_z_s(layers.Layer):
    
    def __init__(self):
        super(p_x__y_z_s,self).__init__()
        self.unet = Unet()
        self.out = layers.Conv2D(RGB, (1, 1), activation='sigmoid')
        self.concat = layers.Concatenate()
        self.bce = losses.BinaryCrossentropy()
        
    def call(self,inputs):
        supervised_image,supervised_mask,z_sample = inputs

        concat = self.concat([supervised_image,supervised_mask,z_sample])
        unet_output = self.unet(concat)
        bern_parameters = self.out(unet_output)
       
        n_logp_x__y_z = tf.reduce_mean(self.bce(supervised_image, bern_parameters))
        self.add_loss(n_logp_x__y_z)
        
        return bern_parameters


# In[19]:


class Supervised_Nirvanna(tf.keras.Model):
    
    def __init__(self):
        super(Supervised_Nirvanna,self).__init__()
        
        self.q_y__x_s = q_y__x_s()
        self.q_k__y_s = q_k__y_s()
        self.p_y__k_s = p_y__k_s()
        self.q_z__y_x_s = q_z__y_x_s()
        self.p_x__y_z_s = p_x__y_z_s()
        
    def call(self,inputs):
        supervised_image,supervised_mask = inputs
        
        q_y__x_s_out = self.q_y__x_s((supervised_mask,supervised_image))
        q_k__y_s_out = self.q_k__y_s(supervised_mask)
        p_y__k_s_out = self.p_y__k_s((supervised_mask,q_k__y_s_out))
        q_z__y_x_s_out = self.q_z__y_x_s((supervised_mask,supervised_image))
        p_x__y_z_s_out = self.p_x__y_z_s((supervised_image,supervised_mask,q_z__y_x_s_out))
        
        return  q_y__x_s_out,q_k__y_s_out,p_y__k_s_out,q_z__y_x_s_out,p_x__y_z_s_out


# In[20]:


model = Supervised_Nirvanna()


# In[21]:


model.compile('Adam')


# In[22]:


BS = 32
train_gen = train_generator(batch_size = BS)
val_gen = val_generator(batch_size = BS)


# In[23]:


model.fit(x = train_gen,
                    steps_per_epoch = NUM_LABELED//BS,
                    epochs=100,
                   validation_data = val_gen,
                   validation_steps = NUM_VALIDATION//BS,
                     validation_freq= 10)


# In[ ]:


# A function that generates samples from a set of categorical distributions 
# in a way that the gradient can propagate through.
def gumbel_softmax(args):
    ind_multinomial = args
    gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(TEMPERATURE, probs=ind_multinomial)
    return gumbel_dist.sample()

