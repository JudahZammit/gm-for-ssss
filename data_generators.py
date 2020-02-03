import albumentations as A
from PIL import Image
import random 
import os
import math
import numpy as np
import gc

import tensorflow as tf
from param import SHAPE,RGB,CLASSES

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

        return ((X_s,Y_s,X_un),())


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
        gc.collect()
        yield getitem(i)



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
        return ((X_s,Y_s,X_un),())

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

