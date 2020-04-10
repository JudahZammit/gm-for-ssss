from caerus_model import FuncCaerus
from data_generators import train_generator,val_generator
import tensorflow as tf
from param import LR,BS,BN,NUM_LABELED,NUM_UNLABELED,NUM_VALIDATION
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import math
import pandas as pd

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def display_val(display_list,epoch):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Reconstructed Mask','Reconstructed Image','Predicted Mask']

    

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    cmap =  plt.get_cmap('magma', 21)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]),cmap = cmap)
    plt.axis('off')
  plt.savefig("./epoch_"+str(epoch)+".png")

def display_train(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Supervised Image', 'True Mask', 'Reconstructed Mask','Reconstructed Image'
          ,'Unsupervised Image','Reconsturced Image','Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    cmap =  plt.get_cmap('magma', 21)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]),cmap = cmap   )
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask

def show_val_predictions(epoch):
    point = next(val_gen)[0]
    out = model.predict_on_batch(point)
    display_val([point[0][0],create_mask(point[1][0]) ,create_mask(out['y_s'][0]), out['x_s'][0],
        create_mask(out['y_u'][0])],epoch)

class DisplayValCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_val_predictions(epoch)
    print ('\nSample Validation Prediction after epoch {}\n'.format(epoch+1))

def show_train_predictions():
    point = next(train_gen)[0]
    out = model.predict_on_batch(point)
    display_train([point[0][0],create_mask(point[1][0]) ,create_mask(out['y_s'][0]), out['x_s'][0],
        point[2][0],out['x_u'][0],create_mask(out['y_u'][0])])

class DisplayTrainCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_train_predictions()
    print ('\nSample Validation Prediction after epoch {}\n'.format(epoch+1))

# Saves the model best weights to a file 
checkpoint = ModelCheckpoint(
    'FuncCaerus.h5', 
    monitor='val_IOU', 
    verbose=0, 
    save_best_only=False, 
    save_weights_only=True,
    mode='max',
    period = 1
)

train_gen = train_generator(batch_size = BS) 
val_gen = val_generator(batch_size = BS)  
df = pd.DataFrame(columns = ['Total Loss', 'Reconstruction Loss','Val IOU'])
for i in range(200,201):
    KL = min(1,i/200)

    print('\n\n\n')
    print('Starting epoch'+str(i))
    print('KL: ' + str(KL))
    print('\n\n\n')
    opt = tf.keras.optimizers.Adam(lr=LR,clipnorm = 1.,clipvalue = 0.5)
    model = FuncCaerus(Batch_Norm = BN,Kull = KL) 
    model.compile(opt)  
     
    if False:
        nex = next(train_gen)
        model.train_on_batch(nex[0],nex[1])
        model.load_weights('FuncCaerus.h5')
    if(KL == 1):
        epochs = 100
    elif(KL == 0):
        epochs = 10
    else:
        epochs = 1
    
    if (i % 5) == 0:
        model.fit(x = train_gen, 
                        steps_per_epoch = NUM_LABELED//BS, 
                        epochs=epochs,
                        callbacks=[checkpoint,DisplayValCallback()])
    else:
        model.fit(x = train_gen, 
                        steps_per_epoch = NUM_LABELED//BS, 
                        epochs=epochs,
                        callbacks=[checkpoint])


    metrics = model.evaluate(val_gen,steps = NUM_VALIDATION//BS)
    df.loc[KL] = [metrics[0],metrics[4],metrics[5]]

df.plot()
plt.show()
