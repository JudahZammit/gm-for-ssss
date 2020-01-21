from custom_models import GM
from data_generators import train_generator,val_generator
import tensorflow as tf
from param import LR,BS,BN,NUM_LABELED,NUM_UNLABELED,NUM_VALIDATION

model = GM(Batch_Norm = BN) 
  
model.compile(tf.keras.optimizers.Adam(lr=LR,clipnorm = 1.,clipvalue = 0.5))  
 
train_gen = train_generator(batch_size = BS) 
val_gen = val_generator(batch_size = BS)  
 
model.fit(x = train_gen, 
                    steps_per_epoch = NUM_UNLABELED//BS, 
                    epochs=100, 
                   validation_data = val_gen, 
                   validation_steps = NUM_VALIDATION//BS, 
                     validation_freq= 1) 


