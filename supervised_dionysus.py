from dionysus_model import Supervised_Dionysus
from data_generators import train_generator,val_generator
import tensorflow as tf
from param import LR,BS,BN,NUM_LABELED,NUM_UNLABELED,NUM_VALIDATION


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


model = Supervised_Dionysus(Batch_Norm = BN) 
  
model.compile(tf.keras.optimizers.Adam(lr=LR,clipnorm = 1.,clipvalue = 0.5))

 
train_gen = train_generator(batch_size = BS) 
val_gen = val_generator(batch_size = BS)  
 
model.fit(x = train_gen, 
                    steps_per_epoch = NUM_LABELED//BS, 
                    epochs=100, 
                   validation_data = val_gen, 
                   validation_steps = NUM_VALIDATION//BS, 
                     validation_freq= 10) 


