from caerus_model import Caerus_2
from data_generators import train_generator,val_generator
import tensorflow as tf
from param import LR,BS,BN,NUM_LABELED,NUM_UNLABELED,NUM_VALIDATION
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


model = Caerus_2(Batch_Norm = BN) 
model.compile(tf.keras.optimizers.Adam(lr=LR,clipnorm = 1.,clipvalue = 0.5))  

#model = tf.keras.models.load_model('Caerus.h5')


train_gen = train_generator(batch_size = BS) 
val_gen = val_generator(batch_size = BS)  

#nex = next(train_gen)
#model.train_on_batch(nex[0],nex[1])
#model.load_weights('Caerus_2.h5')

# Saves the model best weights to a file 
checkpoint = ModelCheckpoint(
    'Caerus_2.h5', 
    monitor='val_IOU', 
    verbose=0, 
    save_best_only=False, 
    save_weights_only=False,
    mode='max',
    period = 1
)

model.fit(x = train_gen, 
                    steps_per_epoch = NUM_LABELED//BS, 
                    epochs=100, 
                   validation_data = val_gen, 
                   validation_steps = NUM_VALIDATION//BS, 
                     validation_freq= 1,
                     callbacks=[checkpoint]) 


