from unet_model import Unet_Model
from param import BS,BN,NUM_LABELED,NUM_VALIDATION,LR
from data_generators import train_generator,val_generator
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


train_gen = train_generator(batch_size = BS)
val_gen = val_generator(batch_size = BS)

model = Unet_Model(Batch_Norm = BN,dropout = 0.0)
model.compile(tf.keras.optimizers.Adam(lr=LR,clipnorm = 1.,clipvalue = 0.5))


model.fit_generator(generator = train_gen,
                    steps_per_epoch = NUM_LABELED//BS,
                    epochs=100,
                   validation_data = val_gen,
                   validation_steps = NUM_VALIDATION//BS)
