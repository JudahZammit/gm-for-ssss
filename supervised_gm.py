from custom_models import Supervised_GM
from data_generators import train_generator,val_generator

# Parameters
SHAPE = 64
RGB = 3
CLASSES = 21
LATENT_DIM = 1
TEMPERATURE = .1
NUM_UNLABELED = 14212
NUM_LABELED = 1456
NUM_VALIDATION = 1457
BS = 32 

model = Supervised_GM() 
  
model.compile('Adam')  
 
train_gen = train_generator(batch_size = BS) 
val_gen = val_generator(batch_size = BS)  
 
model.fit(x = train_gen, 
                    steps_per_epoch = NUM_LABELED//BS, 
                    epochs=100, 
                   validation_data = val_gen, 
                   validation_steps = NUM_VALIDATION//BS, 
                     validation_freq= 10) 


