from custom_models import Supervised_GM
from data_generators import train_generator,val_generator
from param import BN,BS,NUM_LABELED,NUM_VALIDATION


model = Supervised_GM(Batch_Norm = BN) 
  
model.compile('Adam')  
 
train_gen = train_generator(batch_size = BS) 
val_gen = val_generator(batch_size = BS)  
 
model.fit(x = train_gen, 
                    steps_per_epoch = NUM_LABELED//BS, 
                    epochs=100, 
                   validation_data = val_gen, 
                   validation_steps = NUM_VALIDATION//BS, 
                     validation_freq= 10) 


