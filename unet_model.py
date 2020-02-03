from tensorflow.keras import Model,layers,losses
from unet_layers import Unet
from metrics import IouCoef
from param import CLASSES


class Unet_Model(Model):

    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(Unet_Model,self).__init__()

        self.unet = Unet(Batch_Norm=Batch_Norm,dropout = dropout)
        self.cce = losses.CategoricalCrossentropy()
        self.iou = IouCoef()
        self.out = layers.Conv2D(CLASSES, (1, 1), activation='softmax')
    

    def call(self,inputs):
        image ,mask , unsupervised_image = inputs
        out = self.unet(image)
        out = self.out(out)
        loss = self.cce(mask,out)
        iou = self.iou((mask,out))
        self.add_loss(loss)
        self.add_metric(iou,name = 'iou',aggregation = 'mean')

        return mask
