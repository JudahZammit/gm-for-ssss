from custom_layers import (q_y__x,q_k__y,p_y__k,q_z__y_x,p_x__y_z,IouCoef,Gumbel)
from tensorflow.keras import Model,losses
import tensorflow as tf
from param import TEMPERATURE,CLASS_WEIGHT

class Supervised_GM(Model):

    def __init__(self,Batch_Norm = False):
        super(Supervised_GM,self).__init__()

        self.q_y__x_s = q_y__x(False)
        self.q_k__y_s = q_k__y(Batch_Norm)
        self.p_y__k_s = p_y__k(Batch_Norm)
        self.q_z__y_x_s = q_z__y_x(Batch_Norm)
        self.p_x__y_z_s = p_x__y_z(Batch_Norm)
        self.cce = losses.CategoricalCrossentropy()
        self.iou = IouCoef()

    def call(self,inputs):
        supervised_image,supervised_mask,unsupervised_image = inputs
        
        q_y__x_s_out = self.q_y__x_s(supervised_image)
        
        n_logq_y__x_s = tf.reduce_mean(CLASS_WEIGHT*self.cce(supervised_mask,q_y__x_s_out))
        self.add_loss(n_logq_y__x_s)

        iou = self.iou((supervised_mask,q_y__x_s_out))
        self.add_metric(iou,name= 'IOU', aggregation= 'mean')
        
        q_k__y_s_out = self.q_k__y_s(supervised_mask)
        p_y__k_s_out = self.p_y__k_s((supervised_mask,q_k__y_s_out))
        q_z__y_x_s_out = self.q_z__y_x_s((supervised_mask,supervised_image))
        p_x__y_z_s_out = self.p_x__y_z_s((supervised_image,supervised_mask,q_z__y_x_s_out))

        return  q_y__x_s_out,q_k__y_s_out,p_y__k_s_out,q_z__y_x_s_out,p_x__y_z_s_out


class GM(Model):

    def __init__(self,Batch_Norm = False):
        super(GM,self).__init__()
        
        self.q_y__x = q_y__x(Batch_Norm)
        self.q_k__y = q_k__y(Batch_Norm)
        self.p_y__k = p_y__k(Batch_Norm)
        self.q_z__y_x = q_z__y_x(Batch_Norm)
        self.p_x__y_z = p_x__y_z(Batch_Norm)
        self.cce = losses.CategoricalCrossentropy()
        self.iou = IouCoef()
        self.gumbel = Gumbel()

    def call(self,inputs):
        supervised_image,supervised_mask,unsupervised_image = inputs

        #Define  q(y|x) supervised model
        q_y__x_s_out = self.q_y__x(supervised_image)

        n_logq_y__x_s = tf.reduce_mean(CLASS_WEIGHT*self.cce(supervised_mask,q_y__x_s_out))
        self.add_loss(n_logq_y__x_s)

        iou = self.iou((supervised_mask,q_y__x_s_out))
        self.add_metric(iou,name= 'IOU', aggregation= 'mean')

        q_k__y_s_out = self.q_k__y(supervised_mask)
        p_y__k_s_out = self.p_y__k((supervised_mask,q_k__y_s_out))
        q_z__y_x_s_out = self.q_z__y_x((supervised_mask,supervised_image))
        p_x__y_z_s_out = self.p_x__y_z((supervised_image,supervised_mask,q_z__y_x_s_out))
        

        # Define q(y|x) unsupervised model)
        q_y__x_u_param = self.q_y__x(unsupervised_image)
        q_y__x_u_out = self.gumbel(q_y__x_u_param)

        logq_y__x_u = tf.reduce_mean(-self.cce(q_y__x_u_out,q_y__x_u_param))
        self.add_loss(logq_y__x_u)

        q_k__y_u_out = self.q_k__y(q_y__x_u_out)
        p_y__k_u_out = self.p_y__k((q_y__x_u_out,q_k__y_u_out))
        q_z__y_x_u_out = self.q_z__y_x((q_y__x_u_out,unsupervised_image))
        p_x__y_z_u_out = self.p_x__y_z((unsupervised_image,q_y__x_u_out,q_z__y_x_u_out))


        return  q_y__x_s_out,q_k__y_s_out,p_y__k_s_out,q_z__y_x_s_out,p_x__y_z_s_out,q_y__x_u_out,q_k__y_u_out,p_y__k_u_out,q_z__y_x_u_out,p_x__y_z_u_out
