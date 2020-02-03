from custom_layers import (q_y__x,q_k__y,p_y__k,q_z__y_x,p_x__y_z,IouCoef,Gumbel)
from tensorflow.keras import Model,losses
import tensorflow as tf
from param import TEMPERATURE,CLASS_WEIGHT
from custom_layers import (q_a__x,q_y__x_a,q_z__y_x_a,p_a__x_y_z,)

class Hephaestus(Model):

    def __init__(self,Batch_Norm = False):
        super(ADGM,self).__init__()
        
        self.q_y__x_a = q_y__x_a(Batch_Norm)
        self.q_k__y = q_k__y(Batch_Norm)
        self.p_y__k = p_y__k(Batch_Norm)
        self.q_z__y_x_a = q_z__y_x_a(Batch_Norm)
        self.p_x__y_z = p_x__y_z(Batch_Norm)
        self.q_a__x = q_a__x(Batch_Norm)
        self.p_a__x_y_z = p_a__x_y_z(Batch_Norm)

        self.cce = losses.CategoricalCrossentropy()
        self.iou = IouCoef()
        self.gumbel = Gumbel()

    def call(self,inputs):
        supervised_image,supervised_mask,unsupervised_image = inputs

        #Define  q(y|x,a) supervised model
        q_a__x_s_out = self.q_a__x(supervised_image)

        q_y__x_a_s_out = self.q_y__x_a((supervised_image,q_a__x_s_out))

        n_logq_y__x_a_s = tf.reduce_mean(CLASS_WEIGHT*self.cce(supervised_mask,q_y__x_a_s_out))
        self.add_loss(n_logq_y__x_a_s)

        iou = self.iou((supervised_mask,q_y__x_a_s_out))
        self.add_metric(iou,name= 'IOU', aggregation= 'mean')

        q_k__y_s_out = self.q_k__y(supervised_mask)
        p_y__k_s_out = self.p_y__k((supervised_mask,q_k__y_s_out))
        q_z__y_x_a_s_out = self.q_z__y_x_a((supervised_mask,supervised_image,q_a__x_s_out))
        p_x__y_z_s_out = self.p_x__y_z((supervised_image,supervised_mask,q_z__y_x_a_s_out))       
        p_a__x_y_z_s_out = self.p_a__x_y_z((q_a__x_s_out,supervised_image,
            supervised_mask,q_z__y_x_a_s_out))

        # Define q(y|x,a) unsupervised model)
        q_a__x_u_out = self.q_a__x(unsupervised_image)

        q_y__x_a_u_param = self.q_y__x_a((unsupervised_image,q_a__x_u_out))
        q_y__x_a_u_out = self.gumbel(q_y__x_a_u_param)

        logq_y__x_a_u = tf.reduce_mean(-self.cce(q_y__x_a_u_out,q_y__x_a_u_param))
        self.add_loss(logq_y__x_a_u)

        q_k__y_u_out = self.q_k__y(q_y__x_a_u_out)
        p_y__k_u_out = self.p_y__k((q_y__x_a_u_out,q_k__y_u_out))
        q_z__y_x_a_u_out = self.q_z__y_x_a((q_y__x_a_u_out,unsupervised_image,q_a__x_u_out))
        p_x__y_z_u_out = self.p_x__y_z((unsupervised_image,q_y__x_a_u_out,q_z__y_x_a_u_out))
        p_a__x_y_z_u_out = self.p_a__x_y_z((q_a__x_u_out,unsupervised_image,
            q_y__x_a_u_out,q_z__y_x_a_u_out))

        return (q_a__x_u_out,q_y__x_a_u_out, q_k__y_u_out,p_y__k_u_out,
                q_z__y_x_a_u_out,p_x__y_z_u_out,p_a__x_y_z_u_out,
                q_a__x_s_out,q_y__x_a_s_out, q_k__y_s_out,p_y__k_s_out,
                q_z__y_x_a_s_out,p_x__y_z_s_out,p_a__x_y_z_s_out)
