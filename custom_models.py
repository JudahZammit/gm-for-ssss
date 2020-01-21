from custom_layers import (q_y__x_s,q_k__y_s,p_y__k_s,q_z__y_x_s,p_x__y_z_s)
from tensorflow.keras import Model

class Supervised_GM(Model):

    def __init__(self):
        super(Supervised_GM,self).__init__()

        self.q_y__x_s = q_y__x_s()
        self.q_k__y_s = q_k__y_s()
        self.p_y__k_s = p_y__k_s()
        self.q_z__y_x_s = q_z__y_x_s()
        self.p_x__y_z_s = p_x__y_z_s()

    def call(self,inputs):
        supervised_image,supervised_mask = inputs

        q_y__x_s_out = self.q_y__x_s((supervised_mask,supervised_image))
        q_k__y_s_out = self.q_k__y_s(supervised_mask)
        p_y__k_s_out = self.p_y__k_s((supervised_mask,q_k__y_s_out))
        q_z__y_x_s_out = self.q_z__y_x_s((supervised_mask,supervised_image))
        p_x__y_z_s_out = self.p_x__y_z_s((supervised_image,supervised_mask,q_z__y_x_s_out))

        return  q_y__x_s_out,q_k__y_s_out,p_y__k_s_out,q_z__y_x_s_out,p_x__y_z_s_out

