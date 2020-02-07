from caerus_layers import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from metrics import IouCoef


class Caerus(Model):
    
    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(Caerus,self).__init__()
        
        self.p_x__e1 = p_x__e1(
                Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.f_d1__d2_z1_k1 = f_d1__d2_z1_k1(
                Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_d2__d3_z2_k2 = f_d2__d3_z2_k2(
                Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_d3__d4_z3_k3 = f_d3__d4_z3_k3(
                Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_d4__z4_k4_z5_k5 = f_d4__z4_k4_z5_k5(
                Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.p_y__k1 = p_y__k1(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.p_k1__k2 = p_k1__k2(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.p_k2__k3 = p_k2__k3(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.p_k3__k4 = p_k3__k4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.p_k4__k5 = p_k4__k5(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.p_k5 = p_k5(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.p_z1__z2 = p_z1__z2(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.p_z2__z3 = p_z2__z3(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.p_z3__z4 = p_z3__z4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.p_z4__z5 = p_z4__z5(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.p_z5 = p_z5(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.f_e1__x = f_e1__x(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.f_e2__e1 = f_e2__e1(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.f_e3__e2 = f_e3__e2(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.f_e4__e3 = f_e4__e3(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.q_z1__z2_e1 = q_z1__z2_e1(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_z2__z3_e2 = q_z2__z3_e2(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_z3__z4_e3 = q_z3__z4_e3(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_z4__z5_e4 = q_z4__z5_e4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_z5__e4 = q_z5__e4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.q_k1__k2_e1 = q_k1__k2_e1(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_k2__k3_e2 = q_k2__k3_e2(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_k3__k4_e3 = q_k3__k4_e3(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_k4__k5_e4 = q_k4__k5_e4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_k5__e4 = q_k5__e4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.q_k1__y = q_k1__y(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_k2__k1 = q_k2__k1(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_k3__k2 = q_k3__k2(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_k4__k3 = q_k4__k3(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        self.q_k5__k4 = q_k5__k4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.iou = IouCoef()
        self.cce = CategoricalCrossentropy() 
        self.bce = BinaryCrossentropy()


    def call(self,inputs):
        x_s,y_s,x_u = inputs

        #holds all the outputs
        out = {}

        # collect all samples from the inference network
        out['e1_s'] = self.f_e1__x(x_s)
        out['e2_s'] = self.f_e2__e1(out['e1_s'])
        out['e3_s'] = self.f_e3__e2(out['e2_s'])
        out['e4_s'] = self.f_e4__e3(out['e3_s'])

        out['e1_u'] = self.f_e1__x(x_u)
        out['e2_u'] = self.f_e2__e1(out['e1_u'])
        out['e3_u'] = self.f_e3__e2(out['e2_u'])
        out['e4_u'] = self.f_e4__e3(out['e3_u'])

        out['k1_sample_s'] = self.q_k1__y(y_s)
        out['k2_sample_s'] = self.q_k2__k1(out['k1_sample_s'])
        out['k3_sample_s'] = self.q_k3__k2(out['k2_sample_s'])
        out['k4_sample_s'] = self.q_k4__k3(out['k3_sample_s']) 
        out['k5_sample_s'] = self.q_k5__k4(out['k4_sample_s']) 

        out['k5_sample_u'] = self.q_k5__e4(
            (out['e4_u'])) 
        out['k4_sample_u'] = self.q_k4__k5_e4(
                (out['k5_sample_u'],out['e4_u'])) 
        out['k3_sample_u'] = self.q_k3__k4_e3(
                (out['k4_sample_u'],out['e3_u']))
        out['k2_sample_u'] = self.q_k2__k3_e2(
                (out['k3_sample_u'],out['e2_u']))
        out['k1_sample_u'] = self.q_k1__k2_e1(
                (out['k2_sample_u'],out['e1_u']))

        out['z5_sample_u'] = self.q_z5__e4(
                (out['e4_u'])) 
        out['z4_sample_u'] = self.q_z4__z5_e4(
                (out['z5_sample_u'],out['e4_u'])) 
        out['z3_sample_u'] = self.q_z3__z4_e3(
                (out['z4_sample_u'],out['e3_u']))
        out['z2_sample_u'] = self.q_z2__z3_e2(
                (out['z3_sample_u'],out['e2_u']))
        out['z1_sample_u'] = self.q_z1__z2_e1(
                (out['z2_sample_u'],out['e1_u']))

        out['z5_sample_s'] = self.q_z5__e4(
                (out['e4_s'])) 
        out['z4_sample_s'] = self.q_z4__z5_e4(
                (out['z5_sample_s'],out['e4_-'])) 
        out['z3_sample_s'] = self.q_z3__z4_e3(
                (out['z4_sample_-'],out['e3_s']))
        out['z2_sample_s'] = self.q_z2__z3_e2(
                (out['z3_sample_s'],out['e2_s']))
        out['z1_sample_s'] = self.q_z1__z2_e1(
                (out['z2_sample_s'],out['e1_s']))

        # included as outputs to ensure that the 
        # weights are traned
        out['p_k1_s'] = self.p_k1__k2(
                (out['k1_sample_s'],out['k2_sample_s']))
        out['p_k2_s'] = self.p_k2__k3(
                (out['k2_sample_s'],out['k3_sample_s']))
        out['p_k3_s'] = self.p_k3__k4(
                (out['k3_sample_s'],out['k4_sample_s']))
        out['p_k4_s'] = self.p_k4__k5(
                (out['k4_sample_s'],out['k5_sample_s']))
        out['p_k5_s'] = self.p_k5(out['k5_sample_s'])

        # included as outputs to ensure that the 
        # weights are traned
        out['p_k1_u'] = self.p_k1__k2(
                (out['k1_sample_u'],out['k2_sample_u']))
        out['p_k2_u'] = self.p_k2__k3(
                (out['k2_sample_u'],out['k3_sample_u']))
        out['p_k3_u'] = self.p_k3__k4(
                (out['k3_sample_u'],out['k4_sample_u']))
        out['p_k4_u'] = self.p_k4__k5(
                (out['k4_sample_u'],out['k5_sample_u']))
        out['p_k5_u'] = self.p_k5(out['k5_sample_u'])
        
        # included as outputs to ensure that the 
        # weights are traned
        out['p_z1_s'] = self.p_z1__z2(
                (out['z1_sample_s'],out['z2_sample_s']))
        out['p_z2_s'] = self.p_z2__z3(
                (out['k2_sample_s'],out['k3_sample_s']))
        out['p_z3_s'] = self.p_z3__z4(
                (out['z3_sample_s'],out['z4_sample_s']))
        out['p_z4_s'] = self.p_z4__z5(
                (out['z4_sample_s'],out['z5_sample_s']))
        out['p_z5_s'] = self.p_z5(out['z5_sample_s'])
        
        # included as outputs to ensure that the 
        # weights are traned
        out['p_z1_u'] = self.p_z1__z2(
                (out['z1_sample_u'],out['z2_sample_u']))
        out['p_z2_u'] = self.p_z2__z3(
                (out['k2_sample_u'],out['k3_sample_u']))
        out['p_z3_u'] = self.p_z3__z4(
                (out['z3_sample_u'],out['z4_sample_u']))
        out['p_z4_u'] = self.p_z4__z5(
                (out['z4_sample_u'],out['z5_sample_u']))
        out['p_z5_u'] = self.p_z5(out['z5_sample_u'])

        out['d4_s'] = self.f_d4__z4_k4_z5_k5(
                (out['z4_sample_s'],out['k4_sample_s'],
                    out['z5_sample_s'],out['k5_sample_s']))
        out['d3_s'] = self.f_d3__d4_z3_k3(
                (out['d4_s'],out['z3_sample_s'],out['k3_sample_s']))
        out['d2_s'] = self.f_d2__d3_z2_d2(
                (out['d3_s'],out['z2_sample_s'],out['k2_sample_s']))
        out['d1_s'] = self.f_d1__d2_z1_k1(
                (out['d2_s'],out['z1_sample_s'],out['k1_sample_s']))

        # these are the "real" outputs
        out['x_u_reconstructed'] = self.p_x__e1(out['e1_u'])
        out['x_s_reconstructed'] = self.p_x__e1(out['e1_s'])
        out['y_s_reconstructed'] = self.p_y__k1(out['k1_s'])
        # DO NOT add it as an output
        # then the weight will get trained
        y_prediction = self.p_y__k1(out['k1_u'])
        
        n_log_p_x__e1_u = self.bce(x_u,out['x_u_reconstructed'])
        n_log_p_x__e1_s = self.bce(x_s,out['x_s_reconstructed'])
        n_log_p_y__k_1_s = self.cce(y_s,out['y_s_reconstructed'])
        #this iou is for validation
        #it makes no sense while training
        iou = self.iou((mask,cat_param))
        
        self.add_loss(n_log_p_x__e1)
        self.add_loss(n_log_p_x__e1_s)
        self.add_loss(n_log_p_y__k_1_s)
        self.add_metric(iou,name= 'IOU', aggregation= 'mean')

        return out

