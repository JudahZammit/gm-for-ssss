from caerus_layers import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from metrics import IouCoef
from stochastic_unet_layers import PointDecoderLayer,PointEncoderLayer
from tensorflow.keras.utils import plot_model

class FuncCaerus():
    
    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(FuncCaerus,self).__init__()
        
        self.p_x__d1 = p_x__d1(
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
        
        self.p_k1__k2 = PointDecoderLayer(16,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = False,name = 'k2--k1')
        self.p_k2__k3 = PointDecoderLayer(32,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = False,name = 'k3--k2')
        self.p_k3__k4 =PointDecoderLayer(64,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = False,name = 'k4--k3')
        self.p_k4__k5 = PointDecoderLayer(256,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = False,name = 'k5--k4')
        self.p_k5 = p_k5(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.p_z1__z2 = PointDecoderLayer(16,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = False,name = 'z2--z1')
        self.p_z2__z3 = PointDecoderLayer(32,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = False,name = 'z3--z2')
        self.p_z3__z4 = PointDecoderLayer(64,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = False,name = 'z4--z3')
        self.p_z4__z5 = PointDecoderLayer(128,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = False,name = 'z5--z4')
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
        
        self.q_z1__z2_e1 = PointDecoderLayer(16,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = True,name = 'e1-z2--z1')
        self.q_z2__z3_e2 = PointDecoderLayer(32,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = True,name = 'e2-z3--z2')
        self.q_z3__z4_e3 =PointDecoderLayer(64,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = True,name = 'e3-z4--z3')
        self.q_z4__z5_e4 = PointDecoderLayer(128,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = True,name = 'e4-z5--z4')
        self.q_z5__e4 = q_z5__e4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.q_k1__k2_e1 = PointDecoderLayer(16,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = True,name = 'e1-k2--k1')
        self.q_k2__k3_e2 = PointDecoderLayer(32,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = True,name = 'e2-k3--k2')
        self.q_k3__k4_e3 = PointDecoderLayer(64,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = True,name = 'e3-k4--k3')
        self.q_k4__k5_e4 = PointDecoderLayer(128,Batch_Norm = Batch_Norm, 
                dropout = dropout,Skip = True,name = 'e4-k5--k4')
        self.q_k5__e4 = q_k5__e4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.q_k1__y = PointEncoderLayer(16,Batch_Norm = Batch_Norm,
                dropout=dropout,name = 'y--k1')
        self.q_k2__k1 = PointEncoderLayer(32,Batch_Norm = Batch_Norm,
                dropout=dropout,name = 'k1--k2')
        self.q_k3__k2 = PointEncoderLayer(64,Batch_Norm = Batch_Norm,
                dropout=dropout,name = 'k2--k3')
        self.q_k4__k3 = PointEncoderLayer(128,Batch_Norm = Batch_Norm,
                dropout=dropout,name = 'k3--k4')
        self.q_k5__k4 = q_k5__k4(Batch_Norm = Batch_Norm, 
                dropout = dropout)
        
        self.iou = IouCoef()
        self.cce = CategoricalCrossentropy() 
        self.bce = BinaryCrossentropy()


    def viz(self):
        x_s = keras.layers.Input((64,64,3),name = 'x_s')
        x_u = keras.layers.Input((64,64,3),name = 'x_u')
        y_s = keras.layers.Input((64,64,21),name = 'y_s')

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

        out['z5_sample_u'] = self.q_z5__e4(
                (out['e4_u'])) 

        out['z5_sample_s'] = self.q_z5__e4(
                (out['e4_s'])) 

        # included as outputs to ensure that the 
        # weights are traned
        out['p_k5_s'] = self.p_k5(out['k5_sample_s'])
        out['p_k4_s'] = self.p_k4__k5(
                [[out['k5_sample_s']],[]])
        out['p_k3_s'] = self.p_k3__k4(
                [[out['p_k4_s']],[]])
        out['p_k2_s'] = self.p_k2__k3(
                [[out['p_k3_s']],[]])
        out['p_k1_s'] = self.p_k1__k2(
                [[out['p_k2_s']],[]])
        
        # included as outputs to ensure that the 
        # weights are traned
        out['p_k5_u'] = self.p_k5(out['k5_sample_u'])
        out['p_k4_u'] = self.p_k4__k5(
                [[out['k5_sample_u']],[]])
        out['p_k3_u'] = self.p_k3__k4(
                [[out['p_k4_u']],[]])
        out['p_k2_u'] = self.p_k2__k3(
                [[out['p_k3_u']],[]])
        out['p_k1_u'] = self.p_k1__k2(
                [[out['p_k2_u']],[]])
     
        # included as outputs to ensure that the 
        # weights are 
        out['p_z5_s'] = self.p_z5(out['z5_sample_s'])
        out['p_z4_s'] = self.p_z4__z5(
                [[out['z5_sample_s']],[]])
        out['p_z3_s'] = self.p_z3__z4(
                [[out['p_z4_s']],[]])
        out['p_z2_s'] = self.p_z2__z3(
                [[out['p_z3_s']],[]])
        out['p_z1_s'] = self.p_z1__z2(
                [[out['p_z2_s']],[]])
        
        # included as outputs to ensure that the 
        # weights are traned
        out['p_z5_u'] = self.p_z5(out['z5_sample_u'])
        out['p_z4_u'] = self.p_z4__z5(
                [[out['z5_sample_u']],[]])
        out['p_z3_u'] = self.p_z3__z4(
                [[out['p_z4_u']],[]])
        out['p_z2_u'] = self.p_z2__z3(
                [[out['p_z3_u']],[]])
        out['p_z1_u'] = self.p_z1__z2(
                [[out['p_z2_u']],[]])

        out['d4_s'] = self.f_d4__z4_k4_z5_k5(
                (out['p_z4_s'],out['p_k4_s'],
                    out['z5_sample_s'],out['k5_sample_s']))
        out['d3_s'] = self.f_d3__d4_z3_k3(
                (out['d4_s'],out['p_z3_s'],out['p_k3_s']))
        out['d2_s'] = self.f_d2__d3_z2_k2(
                (out['d3_s'],out['p_z2_s'],out['p_k2_s']))
        out['d1_s'] = self.f_d1__d2_z1_k1(
                (out['d2_s'],out['p_z1_s'],out['p_k1_s']))
        
        out['d4_u'] = self.f_d4__z4_k4_z5_k5(
                (out['p_z4_u'],out['p_k4_u'],
                    out['z5_sample_u'],out['k5_sample_u']))
        out['d3_u'] = self.f_d3__d4_z3_k3(
                (out['d4_u'],out['p_z3_u'],out['p_k3_u']))
        out['d2_u'] = self.f_d2__d3_z2_k2(
                (out['d3_u'],out['p_z2_u'],out['p_k2_u']))
        out['d1_u'] = self.f_d1__d2_z1_k1(
                (out['d2_u'],out['p_z1_u'],out['p_k1_u']))

        # these are the "real" outputs
        out['x_u_reconstructed'] = self.p_x__d1(out['d1_u'])
        out['x_s_reconstructed'] = self.p_x__d1(out['d1_s'])
        out['y_s_reconstructed'] = self.p_y__k1(out['p_k1_s'])
        # DO NOT add it as an output
        # then the weight will get trained
        y_prediction = self.p_y__k1(out['p_k1_u'])
        
        n_log_p_x__e1_u = self.bce(x_u,out['x_u_reconstructed'])
        n_log_p_x__e1_s = self.bce(x_s,out['x_s_reconstructed'])
        n_log_p_y__k1_s = self.cce(y_s,out['y_s_reconstructed'])
        #this iou is for validation
        #it makes no sense while training
        iou = self.iou((y_s,y_prediction))
        true_iou = self.iou((y_s,out['y_s_reconstructed']))

        
        actual_out = {}
        actual_out['x_u'] = out['x_u_reconstructed']
        actual_out['x_s'] = out['x_s_reconstructed']
        actual_out['y_s'] = out['y_s_reconstructed']
        actual_out['z5_s'] = out['z5_sample_s']
        actual_out['z5_u'] = out['z5_sample_u']
        actual_out['k5_s'] = out['k5_sample_s']
        actual_out['k5_u'] = out['k5_sample_u']
        
        model = Model(inputs = [x_s,x_u,y_s],outputs = actual_out)
        model.compile(tf.keras.optimizers.Adam(lr=.001,clipnorm = 1.,clipvalue = 0.5))

        plot_model(model,to_file='model.png',expand_nested = True)

