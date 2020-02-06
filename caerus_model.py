from caerus_layers import *

class Caerus(Model):
    
    def __init__(self,dropout = 0.0,Batch_Norm = False):
        super(Caerus,self).__init__()
        
        self.p_x__e1 = p_x__e1(Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.f_d1__d2_z1_k1 = f_d1__d2_z1_k1(Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_d2__d3_z2_k2 = f_d2__d3_z2_k2(Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_d3__d4_z3_k3 = f_d3__d4_z3_k3(Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_d4__z4_k4_z5_k5 = f_d4__z4_k4_z5_k5(Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.p_y__k1 = p_y__k1(Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.p_k1__k2 = p_k1__k2(Batch_Norm = Batch_Norm, dropout = dropout)
        self.p_k2__k3 = p_k2__k3(Batch_Norm = Batch_Norm, dropout = dropout)
        self.p_k3__k4 = p_k3__k4(Batch_Norm = Batch_Norm, dropout = dropout)
        self.p_k4__k5 = p_k4__k5(Batch_Norm = Batch_Norm, dropout = dropout)
        self.p_k5 = p_k5(Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.p_z1__z2 = p_z1__z2(Batch_Norm = Batch_Norm, dropout = dropout)
        self.p_z2__z3 = p_z2__z3(Batch_Norm = Batch_Norm, dropout = dropout)
        self.p_z3__z4 = p_z3__z4(Batch_Norm = Batch_Norm, dropout = dropout)
        self.p_z4__z5 = p_z4__z5(Batch_Norm = Batch_Norm, dropout = dropout)
        self.p_z5 = p_z5(Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.f_e1__x = f_e1__x(Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_e2__e1 = f_e2__e1(Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_e3__e2 = f_e3__e2(Batch_Norm = Batch_Norm, dropout = dropout)
        self.f_e4__e3 = f_e4__e3(Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.q_z1__z2_e1 = q_z1__z2_e1(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_z2__z3_e2 = q_z2__z3_e2(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_z3__z4_e3 = q_z3__z4_e3(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_z4__z5_e4 = q_z4__z5_e4(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_z5__e4 = q_z5__e4(Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.q_k1__k2_e1 = q_k1__k2_e1(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_k2__k3_e2 = q_k2__k3_e2(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_k3__k4_e3 = q_k3__k4_e3(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_k4__k5_e4 = q_k4__k5_e4(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_k5__e4 = q_k5__e4(Batch_Norm = Batch_Norm, dropout = dropout)
        
        self.q_k1__y = q_k1__y(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_k2__k1 = q_k2__k1(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_k3__k2 = q_k3__k2(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_k4__k3 = q_k4__k3(Batch_Norm = Batch_Norm, dropout = dropout)
        self.q_k5__k4 = q_k5__k4(Batch_Norm = Batch_Norm, dropout = dropout)


Caerus()
