#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#NO RISK,epochs =1000, c_index= 0.7043926169695235 ？ remove log_var[i] seems as no difference.  0.7222063242237802
#only: data_mRNA:0.7002432393761625,                                                             0.7053941908713693
#only: data_methylation: 0.5142366576048075                                                      0.5180998712262126              
"""
Demonstrates how the partial likelihood from a Cox proportional hazards
model can be used in a NN loss function. An example shows how a NN with
one linear-activation layer and the (negative) log partial likelihood as
loss function produces approximately the same predictor weights as a Cox
model fit in a more conventional way.
"""
import datetime
import pandas as pd
import numpy as np
import keras
#from lifelines import CoxPHFitter
#from lifelines.datasets import load_kidney_transplant

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import theano
from keras.layers import Dropout, Activation, Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Input,Embedding

from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from lifelines.utils import concordance_index
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.layers.core import Reshape
from keras.utils.vis_utils import plot_model
from keras.initializers import Constant
from keras.layers import  Layer
from sklearn.preprocessing import minmax_scale
import tensorflow.compat.v1 as tf1
from keras.layers import  concatenate
from cca_layer import CCA
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras_custom import cca_loss, batch_cca_loss,  PickMax, UnitNormWithNonnneg
from keras.regularizers import l1
tf1.disable_v2_behavior()

##############################
##############################
#kidtx = pd.read_csv('brca_Surv_data_stage_age.csv',usecols=['MEgreenyellow',	'MEgreen','MEturquoise'	,'MEmagenta','MEbrown',	'MEred',	'MEpink',	'MEblack',	'MEpurple',	'MEblue',	'MEyellow',	'MEgrey','erged_data5_stage','erged_data6_Age','V1','erged_data33'])
#kidtx = pd.read_csv('brca_Surv_data_methylation_mRNA_lmqcm.csv')
#dataX1 =kidtx.drop(["Unnamed: 0","ID","V1","erged_data33"], axis = 1).values
#y = np.transpose(np.array((kidtx["V1"], kidtx["erged_data33"]))) # V1=time; erged_data33=status

#kidtx = pd.read_csv('brca_Surv_data_methylation_mRNA_all_lmqcm.csv')
#dataX1 =kidtx.drop(["Unnamed: 0","ID","V2.x","V3.x"], axis = 1).values
#y = np.transpose(np.array((kidtx["V2.x"], kidtx["V3.x"]))) # V1=time; erged_data33=status
kidtx = pd.read_csv('brca_Surv_data_methylation_mRNA_important_raw_p001_414.csv')

#dataX01 = kidtx.drop(["Unnamed: 0","V1", "erged_data33","X"], axis = 1)
#dataX1 = kidtx.drop(["Unnamed: 0","V1", "erged_data33","X"], axis = 1).values

dataX01 = kidtx.drop(["Unnamed: 0","X.1","V1", "erged_data33","X"], axis = 1)
dataX1 = kidtx.drop(["Unnamed: 0","X.1","V1", "erged_data33","X"], axis = 1).values

y = np.transpose(np.array((kidtx["V1"], kidtx["erged_data33"]))) # V1=time; erged_data33=status

[ m0,n0] = dataX1.shape

col_name=list(dataX01)
y_name=["V1", "erged_data33"]

col_name_meth=col_name[0:679]
col_name_mRNA=col_name[679:n0]

dataX = np.asarray(dataX1)
dataX =minmax_scale(dataX ) 
data_methylation=dataX[:,0:679]
data_mRNA=dataX[:,679:n0]
#dataX=data_mRNA
#dataX=data_methylation
#data_mRNA=dataX 
[ m,n] = dataX.shape
[ m1,n1] = data_methylation.shape
[ m2,n2] = data_mRNA.shape

 
#dataX = dataX.reshape(m,1,n)
x=dataX
#data_methylation = data_methylation.reshape(m1,1,n1)
#data_mRNA = data_mRNA.reshape(m2,1,n2)

ytime=np.transpose(np.array(kidtx["V1"])) # only V1=time;
ystatus= np.transpose(np.array(kidtx["erged_data33"])) #only erged_data33=status
## Build model structure
#model = Sequential()
#model.add(Dense(units = 20, activation = "tanh", use_bias = False, input_shape=[12]))
#model.add(Dense(1, kernel_initializer='normal'))
# Define loss function
# y_true = (n x 2) array with y_true[i, 0] the survival time
#          for individual i and y_true[i, 1] the event indicator
# y_pred = (n x 1) array of linear predictor (x * beta) values
from keras.utils import np_utils
ystatus2= np_utils.to_categorical(ystatus)



##################################################################################################







def neg_log_pl(y_true, y_pred):
    # Sort by survival time (descending) so that
    # - If there are no tied survival times, the risk set
    #   for event i is individuals 0 through i
    # - If there are ties, and time[i - k] through time[i]
    #   represent all times equal to time[i], then the risk set
    #   for events i - k through i is individuals 0 through i
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
#    print('time:',time)
    # For each set of tied survival times, put the sum of the
    # corresponding risk (exp[x * beta]) values at the first
    # position in the sorted array of times while setting other
    # positions to 0 so that the cumsum operation will result                         tied 关联的
    # in each of the positions having the same sum of risks
#    for i in range(time.shape[0] - 1, 0, -1):
##         Going from smallest survival times to largest
#        if time[i] == time[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            xbeta[i - 1] = xbeta[i - 1] + xbeta[i]
#            xbeta[i] = 0
#        
#    risk = K.exp(xbeta)        
#    for i in range(time.shape[0] - 1, 0, -1):
###         Going from smallest survival times to largest
#        if time[i] == time[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0        
    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk) #这个函数的功能是返回给定axis上的累计和
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms   #cast将x的数据格式转化成dtype
#    neg_log=-K.sum(loglik)
#    
#    ##############################################
#    diff = K.categorical_crossentropy(y_pred[i]-event)
#    
#    def Loss2(y_pred, y_true, log_vars):
#   loss = 0
#   for i in range(len(y_pred)):
#       precision = K.exp(-log_vars[i])
#       diff = (y_pred[i]-y_true[i])**2.
#       loss += K.sum(precision * diff + log_vars[i], -1)
#   return K.mean(loss)
    
    
    
    
    return -(loglik)
#    return -K.sum(loglik)

def LOSS_L2(y_true, y_pred):
#    MAX_SEQ_LEN=1
    BATCH_SIZE=int(k_n.get_value())
    L2_NORM = 0.001
    
   
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
#    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred, indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    event =K.gather(y_true[:,1], indices = sorting.indices)
    
#    self.preds = preds
    final_dead_rate = xbeta
    final_survival_rate=1.0-final_dead_rate
    predict=K.stack([final_survival_rate, final_dead_rate])
    cross_entropy = - K.cumsum( event*K.log(final_dead_rate))
    cost=cross_entropy
    
#    final_survival_rate=tf.subtract(tf.constant(1.0, dtype=tf.float32), final_dead_rate)
#    predict = tf.transpose(tf.stack([final_survival_rate, final_dead_rate]), name="predict")
##    predict =predict[-1,:,:]
#    cross_entropy = -tf.reduce_sum( event*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
#    tvars = tf.trainable_variables()  #tf.trainable_variables 返回所有 当前计算图中 在获取变量时未标记 trainable=False 的变量集合
#    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars ]) * L2_NORM
#    cost = tf.add(cross_entropy, lossL2, name = "cost")  / BATCH_SIZE

    
#    Loss2=K.categorical_crossentropy( event, xbeta)
    Loss=cost
    return Loss






def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample =(x.shape[0]).value
	matrix_ones = tf.ones([n_sample, n_sample],tf.int32)
	indicator_matrix = tf.linalg.band_part(matrix_ones,-1,0) #下三角形

	return(indicator_matrix)
    
def neg_log_pl_1(y_true, y_pred):
    
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    ytime = K.gather(y_true[:, 0], indices = sorting.indices)
    yevent = K.gather(y_true[:, 1], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    matrix_risk = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_I = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_max= tf.zeros([int(k_n.get_value())],tf.float32)
     
#    weights = K.cast(K.abs(K.argmax(ytime, axis=0) - K.argmax(risk, axis=0))/(int(k_n.get_value()) - 1), dtype='float32')
    
    kk_ytime_train =k_ytime_train.get_value()
    for i in range(ytime.shape[0] - 1, 0, -1):
        # Going from smallest survival times to largest
#        tf.cond(tf.greater_equal(ytime[i-1], ytime[i]),  risk[i - 1]: (risk[i - 1] + risk[i]),  risk[i - 1]: 0)
#        risk[i - 1]=tf.where(tf.equal(ytime[i-1], ytime[i]), risk[i - 1] + risk[i],  risk[i - 1])
#        risk_i,risk_i-1 = risk_ab( risk[i], risk[i - 1])
#        sess = tf.Session()
#        array_i = np.array(kk_ytime_train[i])
#        array_i_1 = np.array(kk_ytime_train[i-1])
#        feed_dict = {a: array_i, b: array_i_1}
#        tile_b_value = sess.run(tile_tensor_b, feed_dict = feed_dict)
#        print(tile_b_value)
#        def true_proc():
#            risk[i - 1] = risk[i - 1] + risk[i]
#            return risk[i - 1]
#		
#        def false_proc():
#	         risk[i] = 0
#	         return risk[i]
#        def true_proc():
#            result=2
#            return result
#		
#        def false_proc():
#            result=3
#            return result
        a0=tf.constant(0,dtype=tf.float32)      
        a1=K.cast(risk[i - 1],dtype=tf.float32)
        a2=K.cast(risk[i],dtype=tf.float32)
        a3=K.cast(a1+a2,dtype=tf.float32)
        risk_i_1= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: a3, lambda:a1)
        risk_i= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: a0, lambda:a2)
        
#        part1 = matrix_risk[:i-1]
#              
#        part2 = matrix_risk[i+1:]
#        val_i_1= risk_i_1
#        val_i = risk_i
#        matrix_risk2=tf.concat([part1,val_i_1,val_i,part2], axis=0)
#        matrix_risk=K.cast(risk,dtype=tf.float32)
        # 生成一个one_hot张量，长度与tensor_1相同，修改位置为1
        shape = risk.get_shape().as_list()
        one_hot_i = tf.one_hot(i,shape[0],dtype=tf.float32)
        one_hot_i_1 = tf.one_hot(i-1,shape[0],dtype=tf.float32)
       # 做一个减法运算，将one_hot为一的变为原张量该位置的值进行相减
        new_tensor = matrix_risk+risk_i_1 * one_hot_i_1
        matrix_risk = new_tensor+ risk_i * one_hot_i
        
        
        
#        rec_ij = a2 / a1
#        b1=tf.constant(1,dtype=tf.float32) 
#        max_rec_ij= tf.cond(tf.less( rec_ij,  b1), lambda:1-rec_ij, lambda:a0)
#        
#        Iij=tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: b1, lambda:a0)
#        mul_ij=Iij * max_rec_ij
#        
#        one_hot_I = tf.one_hot(i,shape[0],dtype=tf.float32)
#        
#        matrix_I = matrix_I+mul_ij*one_hot_I
#        matrix_max=matrix_max+max_rec_ij*one_hot_I
#
#            
##    cost2 = ( K.sum(K.dot(matrix_I,matrix_max)) )  
#    cost2 =( 1+weights) * ( K.sum(matrix_I) )    
#    cost3 = (1+weights) *( K.sum(matrix_risk) )  
    
#    cost2 =(1.0 + weights) * ( K.sum(matrix_I) )    
#    cost3 =(1.0 + weights) * ( K.sum(matrix_risk) )       
#        new_tensor = tf.concat([part1,val,part2], axis=0)

#        a1_list = []
#        a1_list.append(risk_i_1)
#        matrix_risk[i - 1] = tf.stack(a1_list)
#        risk[i - 1]=K.cast( risk_i_1,dtype=risk_i_1.dtype)
#        
#        risk[i]=K.cast( risk_i_1,dtype=risk_i.dtype)
#        risk_i_1= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: risk[i - 1], lambda:(risk[i - 1] + risk[i]))
#        with tf.Session() as sess: 
#            sess.run(tf.global_variables_initializer()) 
#            y1=sess.run([risk_i_1]) 
#            print(y1)
#        risk_i_1= tf.cond(tf.greater_equal( kk_ytime_train[i],  kk_ytime_train[i-1]), true_fn = true_proc, false_fn = false_proc)
#        n1 = ytime[i]!=tf.constant(0)
#        n2 = ytime[i-1]!=tf.constant(0)
#        if kk_ytime_train[i]<=kk_ytime_train[i-1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0
    
 
#    N=ytime.shape[0]
#    R_matrix = tf.zeros([N,N],tf.int32)
#    
#   
#    
#    for i in range(N - 1, 0, -1):
#        # Going from smallest survival times to largest
#        if ytime[i] == ytime[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0
    n_observed = tf.reduce_sum(yevent,0)
    ytime_indicator = R_set(ytime)
    
    
#    R_matrix1=add_rankLoss_numba(ytime)
    
#    pool = ThreadPool()
#    R_matrix1=pool.map(add_rankLoss_numba(ytime))
#    pool.close()
#    pool.join()
    
#    for i in range(N):
#        for j in range(N):
#            R_matrix[i,j] = y_true[j] >= y_true[i]
            
   
	###if gpu is being used
#    if torch.cuda.is_available():
#        ytime_indicator = ytime_indicator.cuda()
	###
#    risk_set_sum = (ytime_indicator)*(K.exp(y_pred)) 
#    risk_set_sum =  K.sum(K.cast(ytime_indicator, dtype = y_pred.dtype)*(K.exp(xbeta)),axis=1)
    risk_set_sum =  K.sum(matrix_risk)
#    risk_set_sum =  K.sum(K.cast(K.exp(xbeta), dtype = ytime_indicator.dtype)*(ytime_indicator),axis=1,dtype=float32)
    
    diff = xbeta - K.log(risk_set_sum)
#    sum_diff_in_observed = K.cast(K.transpose(diff), dtype = yevent.dtype)*(yevent)
    sum_diff_in_observed = K.cast(yevent, dtype = diff.dtype)*(diff)
#    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    cost = (- K.sum(sum_diff_in_observed))
#    cost = (- K.sum(sum_diff_in_observed / n_observed))
    
    
#    loss = (cost + regularize_layer_params(self.network,l1) * L1_reg
#            + regularize_layer_params(self.network, l2) * L2_reg
#        )
    
    λ1=1
    λ2=0.0
    λ3=0.0
    
#    return(λ1*cost+λ2*cost2+λ3*cost3)
    return(λ1*cost)


## C_index metric function

def c_index3(month,risk, status):

    c_index = concordance_index(np.reshape(month, -1), -np.reshape(risk, -1), np.reshape(status, -1))

    return c_index#def get_bi_lstm_model():  
#    model=Sequential()
#    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(1,input_dim0), merge_mode='concat'))
#
#    model.add(TimeDistributed(Dense(50, activation='tanh')))
#    model.add(Bidirectional(LSTM(50)))
#    model.add(Dropout(0.2))
#    model.add(Dense(1, activation='linear'))
#  
## Compile model
##    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
##        epsilon=None, decay=0.0, amsgrad=False), loss = neg_log_pl)
#    model.compile(optimizer='adam',  loss=neg_log_pl)
#    return model
############################################################################################
#def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):
#
#    grad_name = 'PyFuncGrad_' + str(np.random.randint(0, 1E+8))
#    tf.RegisterGradient(grad_name)(grad_func)
#    g = tf.get_default_graph()
#    with g.gradient_override_map({"PyFunc": grad_name}):
#        func1=tf.py_func(func, inp, Tout, stateful=stateful, name=name)
#        return func1    
###########################################################################################
def ordinal_loss0 (y_true, y_pred):
     
   
#    Y_hazard=k_ytime_train.get_value()
#    Y_survival=k_ystatus_train.get_value()
#   
#    t, H = unique_set(Y_hazard) # t:unique time. H original index.
#    
##    Y_survival=Y_survival.numpy()
##    risk=np.exp(score)
##    Y_hazard=Y_hazard.numpy()
#    actual_event_index = np.nonzero(Y_survival)[0]
#    H = [list(set(h) & set(actual_event_index)) for h in H]
#    n = [len(h) for h in H]
    
    
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    ytime = K.gather(y_true[:, 0], indices = sorting.indices)
    yevent = K.gather(y_true[:, 1], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    matrix_risk = tf.zeros([int(k_n.get_value())],tf.float32)
#    matrix_I = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_max= tf.zeros([int(k_n.get_value())],tf.float32)
    
    kk_t =k_ordinal_t.get_value()
    kk_n =k_ordinal_n.get_value()
    Hj =k_ordinal_H.get_value()
    a1=tf.constant(1,dtype=tf.float32) 
    for j in range(len(Hj)):
#        print('j:',j)
#        m=kk_n[j]
#        Hj=sum(H[j:],[])
        matrix_j = tf.zeros([int(k_n.get_value())],tf.float32)
        for i in range(1):
         # 生成一个one_hot张量，长度与tensor_1相同，修改位置为1
            for ii in  range(j,len(Hj)):
#                print('ii:',ii)
                risk_more_j=xbeta[Hj[ii]]
                risk_j=xbeta[Hj[j]]
            
                rec= a1-K.exp(risk_j-risk_more_j)
                rec2=tf.maximum(0.,rec)
            
                shape = risk.get_shape().as_list()
#                one_hot_j = tf.one_hot(H[j],shape[0],dtype=tf.float32)
                one_hot_more_j = tf.one_hot(Hj[ii],shape[0],dtype=tf.float32)
#                one_hot_more_j =tf.reduce_sum(one_hot_more_j0,axis=0)
               # 做一个减法运算，将one_hot为一的变为原张量该位置的值进行相减
#                new_tensor = matrix_risk+risk_j * one_hot_j
                matrix_j = matrix_j+ rec2 * one_hot_more_j
        #        tf.reduce_sum(tf.one_hot(sum(H[13:],[]),n1,dtype=tf.float32),axis=0)
        matrix_risk= (matrix_risk+ matrix_j) 
    cost2 = K.sum(matrix_risk)/(len(Hj))
#    cost2 = matrix_risk/(len(Hj))
    return cost2 
    
    
#    Y_true=Y_true.numpy()
#    Y_hazard0=Y_true[:,0]
#    Y_survival=Y_true[:,1]
##            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
##            Y_survival_train2=Y_survival_train1[-1,:,:] 
#    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_true.shape[0],1])
#    Y_hazard=Y_hazard1M[-1,:,-1]
#    
#    t, H = unique_set(Y_hazard) # t:unique time. H original index.
#    score=score.numpy()
##    Y_survival=Y_survival.numpy()
##    risk=np.exp(score)
#    Y_hazard=Y_hazard.numpy()
#    actual_event_index = np.nonzero(Y_survival)[0]
#    H = [list(set(h) & set(actual_event_index)) for h in H]
#    n = [len(h) for h in H]
#    
#    matrix_ones = np.zeros([Y_hazard.shape[0], Y_hazard.shape[0]])
#    total = 0.0
#    for j in range(len(t)):
##        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
#        total_1 = score[H[j]]
#        m = n[j]
#        total_2 = 0.0
#        for i in range(m):
#            matrix_ones[H[j],sum(H[j:],[])]=1
#            risk_more_j=np.exp(score[sum(H[j:],[])])
#            risk_j=np.exp(score[H[j]])
#            
#            rec=risk_j-risk_more_j
#            rec2=np.maximum(0,1-rec)
#            matrix_ones[H[j],sum(H[j:],[])]=rec2[:,-1]
#            subtotal = np.sum(matrix_ones[H[j],sum(H[j:],[])])
#            
##            subtotal = np.sum(np.exp(score[sum(H[j:],[])]) )
##            subtotal = np.log(np.absolute(subtotal + epsilon))
#            total_2 = total_2 + subtotal
#        total = total + total_2
#    return tf.to_float(total)  
#      
###############################################################################################
def unique_set(Y_hazard):

    a1 = Y_hazard#.numpy()
#    print('Y_hazard:',Y_hazard)
    # Get unique times
    t, idx = np.unique(a1, return_inverse=True)

    # Get indexes of sorted array
    sort_idx = np.argsort(a1)
#    print(sort_idx)
    # Sort the array using the index
    a_sorted =a1[sort_idx]# a1[np.int(sort_idx)]# a[tf.to_int32(sort_idx)]#
#    print('a_sorted:', a_sorted)
    # Find duplicates and make them 0
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))

    # Difference a[n+1] - a[n] of non zero indexes (Gives index ranges of patients with same timesteps)
    unq_count = np.diff(np.nonzero(unq_first)[0])

    # Split all index from single array to multiple arrays where each contains all indexes having same timestep
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))

    return t, unq_idx

###########################################################################################
def ordinal_loss (Y_true, score, epsilon=1e-8):
    Y_true=Y_true#.numpy()
    print('Y_true:',Y_true)
    Y_hazard=Y_true[:,0]
    print('Y_hazarde:',Y_hazard)
    Y_survival=Y_true[:,1]
#            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
#            Y_survival_train2=Y_survival_train1[-1,:,:] 
#    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_true.shape[0],1])
#    Y_hazard=Y_hazard1M[-1,:,-1]
    
    t, H = unique_set(Y_hazard) # t:unique time. H original index.
    score=score#.numpy()
#    Y_survival=Y_survival.numpy()
#    risk=np.exp(score)
    Y_hazard=Y_hazard#.numpy()
    actual_event_index = np.nonzero(Y_survival)[0]
    H = [list(set(h) & set(actual_event_index)) for h in H]
    n = [len(h) for h in H]
    
    matrix_ones = np.zeros([Y_hazard.shape[0], Y_hazard.shape[0]])
    total = 0.0
    for j in range(len(t)):
#        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
#        total_1 = score[H[j]]
        m = n[j]
        total_2 = 0.0
        for i in range(m):
            matrix_ones[H[j],sum(H[j:],[])]=1
            risk_more_j=np.exp(score[sum(H[j:],[])])
            risk_j=np.exp(score[H[j]])
            
            rec=risk_j-risk_more_j
            rec2=np.maximum(0,1-rec)
            matrix_ones[H[j],sum(H[j:],[])]=rec2[:,-1]
            subtotal = np.sum(matrix_ones[H[j],sum(H[j:],[])])
            
#            subtotal = np.sum(np.exp(score[sum(H[j:],[])]) )
#            subtotal = np.log(np.absolute(subtotal + epsilon))
            total_2 = total_2 + subtotal
        total = total + total_2
    return tf.to_float(total)  
###################################################################################################
###########################################################################################
def ordinal_loss_grad_numpy (Y_true, score, grad, epsilon=1e-8):
    Y_true=Y_true#.numpy()
    Y_hazard0=Y_true[:,0]
    Y_survival=Y_true[:,1]
#            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
#            Y_survival_train2=Y_survival_train1[-1,:,:] 
    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_true.shape[0],1])
    Y_hazard=Y_hazard1M[-1,:,-1]
    
    t, H = unique_set(Y_hazard) # t:unique time. H original index.
    score=score#.numpy()
#    Y_survival=Y_survival.numpy()
#    risk=np.exp(score)
    Y_hazard=Y_hazard#.numpy()
    actual_event_index = np.nonzero(Y_survival)[0]
    H = [list(set(h) & set(actual_event_index)) for h in H]
    n = [len(h) for h in H]
    
    matrix_ones = np.zeros([Y_hazard.shape[0], Y_hazard.shape[0]])
    total = 0.0
    for j in range(len(t)):
#        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
#        total_1 = score[H[j]]
        m = n[j]
        total_2 = 0.0
        for i in range(m):
            matrix_ones[H[j],sum(H[j:],[])]=1
            risk_more_j=np.exp(score[sum(H[j:],[])])
            risk_j=np.exp(score[H[j]])
            
            rec=risk_j-risk_more_j
            rec2=np.maximum(0,1-rec)
            matrix_ones[H[j],sum(H[j:],[])]=rec2[:,-1]
            subtotal = np.sum(matrix_ones[H[j],sum(H[j:],[])])
            
#            subtotal = np.sum(np.exp(score[sum(H[j:],[])]) )
#            subtotal = np.log(np.absolute(subtotal + epsilon))
            total_2 = total_2 + subtotal
        total = total + total_2
    dloss=np.sum(matrix_ones,axis=0)/100
    return np.float32(dloss)# * grad)
############################################################################################
def ordinal_loss_grad(op, grad):
   
   ys_true = op.inputs[0]
   ys_pred= op.inputs[1]
   
#    grad=op.inputs[4]
   tensor1=tf.py_func(ordinal_loss_grad_numpy, [  ys_true ,ys_pred, grad], grad.dtype),\
             tf.zeros(tf.shape(ys_pred)) 
#            tf.zeros(tf.shape( ys_true)), tf.zeros(tf.shape(ys_pred))
   return  tensor1
################################################################################################### 

def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):

    grad_name = 'PyFuncGrad_' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(grad_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        func1=tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        return func1
###########################################################################################        

def ordinal_loss_tf(ys_true, ys_pred):
    # use tf.py_func
     loss = py_func(ordinal_loss,
        [ys_true, ys_pred], [tf.float32],
        name = "ordinal_loss",
        grad_func = ordinal_loss_grad)[0]

     return loss    
###########################################################################################        
#@tf.custom_gradient
#def ordinal_loss_tf(ys_true, ys_pred):
#    # use tf.py_func
#    loss=tf.py_function(func=ordinal_loss, inp=[ys_true, ys_pred], Tout=tf.float32) 
##    loss = tf.py_func(mse_numpy, [y, y_predict], tf.float32, name='my_mse')
#
#    def grad(dy):
#        return tf.py_func(func=ordinal_loss_grad_numpy, inp=[ys_true, ys_pred, dy], Tout=tf.float32, name='my_grad')
#
#    return loss, grad    
#########################################################################################################################
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        log_var=self.log_vars
#        for y_true, y_pred, log_var ,ii in zip(ys_true, ys_pred, self.log_vars,self.nb_outputs):
        for i in range(self.nb_outputs):    
            precision = (K.exp(-log_var[i]))#**0.5
#            precision= tf.clip_by_value(precision, 0., 1.)
            if i==0:
                lossA=neg_log_pl(ys_true[i], ys_pred[i])
#                print("Neg_loss:",lossA.item())
            if i==1: 
#                 lossA = py_func(ordinal_loss,
#                     [ys_true[i-1], ys_pred[i]], [tf.float32],
#                     name = "ordinal_loss",
#                     grad_func = ordinal_loss_grad )[0]
#                 lossA.set_shape((ys_true[i].shape[0],))
#                lossA=LOSS_L2(ys_true[i], ys_pred[i])
                lossA=ordinal_loss0(ys_true[i-1], ys_pred[i])
#                lossA=neg_log_pl(ys_true[i-1], ys_pred[i])
#                print("Ordinal_loss:",lossA.item())
#                lossA=tf.py_function(func=ordinal_loss, inp=[ys_true[i], ys_pred[i]], Tout=tf.float32) 
#                lossA=ordinal_loss(Y_trueM, model_aux(tf.to_float(x_train0M)))
#                lossA=neg_log_pl_1(ys_true[i-1], ys_pred[i])
#            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
            loss += K.sum(precision * lossA+ log_var[i] , -1)
#            loss += K.sum(precision * lossA + log_var[i], -1)
        
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)
#############    #######################################################################
import theano
import theano.tensor
import theano.tensor.nlinalg as TN
import theano.tensor.basic as TB

def scca_loss0(y_true,y_pred):
   
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk) #这个函数的功能是返回给定axis上的累计和
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms   #cast将x的数据格式转化成dtype
    
    
    
    #11
#"""
#    Value to MINIMIZE
#"""
    covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
    diag_sum = tf.reduce_sum(tf.abs(tf.diag_part(covar_mat)))
    cca_score = tf.multiply(-1., diag_sum)
    #inter_sum = tf.reduce_sum(tf.abs(tf.matrix_band_part(covar_mat, 0, -1)))
    #cca_score = tf.multiply(-1., diag_sum - inter_sum) 
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cca_score + tf.add_n(reg_losses)
    return total_loss
    
    
    #2
    x1tx2 = K.placeholder(K.float32, shape=(x_array.shape[-1], y_array.shape[-1]))

    u = K.get_variable('u', shape=(x_array.shape[-1], 1))
    v = K.get_variable('v', shape=(y_array.shape[-1], 1))
    
    #x1tx2 = tf.matmul(tf.transpose(x1), x2)
    ux1tx2 = K.matmul(tf.transpose(u), x1tx2)
    ux1tx2v = K.matmul(ux1tx2, v)
    cca_loss = -ux1tx2v
    
    ## clipping
    clipped_u = K.clip_by_norm(u, clip_norm=1.0, axes=0)
    clip_u = K.assign(u, clipped_u, name='ortho')
    K.add_to_collection('normalize_ops', clip_u)
    clipped_v = K.clip_by_norm(v, clip_norm=1.0, axes=0)
    clip_v = K.assign(v, clipped_v, name='ortho')
    K.add_to_collection('normalize_ops', clip_v)
    
    ## l1 penalty
    l1_u = K.reduce_sum(tf.abs(u))
    l1_v = K.reduce_sum(tf.abs(v))
    
    ## total loss
    total_loss = cca_loss + l1_u * L1_PENALTY + l1_v * L1_PENALTY
    
    
    
    
   # Return Sum of the diagonal - Sum of upper and lower triangles   
    trace       = TN.trace(y_true[0,:,:])
    triu_sum    = K.sum(K.abs(TB.triu(y_true[0,:,:],k=1)))
    tril_sum    = K.sum(K.abs(TB.tril(y_true[0,:,:],k=-1)))
    return trace - tril_sum - triu_sum
   
deflation=True 
ncomponents=10 
def scca_loss0(x_proj,y_proj):
#   
    covar_mat = tf.abs(tf.matmul(tf.transpose(x_proj), y_proj))
#    covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)

    if deflation:
        # diagonal sum
        diag_loss = tf.reduce_sum(tf.diag_part(covar_mat))

        # upper triangle sum
        ux,uy = np.triu_indices(ncomponents, k=1)
        u_idxs = [[aa,bb] for aa,bb in zip(ux,uy)]
        upper_loss = tf.reduce_sum(tf.gather_nd(covar_mat, u_idxs))

        # lower triangle sum
        lx, ly = np.tril_indices(ncomponents, k=-1)
        l_idxs = [[aa,bb] for aa,bb in zip(lx,ly)]
        lower_loss = tf.reduce_sum(tf.gather_nd(covar_mat, l_idxs))

        total_loss = -1.*diag_loss + lower_loss + upper_loss
    else:
        total_loss = -tf.reduce_sum(covar_mat)

#    if sparsity[0] > 0. or sparsity[1] > 0.:
#        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#        total_loss = total_loss + tf.add_n(reg_losses)

    return total_loss
#######################################################################################3
def scca_loss3(x_proj,y_proj,u,v):
#   
    x1tx2 = tf.abs(tf.matmul(tf.transpose(x_proj), y_proj))
    ux1tx2 = tf.matmul(tf.transpose(u), x1tx2)
    ux1tx2v =tf.matmul(ux1tx2, v)
    cca_loss2 = -ux1tx2v
    
    
    clipped_u = tf.clip_by_norm(u, clip_norm=1.0, axes=0)
    clip_u = tf.assign(u, clipped_u, name='ortho')
    tf.add_to_collection('normalize_ops', clip_u)
    clipped_v = tf.clip_by_norm(v, clip_norm=1.0, axes=0)
    clip_v = tf.assign(v, clipped_v, name='ortho')
    tf.add_to_collection('normalize_ops', clip_v)
    
    ## l1 penalty
    l1_u =tf.reduce_sum(tf.abs(u))
    l1_v =tf.reduce_sum(tf.abs(v))
    
    ## total loss
    total_loss = cca_loss2# + l1_u * L1_PENALTY + l1_v * L1_PENALTY
    return total_loss
#######################################################################################3
def scca_loss(x_proj,y_proj,u,v):
#   
    x1tx2 = tf.abs(tf.matmul(tf.transpose(x_proj), y_proj))
    ux1tx2 = tf.matmul(tf.transpose(u), x1tx2)
    ux1tx2v =tf.matmul(ux1tx2, v)
#    cca_loss2 = -ux1tx2v
    
    xu_yv=tf.matmul(x_proj, u)-tf.matmul(y_proj, v)
    omega=tf.reduce_sum(tf.norm(xu_yv,ord=2))
    cca_loss2 = -ux1tx2v/(2*omega*omega)
    
    clipped_u = tf.clip_by_norm(u, clip_norm=1.0, axes=0)
    clip_u = tf.assign(u, clipped_u, name='ortho')
    tf.add_to_collection('normalize_ops', clip_u)
    clipped_v = tf.clip_by_norm(v, clip_norm=1.0, axes=0)
    clip_v = tf.assign(v, clipped_v, name='ortho')
    tf.add_to_collection('normalize_ops', clip_v)
    
    ## l1 penalty
    l1_u =tf.reduce_sum(tf.abs(u))
    l1_v =tf.reduce_sum(tf.abs(v))
    
    ## total loss
    total_loss = cca_loss2# + l1_u * L1_PENALTY + l1_v * L1_PENALTY
    return total_loss

def scca_loss_u2(x_proj,y_proj,u,v):
#   
#    x1tx2 = tf.abs(tf.matmul(tf.transpose(x_proj), y_proj))
#    ux1tx2 = tf.matmul(tf.transpose(u), x1tx2)
#    ux1tx2v =tf.matmul(ux1tx2, v)
#    cca_loss2 = -ux1tx2v
#    
#    clipped_u = tf.clip_by_norm(u, clip_norm=1.0, axes=0)
#    clip_u = tf.assign(u, clipped_u, name='ortho')
#    tf.add_to_collection('normalize_ops', clip_u)
#    clipped_v = tf.clip_by_norm(v, clip_norm=1.0, axes=0)
#    clip_v = tf.assign(v, clipped_v, name='ortho')
#    tf.add_to_collection('normalize_ops', clip_v)
    
    
   # u2=X1'*(x2*v);
    x2v = tf.matmul(y_proj, v)
    u2=tf.matmul(tf.transpose(x_proj), x2v)
    ## l1 penalty
    l1_u =tf.reduce_sum(tf.abs(u))
#    l1_v =tf.reduce_sum(tf.abs(v))
    
    l2u=tf.reduce_sum(tf.norm(u2,ord=2))
    ## total loss
#    total_loss = cca_loss2# + l1_u * L1_PENALTY + l1_v * L1_PENALTY
    return l1_u#+l2u

def scca_loss_v2(x_proj,y_proj,u,v):

    
     # v2=x2'*(x1*u);% 用上面求出的u, 来求v
     # v2=x2'*(x1*u);% 用上面求出的u, 来求v
    x1u = tf.matmul(x_proj, u)
    v2=tf.matmul(tf.transpose(x_proj), x1u)
    ## l1 penalty
   # l1_u =tf.reduce_sum(tf.abs(u))
    l1_v =tf.reduce_sum(tf.abs(v))
    l2v=tf.reduce_sum(tf.norm(v2,ord=2))
    ## total loss
  #  total_loss = cca_loss2# + l1_u * L1_PENALTY + l1_v * L1_PENALTY
    return l1_v#+l2v
    
#    if not deflation:
#        # diagonal sum
#        diag_loss = tf.reduce_sum(tf.diag_part(covar_mat))
#
#        # upper triangle sum
#        ux,uy = np.triu_indices(ncomponents, k=1)
#        u_idxs = [[aa,bb] for aa,bb in zip(ux,uy)]
#        upper_loss = tf.reduce_sum(tf.gather_nd(covar_mat, u_idxs))
#
#        # lower triangle sum
#        lx, ly = np.tril_indices(ncomponents, k=-1)
#        l_idxs = [[aa,bb] for aa,bb in zip(lx,ly)]
#        lower_loss = tf.reduce_sum(tf.gather_nd(covar_mat, l_idxs))
#
#        total_loss = -1.*diag_loss + lower_loss + upper_loss
#    else:
#        total_loss = -tf.reduce_sum(covar_mat)
#
#    if sparsity[0] > 0. or sparsity[1] > 0.:
#        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#        total_loss = total_loss + tf.add_n(reg_losses)
#
#    return total_loss

#    covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
#    diag_sum = tf.reduce_sum(tf.abs(tf.diag_part(covar_mat)))
#    cca_score = tf.multiply(-1., diag_sum)
#    #inter_sum = tf.reduce_sum(tf.abs(tf.matrix_band_part(covar_mat, 0, -1)))
#    #cca_score = tf.multiply(-1., diag_sum - inter_sum) 
#    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#    total_loss = cca_score + tf.add_n(reg_losses)
#    return cca_score  #total_loss  
    
def scca_loss_zz(x_proj,y_proj,u,v,z):
    xu=tf.matmul(x_proj, u)
    xu_z=xu-z
    l3_1=tf.reduce_sum(tf.abs( xu_z))
    l3_2=tf.reduce_sum(tf.norm( xu_z,ord=2))/2
    
    
    return l3_1+l3_2

def scca_loss_hh(x_proj,y_proj,u,v,h):
    yv=tf.matmul(y_proj, v)
    yv_h=yv-h
    l4_1=tf.reduce_sum(tf.abs( yv_h))
    l4_2=tf.reduce_sum(tf.norm( yv_h,ord=2))/2
    return l4_1+l4_2


def L1_loss(x_proj,y_proj):
   

    covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
    diag_sum = tf.reduce_sum(tf.abs(tf.diag_part(covar_mat)))
    cca_score = tf.multiply(-1., diag_sum)
    #inter_sum = tf.reduce_sum(tf.abs(tf.matrix_band_part(covar_mat, 0, -1)))
    #cca_score = tf.multiply(-1., diag_sum - inter_sum) 
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    losses = tf.add_n(reg_losses)
    return losses  #total_loss     
    
#########################################################################################################################
class CustomMultiLossLayer2(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer2, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer2, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
#        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        log_var=self.log_vars
#        for y_true, y_pred, log_var ,ii in zip(ys_true, ys_pred, self.log_vars,self.nb_outputs):
        for i in range(self.nb_outputs):    
            precision = (K.exp(-log_var[i]))#**0.5
#            precision= tf.clip_by_value(precision, 0., 1.)
            if i==0:
#                lossA=neg_log_pl(ys_true[i], ys_pred[i])
                
                lossA=scca_loss(ys_true, ys_pred)
                loss += K.sum( lossA + log_var[i], -1)
#                print("Neg_loss:",lossA.item())
            if i==1: 
#                 lossA = py_func(ordinal_loss,
#                     [ys_true[i-1], ys_pred[i]], [tf.float32],
#                     name = "ordinal_loss",
#                     grad_func = ordinal_loss_grad )[0]
#                 lossA.set_shape((ys_true[i].shape[0],))
#                lossA=LOSS_L2(ys_true[i], ys_pred[i])
                lossA=L1_loss(ys_true, ys_pred)
                loss += K.sum(precision * lossA+ log_var[i], -1)
#                lossA=neg_log_pl(ys_true[i-1], ys_pred[i])
#                print("Ordinal_loss:",lossA.item())
#                lossA=tf.py_function(func=ordinal_loss, inp=[ys_true[i], ys_pred[i]], Tout=tf.float32) 
#                lossA=ordinal_loss(Y_trueM, model_aux(tf.to_float(x_train0M)))
#                lossA=neg_log_pl_1(ys_true[i-1], ys_pred[i])
#            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
#            loss += K.sum(precision * lossA, -1)
#            loss += K.sum(precision * lossA + log_var[i], -1)
        
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs-1][-1]
        ys_pred = inputs[self.nb_outputs-1:][-1]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)
    #######################################################################
   
#############################################################################################################  
#########################################################################################################################
class CustomMultiLossLayer3(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer3, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # initialise log_vars
        self.log_vars = []
#        self.u = []
#        self.log_vars = []
#        u = K.get_variable('u', shape=(ys_true.shape[-1], 1))
#        v = K.get_variable('v', shape=(ys_pred.shape[-1], 1))
        
        self.uu=self.add_weight(name='u', shape=(input_shape[0][-1], 1),  initializer='random_normal', trainable=True)
        self.vv=self.add_weight(name='v', shape=(input_shape[1][-1], 1),  initializer='random_normal', trainable=True)
        self.zz=self.add_weight(name='zz', shape=(int(k_n.get_value()), 1),  initializer='random_normal', trainable=True)
        self.hh=self.add_weight(name='hh', shape=(int(k_n.get_value()), 1),  initializer='random_normal', trainable=True)
        for i in range(self.nb_outputs+3):
           
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer3, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
#        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        log_var=self.log_vars
        uu=self.uu
        vv=self.vv
        zz=self.zz
        hh=self.hh
#        for y_true, y_pred, log_var ,ii in zip(ys_true, ys_pred, self.log_vars,self.nb_outputs):
        for i in range(self.nb_outputs+3):    
            precision = (K.exp(-log_var[i]))#**0.5
#            precision = log_var[i]#**0.5
            #precision= tf.clip_by_value(precision, 0., 1.)
            if i==0:
#                lossA=neg_log_pl(ys_true[i], ys_pred[i])
                
                lossA=scca_loss(ys_true, ys_pred,uu,vv)
#                lossA=scca_loss(ys_true, ys_pred)
                loss += K.sum( lossA + log_var[i], -1)
#                print("Neg_loss:",lossA.item())
            if i==1: 
#                 lossA = py_func(ordinal_loss,
#                     [ys_true[i-1], ys_pred[i]], [tf.float32],
#                     name = "ordinal_loss",
#                     grad_func = ordinal_loss_grad )[0]
#                 lossA.set_shape((ys_true[i].shape[0],))
#                lossA=LOSS_L2(ys_true[i], ys_pred[i])
                lossA=scca_loss_u2(ys_true, ys_pred,uu,vv)
                loss += K.sum(precision * lossA+ log_var[i], -1)
                
            if i==2: 
#                 lossA = py_func(ordinal_loss,
#                     [ys_true[i-1], ys_pred[i]], [tf.float32],
#                     name = "ordinal_loss",
#                     grad_func = ordinal_loss_grad )[0]
#                 lossA.set_shape((ys_true[i].shape[0],))
#                lossA=LOSS_L2(ys_true[i], ys_pred[i])
                lossA=scca_loss_v2(ys_true, ys_pred,uu,vv)
                loss += K.sum(precision * lossA+ log_var[i], -1)
                
            if i==3: 
                lossA=scca_loss_zz(ys_true, ys_pred,uu,vv,zz)
                loss += K.sum(precision * lossA+ log_var[i], -1)          
            if i==4: 
                lossA=scca_loss_hh(ys_true, ys_pred,uu,vv,hh)
                loss += K.sum(precision * lossA+ log_var[i], -1)      
                
#                lossA=neg_log_pl(ys_true[i-1], ys_pred[i])
#                print("Ordinal_loss:",lossA.item())
#                lossA=tf.py_function(func=ordinal_loss, inp=[ys_true[i], ys_pred[i]], Tout=tf.float32) 
#                lossA=ordinal_loss(Y_trueM, model_aux(tf.to_float(x_train0M)))
#                lossA=neg_log_pl_1(ys_true[i-1], ys_pred[i])
#            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
#            loss += K.sum(precision * lossA, -1)
#            loss += K.sum(precision * lossA + log_var[i], -1)
        
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs-1][-1]
        ys_pred = inputs[self.nb_outputs-1:][-1]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)
    #######################################################################
   
#############################################################################################################       
#############################################################################################################
def scheduler2(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0:# and epoch != 0:
        lr2=[0.0005,0.0004,0.0004,0.0002,0.0001,0.00005,0.00004,0.00002,0.00001,0.000005,0.000002]
        #lr2=[0.0005,0.0001,0.00005,0.00001,0.000005,0.000005,0.000005,0.000005,0.000005,0.000005,0.000005]
        learning_rate=lr2[int(epoch/100)]
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, (lr/lr )* learning_rate)
        print("lr changed to {}".format((lr/lr )* learning_rate))
    return K.get_value(model.optimizer.lr)

####################################################################################################################       
    
    
    
    
    
    
#############################################################################################################
def scheduler22(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0:# and epoch != 0:
        lr2=[0.005,0.004,0.004,0.002,0.001,0.0005,0.0004,0.0002,0.0001,0.00005,0.00002]
        learning_rate=lr2[int(epoch/100)]
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, (lr/lr )* learning_rate)
        print("lr changed to {}".format((lr/lr )* learning_rate))
    return K.get_value(model.optimizer.lr)

####################################################################################################################    
sparsity=(1e-5,1e-5)
seed = 63
np.random.seed(seed)
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
ypred=[]
ypred_train=[]
xtest_original=[]
status_new=[]
time_new=[]
index2=[]
iFold = 0
EPOCH =1000
W_uu=[]
W_vv=[]
for train_index, val_index in kf.split(x):
    iFold = iFold+1
#    train_x, test_x, train_y, test_y,= X[train_index], X[val_index], y[train_index], y[val_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
    x_train, x_test, y_train, y_test, ytime_train, ytime_test, ystatus_train, ystatus_test,data_methylation_train,data_methylation_test,data_mRNA_train,data_mRNA_test, ystatus2_train, ystatus2_test =\
        dataX[train_index], dataX[val_index], y[train_index], y[val_index], ytime[train_index], ytime[val_index], ystatus[train_index],ystatus[val_index],\
        data_methylation[train_index],data_methylation[val_index],data_mRNA[train_index],data_mRNA[val_index],ystatus2[train_index],ystatus2[val_index]
    
#    input_dim =x_train.shape[2]
#    output_dimM = y_train.shape[1]
#    output_dimA = 1
#    n1 = y_train.shape[0]
    input_dim =x_train.shape[1]
    input_dim_mRNA =data_mRNA.shape[1]
#    ytrue_dim=y_train.shape[1]
    input_dim_methylation =data_methylation.shape[1]
    
    output_dimM = y_train.shape[1]
    output_dimA = 1
    n1 = y_train.shape[0]
    
#    k_n = theano.shared(np.asarray(n,dtype=theano.config.floatX),borrow=True)
    k_n = theano.shared(n1,borrow=True)
    k_ytime_train = theano.shared(ytime_train,borrow=True)
    k_ystatus_train = theano.shared(ystatus_train,borrow=True)
    N = theano.shared(n1,borrow=True)
    R_matrix = np.zeros([n1, n1], dtype=int)
    R_matrix =theano.shared(R_matrix,borrow=True)
##############################################3    
    
    Y_hazard0=y_train[:,0]
    Y_survival=y_train[:,1]

#    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_train.shape[0],1])
#    Y_hazard=Y_hazard1M[-1,:,-1]
    
    t0, H0 = unique_set(Y_hazard0) # t:unique time. H original index.
    
    actual_event_index = np.nonzero(Y_survival)[0]
    H0 = [list(set(h) & set(actual_event_index)) for h in H0]
    ordinal_n = np.asarray([len(h) for h in H0])
    Hj=sum(H0[0:],[])
    
    k_ordinal_H = theano.shared(np.asarray(Hj),borrow=True)
    k_ordinal_t = theano.shared(t0,borrow=True)
    k_ordinal_n = theano.shared(ordinal_n,borrow=True)
 #########################################################################################################################   
#    例: 记录损失历史
#class LossHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.losses = []
#
#    def on_batch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss'))
#
#model = Sequential()
#model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#
#history = LossHistory()
#model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])
#
#print(history.losses)
## 输出
#'''
#[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
 #########################################################################################################################   
 #########################################################################################################################   
 #########################################################################################################################   
 #########################################################################################################################   
#    seed2 = 63
#    np.random.seed(seed2)
#    kf2 = KFold(n_splits=10, shuffle=True, random_state=seed2)
#    W_uu=[]
#    W_vv=[]
#    iFold2=0
#    EPOCH2 =500
##    data_methylation_train=dataX1[:,0:17]
##    data_mRNA=dataX1[:,17:n0]
#    for train_index2, val_index2 in kf2.split(x_train):
#        iFold2 = iFold2+1
##    train_x, test_x, train_y, test_y,= X[train_index], X[val_index], y[train_index], y[val_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
#        x_train2, x_test2, y_train2, y_test2, ytime_train2, ytime_test2, ystatus_train2, ystatus_test2,data_methylation_train2,data_methylation_test2,data_mRNA_train2,data_mRNA_test2, ystatus2_train2, ystatus2_test2 =\
#            x_train[train_index2], x_train[val_index2], y_train[train_index2], y_train[val_index2], ytime_train[train_index2], ytime_train[val_index2], ystatus_train[train_index2],ystatus_train[val_index2],\
#            data_methylation_train[train_index2],data_methylation_train[val_index2],data_mRNA_train[train_index2],data_mRNA_train[val_index2],ystatus2_train[train_index2],ystatus2_train[val_index2]
#    
#    #    input_dim =x_train.shape[2]
#    #    output_dimM = y_train.shape[1]
#    #    output_dimA = 1
#    #    n1 = y_train.shape[0]
#        input_dim2 =x_train2.shape[1]
#        input_dim_mRNA2 =data_mRNA_train.shape[1]
#    #    ytrue_dim=y_train.shape[1]
#        input_dim_methylation2 =data_methylation_train.shape[1]
    
#    #######################################################################################
    class varHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.lambdas0 = []
            self.lambdas1 = []
            self.lambdas2 = []
            self.lambdas3 = []
            self.u = []
            self.v = []
        def on_epoch_end(self, batch, logs={}):
            self.lambdas0.append( K.get_value(self.model.layers[-1].log_vars[1]))
            self.lambdas1.append( K.get_value(self.model.layers[-1].log_vars[2]))
            self.lambdas2.append( K.get_value(self.model.layers[-1].log_vars[3]))
            self.lambdas3.append( K.get_value(self.model.layers[-1].log_vars[4]))
            self.uu= K.get_value(self.model.layers[-1].uu)
            self.vv= K.get_value(self.model.layers[-1].vv)
            self.zz= K.get_value(self.model.layers[-1].zz)
            self.hh= K.get_value(self.model.layers[-1].hh)
     #############################################################################################################################    
    ##    input_dim0 =theano.shared(input_dim,borrow=True)
    ## Build model structure
    #    # gene Only
    #    gene_input = Input(name='gene_input', shape=(1,input_dim_mRNA))
    ##    out1=Bidirectional(LSTM(55,activation='linear',return_sequences=True,kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), merge_mode='concat')(title_input)
    #    out_gene=Bidirectional(LSTM(55,activation='tanh', return_sequences=False,\
    #        kernel_initializer=glorot_uniform(),kernel_regularizer=l2(0.0005),activity_regularizer=l2(0.001)),  merge_mode='concat')(gene_input)
    #    main_output= Dense(10,activation='linear',name='main_output')(out_gene)
    ##    Bidirectional(LSTM(100,return_sequences=True), merge_mode='concat')(gene_input)
    ##    out2=TimeDistributed(Dense(50, activation='tanh'))(out1)
    ##    out_gene=Bidirectional(LSTM(20))(out2)
    #    
    ##    auxiliary_output = Dense(1, activation='linear', name='aux_output')(out_gene)  
    ##    GRU(100,  activation='linear', return_sequences=True)(title_input)
    #     # clinic Only
    #    clinic_input = Input(name='clinic_input', shape=(1,input_dim_methylation))
    #    
    #    out_clinic=Bidirectional(LSTM(55,activation='tanh', return_sequences=False,\
    #        kernel_initializer=glorot_uniform(),kernel_regularizer=l2(0.0005),activity_regularizer=l2(0.001)),merge_mode='concat')(clinic_input )
    ##    Bidirectional(LSTM(100,activation='tanh',return_sequences=False), merge_mode='concat')(clinic_input )
    #    auxiliary_output = Dense(10,activation='linear', name='aux_output')(out_clinic) #sigmoid
    ##    out_clinic=Bidirectional(LSTM(100,return_sequences=False), merge_mode='concat')(clinic_input )
    ##    
    ##    out21=TimeDistributed(Dense(50, activation='tanh'))( out_gene)
    ##    out22=Bidirectional(LSTM(20,return_sequences=False))(out21)
    ###    model.add(TimeDistributed(Dense(50, activation='tanh')))
    ###    model.add(Bidirectional(LSTM(20)))
    ##    # combined with GRU output
    ###    input_ = Input(shape=(12,8))
    ##   
    ###    com = Concatenate(axis=1)([out_gene, out_clinic])
    #    
    ###############################################################################################################################################    
     ####################
    # X projection model
    ####################
#    modelX = Sequential()
    gene_input = Input(name='gene_input', shape=[input_dim_mRNA])
    out_gene=Dense(10, bias=False, 
                 W_constraint=UnitNormWithNonnneg(False),
                 W_regularizer=l1(sparsity[0]),
                 name='main_output')(gene_input)

    ####################
    # Y projection model
    ####################
#    modelY = Sequential()
    
    clinic_input = Input(name='clinic_input', shape=[input_dim_methylation])
#    out_clinic=TimeDistributed(Dense(10, bias=False,
#        W_constraint=UnitNormWithNonnneg(False),
#        W_regularizer=l1(sparsity[1])))( clinic_input)
    out_clinic=Dense(10, bias=False,
        W_constraint=UnitNormWithNonnneg(False),
        W_regularizer=l1(sparsity[1]), name='aux_output')( clinic_input) #activation='linear',
    
    
#########################################################################################################################################################
##    shared_layer = concatenate([main_output, auxiliary_output], name='shared_layer')
#    shared_layer = concatenate([out_gene, out_clinic], name='shared_layer')
#    
#    cca_layer = CCA(1, name='cca_layer')(shared_layer)
#    model = Model(inputs=[gene_input, clinic_input], outputs=cca_layer)
#    model.summary()
#    u_comp  = model.layers[0].get_weights()[0]
#    v_comp  = model.layers[0].get_weights()[0]
#    
#    
#    model.compile(optimizer='rmsprop', loss=constant_loss, metrics=[mean_pred])
#    model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)),
#              batch_size=batch_size, epochs=epoch_num, shuffle=True, verbose=1,
#              validation_data=([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1))))
#####################################################################################################################################################     
#    u_comp  = model.layers[0].get_weights()[0]
#    v_comp  = model.layers[0].get_weights()[0]
#    
    
    y1_true = Input(shape=(2,), name='y1_true')
#    y1_true = Input(shape=(output_dimM,), name='y1_true')
    y2_true = Input(shape=(2,), name='y2_true')
#    out = CustomMultiLossLayer2(nb_outputs=2)([out_gene, out_clinic])
    out = CustomMultiLossLayer3(nb_outputs=2)([gene_input, clinic_input])
    model =Model([gene_input,clinic_input], out)
    model.summary()
    model.compile(optimizer='adam', loss=None)
    
    
 #####################################################################################################################################################     
   
#    out222=Dense(20, activation='linear')(out22)
#
#    
##    GRU(50, activation='tanh', return_sequences=False)(out1)
#    out3=Dropout(0.1)(out22)
#    main_output= Dense(1,activation='linear',name='main_output')(out_gene)
#    main_output1=main_output[:,-1,:]
#    auxiliary_output1=auxiliary_output[:,-1,:]
#    y1_true = Input(shape=(output_dimM,), name='y1_true')
#    y2_true = Input(shape=(output_dimA,), name='y2_true')
#    model = Model(inputs=[gene_input,clinic_input],outputs=[main_output, auxiliary_output])
#    y1_true = Input(shape=(2,), name='y1_true')
##    y1_true = Input(shape=(output_dimM,), name='y1_true')
#    y2_true = Input(shape=(2,), name='y2_true')
#    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, main_output, auxiliary_output])
#    model =Model([gene_input,clinic_input,y1_true, y2_true], out)
#    model.summary()
#    model.compile(optimizer='adam', loss=None)
    
     #取某一层的输出为输出新建为model，采用函数模型
#    dense1_layer_model = Model(inputs=model.input, outputs=[model.get_layer('main_output').output,model.get_layer('aux_output').output])
#    dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('main_output').output)
#    dense1_layer_model.summary()
    
    
    reduce_lr = LearningRateScheduler(scheduler2)
    Lambdas = varHistory() 
    hist = model.fit([data_mRNA_train,data_methylation_train], batch_size = n1, epochs =EPOCH,callbacks=[reduce_lr,Lambdas])
    
    
    lambda0=np.asarray( Lambdas.lambdas0)
    lambda1=np.asarray( Lambdas.lambdas1)
    lambda2=np.asarray( Lambdas.lambdas2)
    lambda3=np.asarray( Lambdas.lambdas3)
    
    uu=np.asarray( Lambdas.uu)
    vv=np.asarray( Lambdas.vv)
    zz=np.asarray( Lambdas.zz)
    hh=np.asarray( Lambdas.hh)
    
    std0=(np.exp(-lambda0))#**0.5
    std1=(np.exp(-lambda1))#**0.5
    std2=(np.exp(-lambda2))#**0.5
    std3=(np.exp(-lambda3))#**0.5
    print("iFold=",iFold)
    plt.plot(range(EPOCH), lambda0,color='red')
#    plt.show() 
    plt.plot(range(EPOCH), lambda1,color='purple')
    plt.plot(range(EPOCH), lambda2,color='blue')
    plt.plot(range(EPOCH), lambda3,color='black')
    
    plt.title('MT_DSCCA Lambda Curves')
    plt.ylabel('Train-lambda Value')
    plt.xlabel('epoch')
    plt.legend(['lambda1','lambda2','lambda3','lambda4'], loc='center right')
    plt.savefig('MT_DSCCA Lambda Curves1234.jpg', dpi=300) #指定分辨率保存
    plt.show()
    
    plt.plot(range(EPOCH), std0,color='red')
#    plt.show() 
    plt.plot(range(EPOCH), std1,color='purple')
    plt.plot(range(EPOCH), std2,color='blue')
    plt.plot(range(EPOCH), std3,color='black')
    plt.title('MT_DSCCA Weight Curves')
    plt.ylabel('Train-weight Value')
    plt.xlabel('epoch')
    plt.legend(['lambda1','lambda2','lambda3','lambda4'], loc='upper right')
    plt.savefig('MT_DSCCA lambda1-lambda4 Curves term315.jpg', dpi=300) #指定分辨率保存
    plt.show()
    
    plt.plot(range(EPOCH), hist.history['loss'])
    plt.title('Training Loss Curves MT_DSCCA')
    plt.ylabel('train-loss Value')
    plt.xlabel('epoch')
#    plt.legend(['${lambda0}$','${lambda1}$'], loc='upper right')
    plt.savefig('Loss Curves without penalty term315.jpg', dpi=300) #指定分辨率保存
    plt.show()
    
    
    np.savetxt("BRCA_methylation_mRNA_MT_DSCCA_20210902.csv",hist.history['loss'], delimiter=",")
    
#    import pylab
#    pylab.plot(hist.history['loss'])
#    print([np.exp(-K.get_value(log_var[0]))**0.5 for log_var in model.layers[-1].log_vars])
    print('lambda1 for Main Loss=',lambda0[999])
    print('lambda2 for Auxiliary Loss=',lambda1[999])
    ###########
#    print('weight1 for Main Loss=',std0[999])
    print('weight1 for Auxiliary1 Loss=',std0[999])
    print('weight2 for Auxiliary Loss=',std1[999])
    print('weight3 for Auxiliary Loss=',std2[999])
    print('weight4 for Auxiliary Loss=',std3[999])
    
    print('uu for data_gene=',uu)
    print('vv for methylation data=',vv)
    ###############
    idx1=(np.abs(uu)>0.01)[:,-1]
    print('idx1 for data_gene=',idx1)
    W_u=np.arange(input_dim_mRNA)
    W1=W_u[idx1]
    W_uu.extend(W_u[idx1])
    
    idx2=(np.abs(vv)>0.01)[:,-1]
    print('idx2 for data_gene=',idx2)
    W_v=np.arange(input_dim_methylation)
    W2=W_v[idx2]
    W_vv.extend(W_v[idx2])
    print("iFold2=",iFold)
   ##########################################################################################
##########################################################################################
 ##########################################################################################
Au= pd.value_counts(W_uu)
Av=pd.value_counts(W_vv)

Gr_index=Au>=8
Gm_index=Av>=9

idx_r=np.array((Gr_index[Gr_index]).index.tolist())
idx_m=np.array((Gm_index[Gm_index]).index.tolist())

mRNA_names3=[col_name_mRNA[idx_r2] for idx_r2 in idx_r]
meth_names3=[col_name_meth[idx_m2] for idx_m2 in idx_m]


data_mRNA_train3=data_mRNA[:,Gr_index]
data_mRNA_train4=pd.DataFrame(data_mRNA_train3,columns=mRNA_names3)

data_methylation_train3=data_methylation[:,Gm_index]
data_methylation_train4=pd.DataFrame(data_methylation_train3,columns=meth_names3)

data_mm=np.concatenate([data_mRNA_train3,data_methylation_train3],1)
data_mm2=np.concatenate([y,data_mm],1)

data_mm_name4=pd.concat([data_mRNA_train4,data_methylation_train4],1)

y_data_name=pd.DataFrame(y,columns=y_name)

data_y_mm_name55=pd.concat([kidtx["X.1"],y_data_name,data_mm_name4],1) ############ add colomn name
#data_mRNA_test3=data_mRNA_test[:,Gr_index]
#data_methylation_test3=data_methylation_test[:,Gm_index]
#data_mm_test=np.concatenate([data_mRNA_test3,data_methylation_test3],1)

[ m3,n3] = data_mm.shape
#[ m4,n4] = data_mm_test.shape
   

 
#data_mm3 = data_mm.reshape(m3,1,n3)
#data_mm_test3=data_mm_test.reshape(m4,1,n4)

np.savetxt("BRCA_data_mm_dscca_p001_8-9_414.csv",data_mm2, delimiter=",")

#pd.dataFrame.to_save()
data_y_mm_name55.to_csv("BRCA_data_mm_name_dscca_p001_8-9_414.csv")


aaaaaaaaaaaaaaaaaa=0
    ##########################################################################################
#    #'''
#   class varHistory(keras.callbacks.Callback):
#        def on_train_begin(self, logs={}):
#            self.lambdas0 = []
#            self.lambdas1 = []
#        def on_epoch_end(self, batch, logs={}):
#            self.lambdas0.append( K.get_value(self.model.layers[-1].log_vars[0]))
#            self.lambdas1.append( K.get_value(self.model.layers[-1].log_vars[1]))
# #############################################################################################################################    
##    input_dim0 =theano.shared(input_dim,borrow=True)
## Build model structure
#    # gene Only
#    gene_input = Input(name='gene_input', shape=(1,n3))
##    out1=Bidirectional(LSTM(55,activation='linear',return_sequences=True,kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), merge_mode='concat')(title_input)
##    out_gene=Bidirectional(LSTM(55,return_sequences=False), merge_mode='concat')(gene_input)
#    
#    out_gene=Bidirectional(LSTM(55,activation='tanh', return_sequences=False,\
#        kernel_initializer=glorot_uniform(),kernel_regularizer=l2(0.0005),activity_regularizer=l2(0.001)),  merge_mode='concat')(gene_input)
#    main_output= Dense(1,activation='linear',name='main_output')(out_gene)
##    model =Model([gene_input,clinic_input,y1_true, y2_true], out)
#    model = Model(inputs=[gene_input],outputs=[main_output])
#    model.summary()
#    
#    model.compile(optimizer='adam', loss=neg_log_pl)
#    
#     #取某一层的输出为输出新建为model，采用函数模型
##    dense1_layer_model = Model(inputs=model.input, outputs=[model.get_layer('main_output').output,model.get_layer('aux_output').output])
##    dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('main_output').output)
##    dense1_layer_model.summary()
#    
#    
#    reduce_lr = LearningRateScheduler(scheduler)
#    Lambdas = varHistory() 
#    hist = model.fit(data_mm3, y_train, batch_size = n1, epochs =EPOCH,callbacks=[reduce_lr])
#    
#   
##    lambda0=np.asarray( Lambdas.lambdas0)
##    lambda1=np.asarray( Lambdas.lambdas1)
##    
##    std0=(np.exp(-lambda0))**0.5
##    std1=(np.exp(-lambda1))**0.5
##    
##    plt.plot(range(EPOCH), std0)
###    plt.show() 
##    plt.plot(range(EPOCH), std1)
##    plt.title('Lambda Curves without penalty term')
##    plt.ylabel('Train-lambda Value')
##    plt.xlabel('epoch')
##    plt.legend(['lambda1 for Main Loss','lambda2 for Auxiliary Loss'], loc='center right')
##    plt.savefig('Lambda Curves without penalty term.jpg', dpi=300) #指定分辨率保存
##    plt.show()
#    
#    plt.plot(range(EPOCH), hist.history['loss'])
#    plt.title('Training Loss Curves only main loss')
#    plt.ylabel('train-loss Value')
#    plt.xlabel('epoch')
##    plt.legend(['${lambda0}$','${lambda1}$'], loc='upper right')
#    plt.savefig('Loss Curves only main loss316.jpg', dpi=300) #指定分辨率保存
#    plt.show()
#    
#    
#    np.savetxt("hist.history['loss1000_ml_ordCOX20201220']_without_penalty_term_only_main_loss406.csv",hist.history['loss'], delimiter=",")
#    
##    import pylab
##    pylab.plot(hist.history['loss'])
##    print([np.exp(-K.get_value(log_var[0]))**0.5 for log_var in model.layers[-1].log_vars])
#    
##    predicted_main, predicted_aux = dense1_layer_model.predict([x_test,x_test,y_test, ystatus2_test],verbose=1)
#    prediction = model.predict(data_mm_test3)
##    prediction =predicted_main+0*predicted_aux
#    
#    c_index2=c_index3( np.asarray(ytime_test),np.asarray(prediction), np.asarray(ystatus_test))
#    
#    print( c_index2)
#    
############################################################################################################################## 
##############################################################################################################################    
#   
#    ypred.extend(prediction)
##    ypred_train.extend(prediction_train_median)
##    xtest_original.extend(x_test)
#    index2.extend(val_index)
#    status_new.extend(ystatus[val_index])
#    time_new.extend(ytime[val_index])
##    print(ypred.shape)
#    
#    K.clear_session()
#    tf1.reset_default_graph()
#    print(iFold)
#    nowTime = datetime.datetime.now()
#    print("nowTime: ",nowTime)
#np.savetxt("BRCA_prediction20210125_18lstm2222_epoch400_drop01_resnet_without penalty term315.csv", ypred, delimiter=",")
#np.savetxt("BRCA_ytime_test20210125_18lstm2222_epoch400_drop01_resnet_without penalty term315.csv", time_new, delimiter=",")
#np.savetxt("BRCA_ystatus_test20210125_18lstm2222_epoch400_drop01_resnet_without penalty term315.csv", status_new, delimiter=",")
#np.savetxt("BRCA_ypred_train_median20210125_18lstm2222_epoch400_drop01_resnet_without penalty term315.csv", ypred_train, delimiter=",")
#
#df = pd.read_csv("BRCA_prediction20210125_18lstm2222_epoch400_drop01_resnet_without penalty term315.csv",header=None)    
#month=np.asarray(pd.read_csv("BRCA_ytime_test20210125_18lstm2222_epoch400_drop01_resnet_without penalty term315.csv",header=None)) 
#status=np.asarray(pd.read_csv("BRCA_ystatus_test20210125_18lstm2222_epoch400_drop01_resnet_without penalty term315.csv",header=None)) 
#
#
##df=ypred
##month=time_new
##status=status_new
##df1 = pd.read_csv("WAVE_preTest_scores.csv")    
##dataset_init = np.asarray(df1)    # if only 1 column
##dataX, dataY = create_interval_dataset(df, 1)    #这里的输入数据来源是csv文件
#
#risk=np.asarray(df)
#c_indices_lstm = c_index3(month, risk,status)
#print("c_indices:",c_indices_lstm)
##np.savetxt("c_indices_nn827.txt", c_indices_mlp, delimiter=",")
#np.save("c_indices_without penalty term315",c_indices_lstm)
#print("brca_methylation17_mRNA36_lmqcm_lr_backend2233_without_penalty_term3_NO_PPPPplot_new_dataset_20210118_3333_all_0.7222063242237802_no_tf.clip_by_value22.py") 
#print("BRCA:dataX=data_all:",c_indices_lstm)
#data_a=np.load('c_indices_without penalty term315.npy')
#aa=0