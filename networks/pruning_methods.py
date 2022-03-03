# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 23:24:26 2022

@author: xiaohuai le
"""
import tensorflow as tf
import numpy as np
'''
A structured pruning method with the intrinsic sparse structures-based regularization
'''
def get_regular_ISS(rnn, fc, bidirectional = False, *args):
    '''
    get the ISS of the RNN and the following FC
    '''
    if not bidirectional:
        input_weights = rnn.weights[0]
        hidden_weights = rnn.weights[1]
        linear_weights = fc.weights[0]
        # rnn
        t1_inp = tf.square(input_weights)
        t1_hid = tf.square(hidden_weights)
        
        t1_col_sum = tf.reduce_sum(t1_inp, axis = 0) + tf.reduce_sum(t1_hid, axis = 0)
        t1_col_sum1, t1_col_sum2, t1_col_sum3 = tf.split(t1_col_sum, 3)
        t1_row_sum = tf.reduce_sum(t1_hid, axis = 1)
        # linear
        t2 = tf.square(linear_weights)
        t2_row_sum = tf.reduce_sum(t2, axis = 1)
        
        reg_sum = t1_row_sum + \
                  t1_col_sum1 + t1_col_sum2 + t1_col_sum3 + \
                  t2_row_sum+ \
                  tf.constant(1.0e-8)
        reg_sqrt = tf.sqrt(reg_sum)
        reg = tf.reduce_sum(reg_sqrt)
    else:
        forward_input_weights = rnn.weights[0]
        forward_hidden_weights = rnn.weights[1]
        backward_input_weights = rnn.weights[3]
        backward_hidden_weights = rnn.weights[4]
        linear_weights = fc.weights[0]
        # forward
        forward_t1_inp = tf.square(forward_input_weights)
        forward_t1_hid = tf.square(forward_hidden_weights)
        
        forward_t1_col_sum = tf.reduce_sum(forward_t1_inp, axis = 0) + tf.reduce_sum(forward_t1_hid, axis = 0)
        forward_t1_col_sum1, forward_t1_col_sum2, forward_t1_col_sum3 = tf.split(forward_t1_col_sum, 3)
        forward_t1_row_sum = tf.reduce_sum(forward_t1_hid, axis = 1)
        # backward
        backward_t1_inp = tf.square(backward_input_weights)
        backward_t1_hid = tf.square(backward_hidden_weights)
        
        backward_t1_col_sum = tf.reduce_sum(backward_t1_inp, axis = 0) + tf.reduce_sum(backward_t1_hid, axis = 0)
        backward_t1_col_sum1, backward_t1_col_sum2, backward_t1_col_sum3 = tf.split(backward_t1_col_sum, 3)
        backward_t1_row_sum = tf.reduce_sum(backward_t1_hid, axis = 1)
        # linear
        t2 = tf.square(linear_weights)
        t2_row_sum = tf.reduce_sum(t2, axis = 1)
        t2_row_sum_forward,t2_row_sum_backward = tf.split(t2_row_sum,2)
        
        reg_sum = forward_t1_row_sum + \
                  forward_t1_col_sum1 + forward_t1_col_sum2 + forward_t1_col_sum3 + \
                  backward_t1_row_sum + \
                  backward_t1_col_sum1 + backward_t1_col_sum2 + backward_t1_col_sum3 + \
                  t2_row_sum_forward + t2_row_sum_backward + \
                  tf.constant(1.0e-8)
        reg_sqrt = tf.sqrt(reg_sum)
        
        reg = tf.reduce_sum(reg_sqrt)
        
    return reg

def make_mask(rnn,fc,n=1,bidirectional=False):
    
    rnn_weights = rnn.get_weights()
    fc_weights = fc.get_weights()
    h_dim = rnn_weights[1].shape[0]
    if not bidirectional:
        # unidirectional RNN is pruned n dimension per step
        input_weights = rnn_weights[0]**2
        hidden_weights = rnn_weights[1]**2
        dense_weights = fc_weights[0]**2
        
        mag_list = []
        for i in range(h_dim):
            if rnn_weights[5][i,0] == 0:
                mag_list.append(np.inf)
            else:
                t1_row_sum = np.sum(hidden_weights[i,:])
                t1_col_sum1 = np.sum(input_weights[:,i]) + np.sum(hidden_weights[:,i])
                t1_col_sum2 = np.sum(input_weights[:,i+h_dim]) + np.sum(hidden_weights[:,i+h_dim])
                t1_col_sum3 = np.sum(input_weights[:,i+h_dim*2]) + np.sum(hidden_weights[:,i+h_dim*2])
                t2_row_sum = np.sum(dense_weights[i,:])
                reg_sum = t1_row_sum + \
                      t1_col_sum1 + t1_col_sum2 + t1_col_sum3 + \
                      t2_row_sum+ \
                      1.0e-8
                mag_list.append(np.sqrt(reg_sum))
        
        top_n_idx = np.array(mag_list).argsort()[0:n]
        print(top_n_idx)
        for i in top_n_idx:
            rnn_weights[3][0,i] = 0
            rnn_weights[3][0,i+h_dim] = 0
            rnn_weights[3][0,i+h_dim*2] = 0
            rnn_weights[4][0,i] = 0
            rnn_weights[4][0,i+h_dim] = 0
            rnn_weights[4][0,i+h_dim*2] = 0
            
            rnn_weights[5][i,0] = 0
            fc_weights[2][i,0] = 0
        
        rnn.set_weights(rnn_weights)
        fc.set_weights(fc_weights)
    else:
        forward_input_weights = rnn_weights[0]**2
        forward_hidden_weights = rnn_weights[1]**2
        backward_input_weights = rnn_weights[3]**2
        backward_hidden_weights = rnn_weights[4]**2        
        
        dense_weights = fc_weights[0]**2
        #get forward mask
        mag_list = []
        for i in range(h_dim):
            if rnn_weights[8][i,0] == 0:
                mag_list.append(np.inf)
            else:
                t1_row_sum = np.sum(forward_hidden_weights[i,:])
                t1_col_sum1 = np.sum(forward_input_weights[:,i]) + np.sum(forward_hidden_weights[:,i])
                t1_col_sum2 = np.sum(forward_input_weights[:,i+h_dim]) + np.sum(forward_hidden_weights[:,i+h_dim])
                t1_col_sum3 = np.sum(forward_input_weights[:,i+h_dim*2]) + np.sum(forward_hidden_weights[:,i+h_dim*2])
                t2_row_sum = np.sum(dense_weights[i,:])
                reg_sum = t1_row_sum + \
                      t1_col_sum1 + t1_col_sum2 + t1_col_sum3 + \
                      t2_row_sum+ \
                      1.0e-8
                mag_list.append(np.sqrt(reg_sum))
        
        top_n_idx = np.array(mag_list).argsort()[0:n]
        print('foward:',top_n_idx)
        for i in top_n_idx:
            rnn_weights[6][0,i] = 0
            rnn_weights[6][0,i+h_dim] = 0
            rnn_weights[6][0,i+h_dim*2] = 0
            rnn_weights[7][0,i] = 0
            rnn_weights[7][0,i+h_dim] = 0
            rnn_weights[7][0,i+h_dim*2] = 0
            
            rnn_weights[8][i,0] = 0
            fc_weights[2][i,0] = 0
                
        #get backward mask
        mag_list = []
        for i in range(h_dim):
            if rnn_weights[11][i,0] == 0:
                mag_list.append(np.inf)
            else:
                t1_row_sum = np.sum(backward_hidden_weights[i,:])
                t1_col_sum1 = np.sum(backward_input_weights[:,i]) + np.sum(backward_hidden_weights[:,i])
                t1_col_sum2 = np.sum(backward_input_weights[:,i+h_dim]) + np.sum(backward_hidden_weights[:,i+h_dim])
                t1_col_sum3 = np.sum(backward_input_weights[:,i+h_dim*2]) + np.sum(backward_hidden_weights[:,i+h_dim*2])
                t2_row_sum = np.sum(dense_weights[i+h_dim,:])
                reg_sum = t1_row_sum + \
                      t1_col_sum1 + t1_col_sum2 + t1_col_sum3 + \
                      t2_row_sum+ \
                      1.0e-8
                mag_list.append(np.sqrt(reg_sum))
        
        top_n_idx = np.array(mag_list).argsort()[0:n]
        print('backward:',top_n_idx)
        for i in top_n_idx:
            rnn_weights[9][0,i] = 0
            rnn_weights[9][0,i+h_dim] = 0
            rnn_weights[9][0,i+h_dim*2] = 0
            rnn_weights[10][0,i] = 0
            rnn_weights[10][0,i+h_dim] = 0
            rnn_weights[10][0,i+h_dim*2] = 0
            
            rnn_weights[11][i,0] = 0
            fc_weights[2][i+h_dim,0] = 0
        rnn.set_weights(rnn_weights)
        fc.set_weights(fc_weights)