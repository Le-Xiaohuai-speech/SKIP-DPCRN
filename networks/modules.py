# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:53:48 2021

@author: Xiaohuai Le
"""
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Add, RNN
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
import logging

from networks.skip_gru import SkipGRU

#%%
'''
dual path rnn block
'''   
class DprnnBlock(keras.layers.Layer):
    
    def __init__(self, intra_hidden, inter_hidden, batch_size, L, width, channel, causal = False, CUDNN = False, **kwargs):
        super(DprnnBlock, self).__init__(**kwargs)
        '''
        intra_hidden hidden size of the intra-chunk RNN
        inter_hidden hidden size of the inter-chunk RNN
        batch_size 
        L         number of frames, -1 for undefined length
        width     width size output from encoder
        channel   channel size output from encoder
        causal    instant Layer Norm or global Layer Norm
        '''
        self.batch_size = batch_size
        self.causal = causal
        self.L = L
        self.width = width
        self.channel = channel
        
        if CUDNN:
            self.intra_rnn = keras.layers.Bidirectional(keras.layers.CuDNNGRU(units=intra_hidden//2, return_sequences=True))
        else:
            self.intra_rnn = keras.layers.Bidirectional(keras.layers.GRU(units=intra_hidden//2, return_sequences=True,implementation = 1,recurrent_activation = 'sigmoid', unroll = True,reset_after = False))
        
        self.intra_fc = keras.layers.Dense(units = self.channel,)
        
        self.intra_ln = keras.layers.LayerNormalization(center=True, scale=True,epsilon = 1e-8)

        if CUDNN:
            self.inter_rnn = keras.layers.CuDNNGRU(units=inter_hidden, return_sequences=True)
        else:
            self.inter_rnn = keras.layers.GRU(units=inter_hidden, return_sequences=True,implementation = 1,recurrent_activation = 'sigmoid',reset_after = False)

        self.inter_fc = keras.layers.Dense(units = self.channel,) 

        self.inter_ln = keras.layers.LayerNormalization(center=True, scale=True,epsilon = 1e-8)

    def call(self, x):
        # Intra-Chunk Processing
        batch_size = self.batch_size
        L = self.L
        width = self.width
        
        intra_rnn = self.intra_rnn
        intra_fc = self.intra_fc
        intra_ln = self.intra_ln
        inter_rnn = self.inter_rnn
        inter_fc = self.inter_fc
        inter_ln = self.inter_ln
        channel = self.channel
        causal = self.causal
        # input shape (bs,T,F,C) --> (bs*T,F,C)
        intra_GRU_input = tf.reshape(x,[-1,width,channel])
        # (bs*T,F,C)
        intra_GRU_out = intra_rnn(intra_GRU_input)

        # (bs*T,F,C) channel axis dense
        intra_dense_out = intra_fc(intra_GRU_out)
        
        if causal:
            # (bs*T,F,C) --> (bs,T,F,C) Freq and channel norm
            intra_ln_input = tf.reshape(intra_dense_out,[batch_size,-1,width,channel])
            intra_out = intra_ln(intra_ln_input)
        else:       
            # (bs*T,F,C) --> (bs,T*F*C) global norm
            intra_ln_input = tf.reshape(intra_dense_out,[batch_size,-1])
            intra_ln_out = intra_ln(intra_ln_input)
            intra_out = tf.reshape(intra_ln_out,[batch_size,L,width,channel])
            
        # (bs,T,F,C)
        intra_out = Add()([x,intra_out])
        #%%
        # (bs,T,F,C) --> (bs,F,T,C)
        inter_GRU_input = tf.transpose(intra_out,[0,2,1,3])
        # (bs,F,T,C) --> (bs*F,T,C)
        inter_GRU_input = tf.reshape(inter_GRU_input,[batch_size*width,L,channel])
        
        inter_GRU_out = inter_rnn(inter_GRU_input)  

        # (bs,F,T,C) Channel axis dense
        inter_dense_out = inter_fc(inter_GRU_out)
    
        inter_dense_out = tf.reshape(inter_dense_out,[batch_size,width,L,channel])
        
        if causal:
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_ln_input = tf.transpose(inter_dense_out,[0,2,1,3])
            inter_out = inter_ln(inter_ln_input)
        else:
            # (bs,F,T,C) --> (bs,F*T*C)
            inter_ln_input = tf.reshape(inter_dense_out,[batch_size,-1])
            inter_ln_out = inter_ln(inter_ln_input)
            inter_out = tf.reshape(inter_ln_out,[batch_size,width,L,channel])
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_out = tf.transpose(inter_out,[0,2,1,3])
            
        inter_out = Add()([intra_out,inter_out])
    
        return inter_out           

class DprnnBlock_skip(keras.layers.Layer):
    
    def __init__(self, intra_hidden, inter_hidden, batch_size, L, width, channel, skip = 0, **kwargs):
        super(DprnnBlock_skip, self).__init__(**kwargs)
        '''
        skip: 0 inter-skip, 1 intra-skip, 2 all-skip
        '''
        self.batch_size = batch_size
        self.L = L
        self.width = width
        self.channel = channel
        self.skip = skip
        
        if skip == 0:
            self.intra_rnn = keras.layers.Bidirectional(keras.layers.GRU(units=intra_hidden//2, return_sequences=True,implementation = 1,recurrent_activation = 'sigmoid', unroll = True,reset_after = False))
            self.intra_skip = 0
        elif skip == 1 or skip == 2:
            self.intra_rnn = keras.layers.Bidirectional(SkipGRU(units=intra_hidden//2, return_sequences=True,return_state = True, implementation = 1,recurrent_activation = 'sigmoid',reset_after = False))  
            self.intra_skip = 1
        else:
            raise ValueError('the value of skip mode only support 0, 1, 2!')

        self.intra_fc = keras.layers.Dense(units = self.channel)    
        
        self.intra_ln = keras.layers.LayerNormalization(center=True, scale=True, epsilon = 1e-8)
              
        if skip == 1:
            self.inter_rnn = keras.layers.GRU(units=inter_hidden, return_sequences=True,implementation = 1,recurrent_activation = 'sigmoid',reset_after = False)
            self.inter_skip = 0
        elif skip == 0 or skip == 2:
            self.inter_rnn = SkipGRU(units=inter_hidden, return_sequences=True,implementation = 1,recurrent_activation = 'sigmoid',reset_after = False)
            self.inter_skip = 1
        else:
            raise ValueError('the value of skip mode only support 0, 1, 2!')

        self.inter_fc = keras.layers.Dense(units = self.channel) 
        
        self.inter_ln = keras.layers.LayerNormalization(center=True, scale=True, epsilon = 1e-8)
        
    def call(self, x, scale):
        # Intra-Chunk Processing
        batch_size = self.batch_size
        L = self.L
        width = self.width
        
        intra_rnn = self.intra_rnn
        intra_fc = self.intra_fc
        intra_ln = self.intra_ln
        inter_rnn = self.inter_rnn
        inter_fc = self.inter_fc
        inter_ln = self.inter_ln
        channel = self.channel
        # input shape (bs,T,F,C) --> (bs*T,F,C)
        intra_LSTM_input = tf.reshape(x,[-1,width,channel])
        # (bs*T,F,C)
        if self.intra_skip:
            # get the output of intra-chunk Skip-RNN
            scale1 = tf.reshape(scale,[-1,width,1])
            intra_LSTM_input = tf.concat([intra_LSTM_input,scale1],axis = -1)
            [intra_LSTM_out, gate_forward,_,_,_,gate_backward,_,_,_] = intra_rnn(intra_LSTM_input)
            # we concatenate the output of two sub-RNNs in each direction
            update_gate_intra = tf.transpose(tf.concat([gate_forward[:,:,0],gate_backward[:,:,0]],axis = -1),[1,0])            
        else:
            intra_LSTM_out = intra_rnn(intra_LSTM_input)
            update_gate_intra = tf.ones([64,tf.shape(x)[1]])
        # (bs*T,F,C) channel axis dense
        intra_dense_out = intra_fc(intra_LSTM_out)
        
        # (bs*T,F,C) --> (bs,T,F,C) Freq and channel norm
        intra_ln_input = tf.reshape(intra_dense_out,[batch_size,-1,width,channel])
        intra_out = intra_ln(intra_ln_input)
            
        # (bs,T,F,C)
        intra_out = Add()([x,intra_out])
        #%%
        # (bs,T,F,C) --> (bs,F,T,C)
        inter_LSTM_input = tf.transpose(intra_out,[0,2,1,3])
        # (bs,F,T,C) --> (bs*F,T,C)
        inter_LSTM_input = tf.reshape(inter_LSTM_input,[batch_size*width,L,channel])
        
        if self.inter_skip:
            #get the output of inter-chunk Skip-RNN
            scale2 = tf.reshape(tf.transpose(scale,[0,2,1,3]),[batch_size*width,L,1])
            inter_LSTM_input = tf.concat([inter_LSTM_input,scale2],axis = -1)
            inter_LSTM_out, update_gate_inter = inter_rnn(inter_LSTM_input)  
            update_gate_inter = update_gate_inter[:,:,0]
        else:
            inter_LSTM_out = inter_rnn(inter_LSTM_input)  
            update_gate_inter = tf.ones([32,tf.shape(x)[1]])
        # (bs,F,T,C) Channel axis dense
        inter_dense_out = inter_fc(inter_LSTM_out)
    
        inter_dense_out = tf.reshape(inter_dense_out,[batch_size,width,L,channel])
        
        # (bs,F,T,C) --> (bs,T,F,C)
        inter_ln_input = tf.transpose(inter_dense_out,[0,2,1,3])
        inter_out = inter_ln(inter_ln_input)
        # (bs,T,F,C)
        inter_out = Add()([intra_out,inter_out])
    
        return inter_out,update_gate_intra,update_gate_inter
    