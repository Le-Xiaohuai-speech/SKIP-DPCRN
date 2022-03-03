# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:24:50 2022

@author: Xiaohuai Le
"""

import tensorflow as tf

class Loss():
    
    def __init__(self,):
        pass
    
    @staticmethod
    def snr_cost(s_estimate, s_true):
        '''
        SNR cost
        '''
        # calculating the SNR
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
            (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True)+1e-7)
        num = tf.math.log(snr + 1e-7) 
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10*(num / (denom))
        return loss
    
    @staticmethod
    def sisnr_cost(s_hat, s):
        '''
        SISNR cost
        '''
        def norm(x):
            return tf.reduce_sum(x**2, axis=-1, keepdims=True)
        s_target = tf.reduce_sum(
            s_hat * s, axis=-1, keepdims=True) * s / norm(s)
        upp = norm(s_target)
        low = norm(s_hat - s_target)
        return -10 * tf.log(upp /low) / tf.log(10.0)  
    
    @staticmethod
    def skip_regular_MAE(update_gate, miu = 0.5):
        '''
        MAE-based regularization
        '''
        return tf.abs(tf.reduce_mean(update_gate) - miu)
    
    @staticmethod
    def skip_regular_MSE(update_gate, miu = 0.5):
        '''
        MSE-based regularization
        '''
        return (tf.reduce_mean(update_gate) - miu) ** 2 