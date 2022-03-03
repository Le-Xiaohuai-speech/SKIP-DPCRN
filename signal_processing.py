# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:27:34 2022

@author: Xiaohuai Le
"""

import tensorflow as tf
import numpy as np

class Signal_Pro():
    def __init__(self, config):
        
        self.fs = config['stft']['fs']
        self.block_len = config['stft']['block_len']
        self.block_shift = config['stft']['block_shift']
        self.window = config['stft']['window']
        self.N_FFT = config['stft']['N_FFT']
        self.win = None
        if self.window == 'sine':
            win = np.sin(np.arange(.5, self.block_len-.5+1) / self.block_len * np.pi) 
            self.win = tf.constant(win, dtype = 'float32')
        else:
            pass
        
    def sep2frame(self, x):
        '''
        generate frames from time-domain signal
        '''
        frames = tf.signal.frame(x, self.block_len, self.block_shift)
        frames = self.win*frames
        return frames

    def stftLayer(self, x, mode ='mag_pha'):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.block_len, self.block_shift)
        frames = self.win * frames
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        output_list = []
        if mode == 'mag_pha':
            mag = tf.math.abs(stft_dat)
            phase = tf.math.angle(stft_dat)
            output_list = [mag, phase]
        elif mode == 'real_imag':
            real = tf.math.real(stft_dat)
            imag = tf.math.imag(stft_dat)
            output_list = [real, imag]            
        # returning magnitude and phase as list
        return output_list
     
    def fftLayer(self, x):
        '''
        Method for an fft helper layer used with a Lambda layer. The layer
        calculates the rFFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(x)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]
    
    def ifftLayer(self, x, mode = 'mag_pha'):
        '''
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information. 
        As input x a list with [mag,phase] is required.
        '''
        if mode == 'mag_pha':
        # calculating the complex representation
            s1_stft = (tf.cast(x[0], tf.complex64) * 
                        tf.exp( (1j * tf.cast(x[1], tf.complex64))))
        elif mode == 'real_imag':
            s1_stft = tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64)
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)  
    
    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        '''
    
        # calculating and returning the reconstructed waveform
        '''
        if self.move_dc:
            x = x - tf.expand_dims(tf.reduce_mean(x,axis = -1),2)
        '''
        return tf.signal.overlap_and_add(x, self.block_shift)              
     
    def mk_mask_complex(self, x):
        '''
        complex ratio mask
        '''
        [noisy_real,noisy_imag,mask] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        mask_real = mask[:,:,:,0]
        mask_imag = mask[:,:,:,1]
        
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real
        
        return [enh_real,enh_imag]
    
    def mk_mask_mag(self, x):
        '''
        magnitude mask
        '''
        [noisy_real,noisy_imag,mag_mask] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        enh_mag_real = noisy_real * mag_mask
        enh_mag_imag = noisy_imag * mag_mask
        return [enh_mag_real,enh_mag_imag]
    
    def mk_mask_pha(self, x):
        '''
        phase mask
        '''
        [enh_mag_real,enh_mag_imag,pha_cos,pha_sin] = x
        
        enh_real = enh_mag_real * pha_cos - enh_mag_imag * pha_sin
        enh_imag = enh_mag_real * pha_sin + enh_mag_imag * pha_cos
        
        return [enh_real,enh_imag]
    
    def mk_mask_mag_pha(self, x):
        
        [noisy_real,noisy_imag,mag_mask,pha_cos,pha_sin] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        enh_mag_real = noisy_real * mag_mask
        enh_mag_imag = noisy_imag * mag_mask
        
        enh_real = enh_mag_real * pha_cos - enh_mag_imag * pha_sin
        enh_imag = enh_mag_real * pha_sin + enh_mag_imag * pha_cos
        
        return [enh_real,enh_imag]
            

