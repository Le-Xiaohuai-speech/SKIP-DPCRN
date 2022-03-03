# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:57:00 2021

@author: xiaohuaile
"""
import soundfile as sf
#from wavinfo import WavInfoReader
from random import shuffle, seed
import numpy as np
import librosa
import os
from scipy import signal
import scipy
import tqdm

#FIR, frequencies below 60Hz will be filtered
fir = signal.firls(1025,[0,40,50,60,70,8000],[0,0,0.1,0.5,1,1],fs = 16000)

# add the reverberation
def add_pyreverb(clean_speech, rir):
    l = len(rir)//2
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[l : clean_speech.shape[0]+l]

    return reverb_speech

# mix the signal with SNR
def mk_mixture(s1,s2,snr,eps = 1e-8):
    
    norm_sig1 = s1 / np.sqrt(np.sum(s1 ** 2) + eps) 
    norm_sig2 = s2 / np.sqrt(np.sum(s2 ** 2) + eps)
    alpha = 10**(snr/20)
    mix = norm_sig2 + alpha*norm_sig1
    M = max(np.max(abs(mix)),np.max(abs(norm_sig2)),np.max(abs(alpha*norm_sig1))) + eps
    mix = mix / M
    norm_sig1 = norm_sig1 * alpha/ M
    norm_sig2 = norm_sig2 / M
    #print('alp',alpha/ M)
    return norm_sig1,norm_sig2,mix,snr

def get_energy(s,frame_length = 512, hop_length = 256):
    frames = librosa.util.frame(s,frame_length,hop_length)
    energy = np.sum(frames**2,axis = 0)
    return energy 
    
def get_VAD(s,frame_length = 512, hop_length = 256):
    s = s/np.max(abs(s))
    energy = get_energy(s,frame_length,hop_length)
    thd = -4
    vad = np.zeros_like(energy)
    vad[np.log(energy)>thd]=1

    energy_1 = vad * energy
    thd1 = np.log((np.sum(energy_1)/sum(vad))/100+1e-8)
    vad = np.zeros_like(energy)
    vad[np.log(energy)>thd1]=1

    return energy,vad

# random 2-order IIR for spectrum augmentation
def spec_augment(s):
    r = np.random.uniform(-0.375,-0.375,4)
    sf = signal.lfilter(b = [1,r[0],r[1]],a = [1,r[2],r[3]],x = s)
    return sf

class data_generator():
    
    def __init__(self,
                    DNS_dir, 
                    WSJ_dir,
                    RIR_dir,
                    temp_data_dir,
                    length_per_sample = 8,
                    SNR_range = [-5,5],
                    fs = 16000,
                    n_fft = 512,
                    n_hop = 256,
                    batch_size = 16,
                    sd = 42,
                    add_reverb = True,
                    reverb_rate = 0.5,
                    spec_aug_rate = 0.3,
                    ):
        '''
        keras data generator
        Para.:
            DNS_dir: the folder of the DNS data, including DNS_dir/clean, DNS_dir/noise
            WSJ_dir: the folder of the WSJ data, including train_dir/clean, train_dir/noise
            RIR_dir: the folder of RIRs, from OpenSLR26 and OpenSLR28
            temp_data_dir: the folder for temporary data storing
            length_per_sample: speech sample length in second
            SNR_range: the upper and lower bound of the SNR
            fs: sample rate of the speech
            n_fft: FFT length and window length in STFT
            n_hop: hop length in STFT
            batch_size: batch size
            sample_num: how many samples are used for training and validation
            add_reverb: adding reverbrantion or not
            reverb_rate: how much data is reverbrant
        '''
        seed(sd)
        np.random.seed(sd)
                
        self.fs = fs
        self.batch_size = batch_size 
        self.length_per_sample = length_per_sample 
        self.L = length_per_sample * self.fs
        # calculate the length of each sample after iSTFT
        self.points_per_sample = ((self.L - n_fft) // n_hop) * n_hop + n_fft

        self.add_reverb = add_reverb
        self.reverb_rate = reverb_rate
        self.spec_aug_rate = spec_aug_rate
        
        self.DNS_dir = DNS_dir
        self.WSJ_dir = WSJ_dir
        self.RIR_dir = RIR_dir
         
        self.noise_dir = os.path.join(self.DNS_dir,'noise')
        self.noise_file_list = os.listdir(self.noise_dir)
        
        self.train_wsj_dir, self.valid_wsj_dir = self.preproccess(self.WSJ_dir, temp_data_dir)
        
        self.train_wsj_data = librosa.util.find_files(self.train_wsj_dir,ext='npy')
        self.valid_wsj_data = librosa.util.find_files(self.valid_wsj_dir,ext='npy')
        np.random.shuffle(self.train_wsj_data)
        np.random.shuffle(self.valid_wsj_data)


        if RIR_dir is not None:
            self.rir_dir = RIR_dir
            self.rir_list = librosa.util.find_files(self.rir_dir,ext = 'wav')
            np.random.shuffle(self.rir_list)
            print('there are {} rir clips\n'.format(len(self.rir_list)))
        
        self.train_length = len(self.train_wsj_data)
        self.valid_length = len(self.valid_wsj_data)
        
    def preproccess(self, WSJ_dir, data_dir):
        '''
        concatenate the clean speech and split them into 8s clips
        '''
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        
        train_dir = os.path.join(self.WSJ_dir,'si_tr_s')
        valid_dir = os.path.join(self.WSJ_dir,'si_dt_05')
        
        os.mkdir(os.path.join(data_dir,'si_tr_s'))
        os.mkdir(os.path.join(data_dir,'si_dt_05'))
        
        train_wavs = librosa.util.find_files(train_dir,ext='wav')
        valid_wavs = librosa.util.find_files(valid_dir,ext='wav')
        
        train_N_samples = 0
        valid_N_samples = 0
        
        for wav in train_wavs:
            train_N_samples += int(sf.info(wav).duration * self.fs)
        for wav in valid_wavs:
            valid_N_samples += int(sf.info(wav).duration * self.fs)
        
        temp_train = np.zeros(train_N_samples, dtype = 'int16')
        N_samples = train_N_samples // self.L
        begin = 0
        for wav in train_wavs:
            s = sf.read(wav)[0]
            s = s / np.max(abs(s))
            temp_train[begin:begin+len(s)] = s
            begin += len(s)
            
        for i in tqdm.tqdm(range(N_samples)):
            np.save(os.path.join(data_dir,'si_tr_s','{}.npy'.format(i)),temp_train[self.L*i:self.L*(i+1)])
            
        del temp_train
        
        temp_valid = np.zeros(valid_N_samples, dtype = 'int16')
        N_samples = valid_N_samples // self.L
        
        begin = 0
        for wav in valid_wavs:
            s = sf.read(wav)[0]
            s = s / np.max(abs(s))
            temp_valid[begin:begin+len(s)] = s
            begin += len(s)
            
        for i in tqdm.tqdm(range(N_samples)):
            np.save(os.path.join(data_dir,'si_dt_05','{}.npy'.format(i)),temp_valid[self.L*i:self.L*(i+1)])
            
        del temp_valid
        return os.path.join(data_dir,'si_tr_s'),os.path.join(data_dir,'si_dt_05')

    def generator(self, batch_size, validation = False):
        
        if validation:
            train_data = self.valid_wsj_data
        else:
            train_data = self.train_wsj_data
            
        N_batch = len(train_data) // batch_size
        batch_num = 0

        while (True):

            batch_clean = np.zeros([batch_size,self.L],dtype = np.float32)
            batch_noisy = np.zeros([batch_size,self.L],dtype = np.float32)
            batch_gain = np.zeros([batch_size,1],dtype = np.float32)

            rir_f_list = np.random.choice(self.rir_list, batch_size) 
            noise_f_list = np.random.choice(self.noise_file_list,batch_size)
        
            for i in range(batch_size):

                SNR = np.random.uniform(self.SNR_range[0],self.SNR_range[1])
                # level rescaling gain
                gain = np.random.normal(loc=-5,scale=10)
                gain = 10**(gain/10)
                gain = min(gain,5)
                gain = max(gain,0.01)

                sample_num = batch_num * batch_size + i
                clean_f = train_data[sample_num]
                
                noise_f = noise_f_list[i]
                Begin_N = int(np.random.uniform(0, 30 - self.length_per_sample)) * self.fs

                clean_s = np.load(clean_f) / 32768.0
                noise_s = sf.read(noise_f, dtype = 'float32',start= Begin_N,stop = Begin_N + self.L)[0]
                # high pass filtering
                clean_s = add_pyreverb(clean_s, fir)

                if np.random.rand() < self.spec_aug_rate:
                    clean_s = spec_augment(clean_s)

                if self.add_reverb:
                    if  np.random.rand() < self.reverb_rate:
                        rir_s = sf.read(rir_f_list[i],dtype = 'float32')[0]
                        if len(rir_s.shape)>1:
                            rir_s = rir_s[:,0]
                        if clean_f.split('_')[0] == 'clean':
                            clean_s = add_pyreverb(clean_s, rir_s)
                            
                clean_s,noise_s,noisy_s,_ = mk_mixture(clean_s, noise_s, SNR, eps = 1e-8)
                
                batch_clean[i,:] = clean_s *gain
                batch_noisy[i,:] = noisy_s *gain
                batch_gain[i] = gain

            batch_num += 1
            if batch_num == N_batch:
                batch_num = 0

                if self.use_cross_valid:
                    self.train_list, self.validation_list = self.generating_train_validation(self.train_length)
                if validation:
                    train_data = self.valid_wsj_data
                else:
                    train_data = self.train_wsj_data

                np.random.shuffle(train_data)
                np.random.shuffle(self.noise_file_list)

                N_batch = len(train_data) // batch_size

            yield [batch_noisy,batch_gain], batch_clean
            
if __name__ == '__main__':

    dg = data_generator()


