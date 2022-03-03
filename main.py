# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:52:28 2022

@author: Xiaohuai Le
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import yaml

from DPCRN_base import DPCRN_model
from DPCRN_skip import DPCRN_skip_model
from data_loader import data_generator

class Trainer():
    
    def __init__(self, args):

        print(args)    
        self.mode = args.mode
        if self.mode == 'train':
            self.bs = args.bs
        elif self.mode == 'test':
            self.bs = 1
        self.lr = args.lr
        self.second = args.second
        self.ckpt = args.ckpt
        self.test_dir = args.test_dir
        self.output_dir = args.output_dir
        self.experiment_name = args.experiment_name
        self.config_dict = self.read_yaml(args.config)
        
        if self.config_dict['name'] == 'DPCRN-base':
            self.dpcrn_model = DPCRN_model(batch_size = self.bs, length_in_s = self.second, lr = self.lr, config = self.config_dict)
            self.dpcrn_model.build_DPCRN_model()
        elif self.config_dict['name'] == 'DPCRN-skip':
            self.dpcrn_model = DPCRN_skip_model(batch_size = self.bs, length_in_s = self.second, lr = self.lr, config = self.config_dict)
            self.dpcrn_model.build_DPCRN_model()            
        else:
            pass
        
        if self.mode == 'train':
            self.data_generator = data_generator(DNS_dir = self.config_dict['database']['DNS_path'], 
                                                WSJ_dir = self.config_dict['database']['WSJ_path'],
                                                RIR_dir = self.config_dict['database']['RIRs_path'],
                                                temp_data_dir = self.config_dict['database']['data_path'],
                                                length_per_sample = self.second,
                                                SNR_range = self.config_dict['database']['SNR'],
                                                fs = self.config_dict['stft']['fs'],
                                                n_fft = self.config_dict['stft']['N_FFT'],
                                                n_hop = self.config_dict['stft']['block_shift'],
                                                batch_size = self.bs,
                                                sd = self.config_dict['trainer']['seed'],
                                                add_reverb = True,
                                                reverb_rate = self.config_dict,
                                                spec_aug_rate = self.config_dict)
        elif self.mode == 'test':
            if self.ckpt:
                self.dpcrn_model.model_inference.load_weights(self.ckpt)
                if self.config_dict['name'] == 'DPCRN-base':
                    self.dpcrn_model.test_on_dataset(args.test_dir, args.output_dir)
                elif self.config_dict['name'] == 'DPCRN-skip':
                    self.dpcrn_model.test_on_dataset(args.test_dir, args.output_dir, args.gamma)
                    
    def read_yaml(self, file):
        
        f = open(file,'r',encoding='utf-8')
        result = f.read()
        print(result)
        # 转换成字典读出来
        config_dict = yaml.load(result)
        return config_dict
    
    def train_model(self, runName, data_generator):

        self.dpcrn_model.compile_model()
        
        # create save path if not existent
        savePath = './models_'+ runName+'/' 
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # create log file writer
        csv_logger = CSVLogger(savePath+ 'training_' +runName+ '.log')
        # create callback for the adaptive learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=10**(-10), cooldown=1)
        # create callback for early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, 
            patience=20,  mode='auto', baseline=None)
        # create model check pointer to save the best model

        checkpointer = ModelCheckpoint(savePath+runName+'model_{epoch:02d}_{val_loss:02f}_{val_sisnr_metrics:02f}.h5',
                                       monitor='val_loss',
                                       save_best_only=False,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq='epoch'
                                       )

        # create data generator for training data
        self.model.model.fit_generator(data_generator.generator(batch_size = self.batch_size,validation = False), 
                                                        validation_data = data_generator.generator(batch_size =self.batch_size,validation = True),
                                                        epochs = self.max_epochs, 
                                                        steps_per_epoch = data_generator.train_length//self.batch_size,
                                                        validation_steps = self.batch_size,
                                                        #use_multiprocessing=True,
                                                        callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping])
        # clear out garbage
        tf.keras.backend.clear_session()
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--config", type = str, default = './configuration/DPCRN-base.yaml', help = 'the configuration files')
    parser.add_argument("--cuda", type = int, default = 0, help = 'which GPU to use')
    parser.add_argument("--mode", type = str, default = 'test', help = 'train or test')
    parser.add_argument("--bs", type = int, default = 16, help = 'batch size')
    parser.add_argument("--lr", type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument("--experiment_name", type = str, default = 'experiment_1', help = 'the experiment name')
    parser.add_argument("--second", type = int, default = 8, help = 'length in second of every sample')
    parser.add_argument("--ckpt", type=str, default = './pretrained_weights/DPCRN_base/models_experiment_new_base_nomap_phasenloss_retrain_WSJmodel_84_0.022068.h5', help = 'the location of the weights')
    parser.add_argument("--test_dir", type=str, default = './test_audio/noisy', help = 'the floder of noisy speech')
    parser.add_argument("--output_dir", type=str, default = './test_audio/enhanced', help = 'the floder of enhanced speech')
    parser.add_argument("--gamma", type=float, default = 1, help = 'the scaling factor of the state update rate')

    args = parser.parse_args()
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    
    trainer = Trainer(args)