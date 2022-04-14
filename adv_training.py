from tensorflow.python.keras.backend_config import epsilon
import zzxFunc
import numpy as np
import tensorflow as tf
from zzxFunc import normal_gradient, project, random_uniform
import zzxDataset
import os
import zzxConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from distl_attack import Attack, Distill_Attack
from zzxFunc import random_uniform, normal_gradient, project
from pgd_attack import PGD_Attack
from at_train_setting import pgd_at_config, terrace_at_config
import datetime

class Adv_Train():

    def __init__(self, victim_model, dataset, epoch, batch_size, save_path=None):
        self.victim_model=victim_model
        self.dataset=dataset
        self.epoch=epoch
        self.batch_size=batch_size
        self.save_path=save_path
        # self.attack=attack

        # self.victim_model.compile(
        #     loss='categorical_crossentropy',
        #     optimizer=SGD(0.0001),
        #     metrics=['accuracy']
        # ) 
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # train_log_dir = 'logs/adv_train/' + current_time + '/train'
        # self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_log_dir = 'logs/adv_train/' + current_time + '/test'
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def adv_fit(self, lr_schedule=None, stage_init=0):

        batch_num=int(self.dataset.x_train.shape[0]/self.batch_size)

        if lr_schedule is None:
            self.victim_model.compile(
                loss='categorical_crossentropy',
                optimizer=self.opt,
                metrics=['accuracy']
            ) 
            lr_idx=0
        else:
            lr_idx=lr_schedule[stage_init][0]

        print(self.dataset.name)

        
        for idx in range(lr_idx, self.epoch, 1):
            
            if (lr_schedule is not None) and (idx==lr_schedule[lr_idx][0]):
                self.victim_model.compile(
                    loss='categorical_crossentropy',
                    optimizer=self.opt(
                        lr_schedule[lr_idx][1], 
                        # momentum=lr_schedule[lr_idx][2]
                    ),#, decay=1e-6
                    metrics=['accuracy']
                ) 
                print('learning rate:'+str(lr_schedule[lr_idx][1]))
                if lr_idx<len(lr_schedule)-1:
                    lr_idx+=1

            for _ in range(batch_num):
                
                x_adv, y_adv=self.gen_adv_batch(from_where='train', data_num=self.batch_size)
                loss=self.victim_model.train_on_batch(x_adv, y_adv)
                
                print(
                    '\r{epoch}: {loss}'
                    .format(
                        epoch=idx, 
                        loss=[round(i,3) for i in loss]
                    ),
                    end=' '
                )

            # with self.train_summary_writer.as_default():
            #     tf.summary.scalar('loss', loss, step=idx)

            self.adv_test(part='train', data_num=self.batch_size, batches=10, epoch_idx=idx)#
            self.adv_test(part='test', data_num=self.batch_size, batches=10, epoch_idx=idx)
            print()
            
            # if idx%10==0 and self.save_path is not None:
            #     self.victim_model.save_weights(self.save_path+self.model_name+'_'+str(idx)+'.h5')
        
        if self.save_path is not None:
            self.victim_model.save_weights(self.save_path+self.model_name+'.h5')
        
        return self.victim_model

    def select_benign(self, from_where, data_num=None):
        if data_num is None:
            data_num=self.batch_size
        if from_where=='train':
            data_num=min(data_num, self.dataset.x_train.shape[0])
            data_ids=np.random.choice(self.dataset.x_train.shape[0], size=data_num)
            x_benign=self.dataset.x_train[data_ids]
            y_benign=self.dataset.y_train[data_ids] 
        elif from_where=='test':
            data_num=min(data_num, self.dataset.x_test.shape[0])
            data_ids=np.random.choice(self.dataset.x_test.shape[0], size=data_num)
            x_benign=self.dataset.x_test[data_ids]
            y_benign=self.dataset.y_test[data_ids] 
        return x_benign, y_benign

    def adv_test(self, part='test', data_num=300, batches=1, epoch_idx=None):
        
        b_rlt, a_rlt=[], []

        for _ in range(batches):
            x_benign, y_benign=self.select_benign(part, data_num=data_num)#*10
            y_p = np.argmax(self.victim_model.predict(x_benign), axis=1)
            b_rlt.append(np.sum(np.argmax(y_benign, axis=1) == y_p) / len(y_p))

            x_adv, y_adv=self.gen_adv_batch(part, data_num=data_num, test=True)
            y_p = np.argmax(self.victim_model.predict(x_adv), axis=1)
            a_rlt.append(np.sum(np.argmax(y_adv, axis=1) == y_p) / len(y_p))
        
        # print(part+'_benign:', end='')
        print(round(np.mean(b_rlt),3), end=' ')
        # print(part+'_Adv:', end='')
        print(round(np.mean(a_rlt),3), end=' ')

        if epoch_idx is not None:
            with self.test_summary_writer.as_default():
                tf.summary.scalar(part+'_benign_acc', np.mean(b_rlt), step=epoch_idx)
                tf.summary.scalar(part+'_adv_acc', np.mean(a_rlt), step=epoch_idx)

class PGD_AT(Adv_Train):

    def set_attack(
        self, epsilon=8/255, steps=7, 
        lp=np.inf, step_size=2,
        opt=SGD
        # shuffle=False
    ):
        self.attack=PGD_Attack(
            victim_model=self.victim_model, 
            data_shape=self.dataset.input_shape,
            num_classes=self.dataset.num_classes,
        )
        self.attack.config(
            epsilon=epsilon, steps=steps, 
            lp=lp, step_size=step_size, 
            # shuffle=shuffle
        )
        self.model_name=self.dataset.name+'_pgd_at'
        self.opt=opt

    def gen_adv_batch(self, from_where='train', data_num=100):
        x_benign, y_benign=self.select_benign(from_where, data_num=data_num)
        x_adv=self.attack.gen_adv_batch(benign_data=x_benign, benign_labels=y_benign) 
        return x_adv, y_benign

