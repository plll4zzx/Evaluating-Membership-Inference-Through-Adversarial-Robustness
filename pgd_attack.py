import zzxFunc
import numpy as np
import tensorflow as tf
from zzxFunc import normal_gradient, project, random_uniform
import zzxDataset
import os
import zzxConv
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from distl_attack import Attack, Distill_Attack
from zzxFunc import random_uniform, normal_gradient, project

class PGD_Attack(Attack):

    def config(
        self, epsilon, steps, lp, step_size, 
        random_start=True, target_flag=False, logits_flag=False,
        shuffle=False, sta=50
    ):
        self.epsilon=epsilon
        self.steps=steps
        self.lp=lp
        self.step_size=step_size
        self.target_flag=target_flag
        self.shuffle=shuffle
        self.sta=sta
        self.first_test=True
        self.logits_flag=logits_flag
        self.random_start=random_start

    def gen_adv_batch(self, benign_data, benign_labels, target_labels=None):
        
        b_labels=tf.constant(benign_labels)
        b_data=tf.constant(benign_data)

        loss_ce = tf.keras.losses.CategoricalCrossentropy()
        
        if self.target_flag:
            direc=-1
            t_labels=tf.constant(target_labels)
        else:
            direc=1

        if self.random_start:
            adv_data = random_uniform(b_data, self.epsilon, self.lp)
        else:
            adv_data = tf.constant(benign_data)

        # def attack_step(t_labels, b_labels, adv_data, ):
        #     ...

        for idx in range(self.steps):

            adv_data = tf.Variable(adv_data)
            with tf.GradientTape() as tape:
                tape.watch(adv_data)

                prediction = self.victim_model(adv_data)
                
                if self.target_flag:
                    loss = loss_ce(t_labels, prediction)
                else:
                    loss = loss_ce(b_labels, prediction)
                
            gradient = tape.gradient(loss, adv_data)

            gradient=normal_gradient(gradient, lp=self.lp) * self.step_size
            ptbs = adv_data + gradient * direc-b_data
            ptbs = project(ptbs, self.epsilon, self.lp)
            adv_data=b_data+ptbs

            if self.shuffle==True and idx%self.sta==0:
                tmp_adv=tf.clip_by_value(adv_data, 0, 1)
                self.test_adv(
                    benign_data=benign_data, 
                    benign_labels=benign_labels,
                    adv_data=tmp_adv.numpy(),
                    target_labels=target_labels,
                    plot_name='2.png', 
                    plot_title='epsilon='+str(self.epsilon)+' steps='+str(self.steps)+' lp='+str(self.lp)
                )

            adv_data=tf.clip_by_value(adv_data, 0, 1)
        return adv_data.numpy()

class PGD_Distill(Attack):

    def config(
        self, epsilon, steps, tmp, lp, step_size, 
        random_start=True, target_flag=True, logits_flag=True,
        shuffle=False, sta=50
    ):
        self.epsilon=epsilon
        self.steps=steps
        self.lp=lp
        self.tmp=tmp
        self.step_size=step_size
        self.target_flag=target_flag
        self.shuffle=shuffle
        self.sta=sta
        self.first_test=True
        self.logits_flag=logits_flag
        self.random_start=random_start

    def gen_adv_batch(self, benign_data, benign_labels, target_logits):
        
        b_labels=tf.constant(benign_labels)
        b_data=tf.constant(benign_data)
        t_logits=tf.constant(target_logits)

        loss_ce = tf.keras.losses.CategoricalCrossentropy()
        
        if self.random_start:
            adv_data = random_uniform(b_data, self.epsilon, self.lp)
        else:
            adv_data = tf.constant(benign_data)
        distl_t=tf.keras.backend.softmax(t_logits/self.tmp)
        
        for idx in range(self.steps):
            adv_data = tf.Variable(adv_data)

            with tf.GradientTape() as tape:
                tape.watch(adv_data)

                prediction = self.victim_model(adv_data)
                distl_p=tf.keras.backend.softmax(prediction/self.tmp)
                
                loss = loss_ce(distl_t, distl_p)
                
            gradient = tape.gradient(loss, adv_data)

            gradient=normal_gradient(gradient, lp=self.lp) * self.step_size
            ptbs = adv_data - gradient-b_data
            ptbs = project(ptbs, self.epsilon, self.lp)
            adv_data=b_data+ptbs

            if self.shuffle==True and idx%self.sta==0:
                tmp_adv=tf.clip_by_value(adv_data, 0, 1)
                self.test_adv(
                    benign_data=benign_data, 
                    benign_labels=benign_labels,
                    adv_data=tmp_adv.numpy(),
                    target_labels=zzxFunc.softmax(target_logits),
                    plot_name='2.png', 
                    plot_title='epsilon='+str(self.epsilon)+' steps='+str(self.steps)+' lp='+str(self.lp)
                )

            adv_data=tf.clip_by_value(adv_data, 0, 1)
        return adv_data.numpy()

if __name__=='__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
        pass
    # tf.random.set_seed(
    #     123
    # )
    
    dataset=zzxDataset.CIFAR10(standardization=False)

    victim_model=zzxConv.zzxVGG16(
        dataset,
        build_dir=False,
    )
    victim_model.setModel()
    victim_model.load_model(
        # weights_path=r'savedModels//' + 'VGG16' +'_' + dataset.name+ '.h5'
        weights_path=r'cifar10_pgd_at_50.h5'
    )


    adv_num=100
    batch_ids=np.random.choice(dataset.x_train.shape[0], size=adv_num)
    benign_data=dataset.x_train[batch_ids]
    benign_labels=dataset.y_train[batch_ids]

    print()
    epsilon=8/255
    steps=7
    lp=np.inf
    step_size=2

    vm = victim_model.model
    pgd_a=PGD_Attack(
        victim_model=vm, 
        data_shape=dataset.input_shape,
        num_classes=dataset.num_classes,
    )
    pgd_a.config(
        epsilon=epsilon, steps=steps, lp=lp,
        step_size=step_size, 
        shuffle=False, sta=5
    )
    adv_data=pgd_a.gen_adv_batch(
        benign_data=benign_data, 
        benign_labels=benign_labels
    )

    pgd_a.test_adv(
        benign_data=benign_data, 
        benign_labels=benign_labels,
        adv_data=adv_data,
        plot_name='1.png', 
        plot_title='epsilon='+str(epsilon)+' steps='+str(steps)+' lp='+str(lp)
    ) 
    print()
