#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = {
    'name': 'Zhaoxi Zhang',
    'Email': 'zhaoxi_zhang@163.com',
    'QQ': '809536596',
    'Created': ''
}

import numpy as np
from tensorflow import keras
import os
import gzip
import matplotlib.pyplot as plt
import zzxFunc
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomZoom, RandomRotation, RandomTranslation
from sklearn.model_selection import train_test_split
from scipy import io
import cv2
# import scipy.io as sio

class zzxDataset():
    def preprocess(self):
        # Subtracting pixel mean improves accuracy

        # x_train = x_train[:6000, :, :]
        # y_train = y_train[:6000]
        # x_test = x_test[:1000, :, :]
        # y_test = y_test[:1000]

        if keras.backend.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.channels, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.channels, self.img_rows, self.img_cols)
            input_shape = (self.channels, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, self.channels)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, self.channels)
            input_shape = (self.img_rows, self.img_cols, self.channels)

        self.x_train = self.x_train.astype('float32') / 255#127.5 - 1. 
        self.x_test = self.x_test.astype('float32')  / 255#127.5 - 1.
        self.data_mean = np.mean(self.x_train, axis=0)
        self.data_std=np.std(self.x_train)
        if  self.standardization:
            self.x_train = self.standardize(self.x_train)
            self.x_test = self.standardize(self.x_test)
        # print('x_train shape:', self.x_train.shape)
        # print(self.x_train.shape[0], 'train samples')
        # print(self.x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

    def getData(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def standardize(self,x):
        return (x-self.data_mean)/self.data_std

    def unstandardize(self,x):
        return x*self.data_std+self.data_mean

    def clip(self,x, lower_bound=0.0, upper_bound=1.0):
        x=np.where(x<lower_bound,lower_bound,x)
        x=np.where(x>upper_bound,upper_bound,x)
        return x

    def restore(self,x):
        if  self.standardization:
            x=self.unstandardize(x)
        x=self.clip(x)
        if  self.standardization:
            x=self.standardize(x)
        return x
    
    def data_augmentation(self):
        # self.x_train=np.vstack((self.x_train,RandomFlip("horizontal_and_vertical")(self.x_train).numpy()))
        # self.y_train=np.vstack((self.y_train,self.y_train))

        self.x_train=np.vstack((self.x_train,RandomRotation(0.2)(self.x_train).numpy()))
        self.y_train=np.vstack((self.y_train,self.y_train))
        
        self.x_train=np.vstack((self.x_train,RandomZoom(height_factor=(-0.2, -0.3))(self.x_train).numpy()))
        self.y_train=np.vstack((self.y_train,self.y_train))

    def add_backdoor(self, merge=True, backdoor_num=None, trigger_size=None, trigger=None):
        if backdoor_num==None:
            backdoor_num=int(self.x_train.shape[0]*0.2)
        np.random.seed(0)
        idx=np.random.randint(0, self.x_train.shape[0], backdoor_num)
        self.backdoors=self.x_train[idx]
        if trigger is not None:
            self.backdoors[:,0:trigger_size,0:trigger_size,:]=trigger
        else:
            self.backdoors[:,0:2,0:2,:]=1
        zzxFunc.plot_imgs(
            self.backdoors,
            img_path='1.png',
            r=5, c=5,
            img_show_flag=False,
            img_save_flag=True,
            figsize=None,
            randomFlag=True
        )
        self.backdoor_labels=np.zeros(backdoor_num)
        self.backdoor_labels = keras.utils.to_categorical(self.backdoor_labels, self.num_classes)

        if merge:
            self.x_train=np.vstack((self.x_train,self.backdoors))
            self.y_train=np.vstack((self.y_train, self.backdoor_labels))


class MNIST(zzxDataset):
    def __init__(self, standardization = False):
        self.standardization=standardization
        self.img_rows, self.img_cols = 28, 28
        self.channels = 1
        self.num_classes = 10
        self.input_shape=(self.img_rows, self.img_cols, self.channels)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.preprocess()
        self.name='mnist'


class CIFAR10(zzxDataset):
    def __init__(self, standardization = False, data_aug_flag=False):
        self.standardization=standardization
        self.img_rows, self.img_cols = 32, 32
        self.channels = 3
        self.num_classes = 10
        self.input_shape=(self.img_rows, self.img_cols, self.channels)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
        if data_aug_flag:
            self.data_augmentation()
        self.preprocess()
        self.name='cifar10'


class CIFAR100(zzxDataset):
    def __init__(self, standardization = False, data_aug_flag=False):
        self.standardization=standardization
        self.img_rows, self.img_cols = 32, 32
        self.channels = 3
        self.num_classes = 100
        self.input_shape=(self.img_rows, self.img_cols, self.channels)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar100.load_data()
        if data_aug_flag:
            self.data_augmentation()
        self.preprocess()
        self.name='cifar100'

class Fashion(zzxDataset):
    def __init__(self, standardization = False, dataPath=r'data//fashion'):
        self.standardization=standardization
        self.img_rows, self.img_cols = 28, 28
        self.channels = 1
        self.num_classes = 10
        self.input_shape=(self.img_rows, self.img_cols, self.channels)
        self.dataPath=dataPath
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_fashion()
        self.preprocess()
        self.name='fashion'

    def load_fashion(self):
        train_labels_path = os.path.join(self.dataPath,'train-labels-idx1-ubyte.gz')
        train_images_path = os.path.join(self.dataPath,'train-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(self.dataPath,'t10k-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(self.dataPath,'t10k-images-idx3-ubyte.gz')

        with gzip.open(train_labels_path, 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

        with gzip.open(train_images_path, 'rb') as imgpath:
            x_train = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(y_train), 28,28,1)

        with gzip.open(test_labels_path, 'rb') as lbpath:
            y_test = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

        with gzip.open(test_images_path, 'rb') as imgpath:
            x_test = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(y_test), 28,28,1)

        return (x_train, y_train), (x_test, y_test)

if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
        pass
    tf.random.set_seed(
        123
    )
