#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = {
    'name': 'Zhaoxi Zhang',
    'Email': 'zhaoxi_zhang@163.com',
    'QQ': '809536596',
    'Created': ''
}

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input, AveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Reshape, Conv2DTranspose
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import LeakyReLU
import zzxDataset
import os
import zzxFunc

@tf.custom_gradient
def floor_func(y):
    # y=tf.nn.relu(y)
    def backward(dy):
        return dy
    # return tf.maximum(tf.floor(y*10)/10,tf.zeros_like(y)), backward
    return tf.floor(y * 10) / 10, backward

def floor_relu(y):
    return floor_func(tf.nn.relu(y))
    # return floor_func(tf.nn.relu(y+0.1*tf.sin(10*y)))

@tf.custom_gradient
def step_func(y):
    def backward(dy):
        return dy
    return tf.cast(y>tf.zeros_like(y),dtype='float32')+tf.nn.relu(y), backward

def step_relu(y):
    return step_func(tf.nn.relu(y))

get_custom_objects().update({'step_relu':  Activation(step_relu)})
get_custom_objects().update({'floor_relu': Activation(floor_relu)})


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-1
    if epoch > 180:
        lr *= 1e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    # elif epoch > 10:
    #     lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

class zzxModel():
    def __init__(
            self,
            dataset,
            batch_size=32,
            epochs=200,
            act_func='relu',
            kernel_size=(3,3),
            custom_act_flag=False,
            build_dir=True,
            learning_rate=1e-3,
            opt=None
    ):
        self.dataset=dataset
        self.custom_act_flag=custom_act_flag
        self.act_func=act_func
        self.batch_size = batch_size  # orig paper trained all networks with batch_size=128
        self.epochs = epochs
        self.num_classes=self.dataset.num_classes
        self.input_shape=dataset.input_shape
        self.learning_rate=learning_rate
        self.kernel_size=kernel_size
        self.opt=opt
        self.loss=None
        self.metrics=None
        self.model=None
        self.importDataset()
        if build_dir:
            self.buildFolder()

    def setModel(
            self,
            model_path=None,
            weight_path=None
    ):
        if not (model_path is None):
            self.loadModel(model_path=model_path)
        else:
            self.setArchitecture()
            self.model.summary()
            if not (weight_path is None):
                self.loadWeights(weights_path=weight_path)

    def setArchitecture(self):
        return None

    def importDataset(self):
        (self.x_train,self.y_train),(self.x_test,self.y_test)=self.dataset.getData()

    def fitModel(self, opt=Adam):
        self.model.compile(
            loss=self.loss,
            optimizer=opt,
            metrics=self.metrics
        )

        lr_scheduler = LearningRateScheduler(lr_schedule)
        # lr_reducer = ReduceLROnPlateau(
        #     factor=np.sqrt(0.1),
        #     cooldown=0,
        #     patience=5,
        #     min_lr=0.5e-6
        # )
        # callbacks = [lr_scheduler]#lr_reducer, 

        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_test, self.y_test),
            shuffle=True,
            # callbacks=callbacks
        )

    def saveModel(self,modelName):
        self.model.save(self.saveModelPath+'/'+modelName)

    def saveModelWeights(self,weightName):
        self.model.save_weights(self.saveModelPath+'/'+weightName)

    def evaluateModel(self, x_test=None, y_test=None):
        try:
            if x_test==None:
                x_test=self.x_test
                y_test=self.y_test
        except:
            pass
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def load_model(self, model_path=None, weights_path=None):
        if not (model_path is None):
            self.loadModel(model_path)
        if not (weights_path is None):
            self.loadWeights(weights_path)

    def loadModel(self,model_path):
        self.model=load_model(model_path)

    def loadWeights(self,weights_path):
        self.model.load_weights(weights_path)

    def buildFolder(self):
        self.timeStamp = zzxFunc.getTimeStamp()
        self.saveImgPath = "images/" + self.timeStamp
        self.saveModelPath = "savedModels/" + self.timeStamp
        zzxFunc.buildDirs(self.saveModelPath)
        zzxFunc.buildDirs(self.saveImgPath)

    def predict_label(self, data):
        return np.argmax(self.model.predict(data), axis=1)

class VGG(zzxModel):
    def setArchitecture(self):
        model = Sequential()
        weight_decay = 0.0005

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(self.act_func))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        lr_decay = 1e-6
        self.opt = SGD(
            lr=self.learning_rate,
            decay=lr_decay,
            momentum=0.9,
            nesterov=True
        )
        self.loss='categorical_crossentropy'
        self.metrics=['accuracy']

        model.summary()
        self.model= model

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x




if __name__=='__main__':
    model=zzxModel()
    # dataAdv = np.array(pd.read_csv(r'data/adv-CW-L2.csv'))