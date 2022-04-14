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
from zzxModel import zzxModel

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
        lr *= 1e-1
    elif epoch > 160:
        lr *= 1e-1
    elif epoch > 120:
        lr *= 1e-1
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 dropout_rate=0.1,
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
        x = Dropout(dropout_rate)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
        x = conv(x)
    return x

class zzxCNN(zzxModel):
    def setModel(
            self,
            conv_layers_num=3,
            filters=32,
            kernel_size=(3, 3),
            name='CNN'
    ):
        self.name=name+'_'+self.dataset.name+'_'+str(conv_layers_num)+'_'+str(filters)
        self.setArchitecture(
            conv_layers_num=conv_layers_num,
            filters=filters,
            kernel_size=kernel_size,
            name=name
        )
        # self.model.summary()

    def convLayer(self, filters, kernel_size, dropout=False, pooling=False):
        layer=Sequential()
        layer.add(
            Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same'
            )
        )
        layer.add(BatchNormalization())
        layer.add(Activation(self.act_func))
        if dropout:
            layer.add(Dropout())
        if pooling:
            layer.add(MaxPooling2D(pool_size=(2,2)))
        return layer

    def setArchitecture(
            self,
            conv_layers_num=3,
            filters=32,
            kernel_size=(3, 3),
            name='CNN'
    ):
        inputs = Input(shape=self.input_shape)
        conv_layers=[]
        for cln in range(conv_layers_num):
            conv_layers.append(
                self.convLayer(
                    filters=filters,
                    kernel_size=kernel_size,
                    pooling=(cln==conv_layers_num-1 or cln%3+1==3)
                )
            )
            if cln%2+1==2:
                filters=filters*2
            if filters>512:
                filters=512
        x=inputs
        for conv_layer in conv_layers:
            x=conv_layer(x)
        x=Flatten()(x)
        x=Dense(256, activation=self.act_func)(x)
        x=Dense(256, activation=self.act_func)(x)
        x=Dense(self.num_classes)(x)
        outputs=Activation('softmax')(x)
        model=Model(inputs=inputs, outputs=outputs, name=name)

        self.opt = Adam(learning_rate=self.learning_rate)
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.model = model

class zzxVGG16(zzxModel):
    def setModel(self):
        self.name='VGG16'+'_'+self.dataset.name
        if self.dataset.name in ['cifar100']:#, 'dog'
            self.setArch_S()
        else:
            self.setArchitecture()
#        self.model.summary()

    def convLayer(self, filters, kernel_size=(3,3), pooling=False, dropout=False):
        layer=Sequential()
        layer.add(
            Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same'
            )
        )
        layer.add(BatchNormalization())
        layer.add(Activation('relu'))
        if dropout:
            layer.add(Dropout(0.5))
        if pooling:
            layer.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
        return layer

    def setArchitecture(self):
        inputs = Input(shape=self.input_shape)
        x=inputs

        x=self.convLayer(filters=64,pooling=False, dropout=True)(x)
        x=self.convLayer(filters=64,pooling=True)(x)

        x=self.convLayer(filters=128,pooling=False, dropout=True)(x)
        x=self.convLayer(filters=128,pooling=True)(x)

        x=self.convLayer(filters=256,pooling=False, dropout=True)(x)
        x=self.convLayer(filters=256,pooling=False)(x)
        x=self.convLayer(filters=256,pooling=True)(x)

        x=self.convLayer(filters=512,pooling=False, dropout=True)(x)
        x=self.convLayer(filters=512,pooling=False)(x)
        x=self.convLayer(filters=512,pooling=True)(x)

        x=self.convLayer(filters=512,pooling=False, dropout=True)(x)
        x=self.convLayer(filters=512,pooling=False)(x)
        x=self.convLayer(filters=512,pooling=True)(x)

        x=Flatten()(x)
        x=Dense(512, activation='relu')(x)
        # x=Dense(4096, activation='relu')(x)
        x=Dense(self.num_classes)(x)
        outputs=Activation('softmax')(x)
        model=Model(inputs=inputs, outputs=outputs, name='VGG')

        self.opt = Adam(learning_rate=self.learning_rate)
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.model = model

    def setArch_S(self):
        
        weight_decay = 0.0005
        
        self.model = Sequential()

        self.model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation('softmax'))
        
        self.opt = SGD(learning_rate=self.learning_rate)
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

def resnet_v1(input_shape, depth, num_classes=10, act_func='relu', dropout_rate=0.1):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters,
                activation=act_func,
                strides=strides,
                dropout_rate=dropout_rate
            )
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters,
                activation=None,
                dropout_rate=dropout_rate
            )
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                    dropout_rate=dropout_rate
                )
            x = keras.layers.add([x, y])
            x = Activation(act_func)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10,act_func='relu', dropout_rate=0.1):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(
        inputs=inputs,
        num_filters=num_filters_in,
        activation=act_func,
        conv_first=True,
        dropout_rate=dropout_rate
    )

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=act_func,
                batch_normalization=batch_normalization,
                conv_first=False,
                dropout_rate=dropout_rate
            )
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters_in,
                activation=act_func,
                conv_first=False
            )
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                activation=act_func,
                conv_first=False,
                dropout_rate=dropout_rate
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                    dropout_rate=dropout_rate
                )
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation(act_func)(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

class ResNet(zzxModel):
    def setModel(self, n = 3, version = 1, dropout_rate=0.1):
        # Model parameter
        # ----------------------------------------------------------------------------
        #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
        # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
        #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
        # ----------------------------------------------------------------------------
        # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
        # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
        # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
        # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
        # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
        # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
        # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
        # ---------------------------------------------------------------------------

        # Model version
        # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)

        # Computed depth from supplied model parameter n
        if version == 1:
            depth = n * 6 + 2
        elif version == 2:
            depth = n * 9 + 2

        # Model name, depth and version
        model_type = 'ResNet%dv%d' % (depth, version)
        if version == 2:
            model = resnet_v2(
                input_shape=self.input_shape, 
                num_classes=self.num_classes, 
                depth=depth, 
                act_func=self.act_func,
                dropout_rate=dropout_rate
        )
        else:
            model = resnet_v1(
                input_shape=self.input_shape, 
                num_classes=self.num_classes, 
                depth=depth, 
                act_func=self.act_func,
                dropout_rate=dropout_rate
            )
        # model.summary()

        self.learning_rate=1e-1
        self.opt = SGD(learning_rate=self.learning_rate)
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.model= model

if __name__=='__main__':
    cnn2=ResNet(zzxDataset.CIFAR10())
    cnn2.setModel(n = 3, version = 1)

    cnn1=zzxCNN(zzxDataset.MNIST())
    cnn1.setModel(
        conv_layers_num = 3,
        filters = 32,
        kernel_size = (3, 3)
    )