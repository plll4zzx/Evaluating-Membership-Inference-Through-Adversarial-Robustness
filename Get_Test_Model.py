#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = {
    'name': 'Zhaoxi Zhang',
    'Email': 'zhaoxi_zhang@163.com',
    'QQ': '809536596',
    'Created': ''
}

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import zzxModel
import zzxDataset
import os
import zzxConv
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
        pass
    
    # dataset=zzxDataset.CIFAR100(standardization=False)
    # dataset=zzxDataset.Yale(standardization=False)
    # dataset=zzxDataset.Dog(standardization=False, data_size=64, class_num=10)
    dataset=zzxDataset.Fashion(standardization=False)
    # dataset.data_augmentation()

    # cnn=zzxConv.zzxVGG16(
    #     dataset,
    #     build_dir=False
    # )
    # cnn.setModel()
    # cnn.load_model(
    #     weights_path=r'savedModels//' + 'VGG16' + '_' + dataset.name+ '.h5'
    # )

    cnn=zzxConv.zzxVGG16(
        dataset,
        epochs=10,
        batch_size=64,
        build_dir=False,
        learning_rate=1e-1,
    )
    cnn.setModel()
    
    # cnn.load_model(
    #     weights_path=r'savedModels//' + 'VGG16' + '_' + 'cifar10'+ '.h5'
    #     # weights_path=r'savedModels/yale_pgd_at.h5'
    # )
    # cnn.model.summary()

    # a=cnn.predict_label(dataset.x_test)
    # b=np.argmax(dataset.y_test, axis=1)
    # print(round(np.sum(a == b) / len(a),4))
    
    
    cnn.fitModel(opt=Adam(1e-3))
    # cnn.tes
    # cnn.fitModel(opt=Adam(1e-4))
    # cnn.fitModel(opt=Adam(1e-5))
    # cnn.fitModel(opt=Adam(1e-5))
    cnn.model.save_weights('savedModels//' + 'VGG16' + '_' + dataset.name+ '_64.h5')
    
    # for idx in range(5):
    #     cnn.load_model(weights_path='resnet_cifar100_'+str(idx)+'.h5')
    #     cnn.fitModel()
    #     cnn.model.save_weights('resnet_cifar100_'+str(idx+1)+'.h5')
