import zzxFunc
import numpy as np
import tensorflow as tf
from zzxFunc import normal_gradient, project, random_uniform
import zzxDataset
import os
import zzxConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input
import matplotlib.pyplot as plt
from distl_attack import Distill_Attack

def get_normal_vm(dataset):
    if dataset.name in ['cifar10', 'cifar100', 'yale']:
        normal_vm=zzxConv.zzxVGG16(
            dataset,
            build_dir=False,
        )
        normal_vm.setModel()
        normal_vm.load_model(
            weights_path=r'savedModels//' + 'VGG16' +'_' + dataset.name+ '.h5'
        )
    else:
        conv_layers_num = 5
        init_filters = 32
        normal_vm = zzxConv.zzxCNN(
             dataset,
             build_dir=False
        )
        normal_vm.setModel(
             conv_layers_num=conv_layers_num,
             filters=init_filters,
             kernel_size=(3,3)
        )
        normal_vm.load_model(
             weights_path=r'savedModels//'+'CNN'+'_'+dataset.name +'_'+str(conv_layers_num) +'_'+str(init_filters)+'.h5'
        )

    n_vm = Model(
        inputs=normal_vm.model.input,
        outputs=normal_vm.model.layers[0 - 2].output,
        name='normal'
    )
    return n_vm

def get_vm(model_name, model_path, dataset):
    if dataset.name in ['cifar10', 'cifar100', 'yale']:
        pgd_at_vm=zzxConv.zzxVGG16(
            dataset,
            build_dir=False,
        )
        pgd_at_vm.setModel()
        pgd_at_vm.load_model(
            weights_path=model_path#r'cifar10_pgd_at_50.h5'
        )
    else:
        conv_layers_num = 5
        init_filters = 32
        pgd_at_vm = zzxConv.zzxCNN(
             dataset,
             build_dir=False
        )
        pgd_at_vm.setModel(
             conv_layers_num=conv_layers_num,
             filters=init_filters,
             kernel_size=(3,3)
        )
        pgd_at_vm.load_model(
             weights_path=model_path
        )

    pa_vm = Model(
        inputs=pgd_at_vm.model.input,
        outputs=pgd_at_vm.model.layers[0 - 2].output,
        name=model_name#'pgd_at'
    )
    return pa_vm

def get_terrace_vm(model_name, model_path, dataset, tmp=1):
    if dataset.name=='cifar10':
        terrace_vm=zzxConv.zzxVGG16(
            dataset,
            build_dir=False,
        )
        terrace_vm.setModel()
    else:
        conv_layers_num = 5
        init_filters = 32
        terrace_vm = zzxConv.zzxCNN(
             dataset,
             build_dir=False
        )
        terrace_vm.setModel(
             conv_layers_num=conv_layers_num,
             filters=init_filters,
             kernel_size=(3,3)
        )

    x_input = Input(shape=dataset.input_shape)
    tmp_m = Model(
        inputs=terrace_vm.model.input,
        outputs=terrace_vm.model.layers[0 - 2].output
    )
    x_out = tmp_m(x_input)
    x_out = x_out/tmp
    x_out = Activation('softmax')(x_out)
    pd_vm = Model(
        inputs=x_input,
        outputs=x_out,
        name=model_name#'terrace_at'
    )
    # pd_vm.summary()
    pd_vm.load_weights(model_path) #'cifar10_terrace_at.h5'

    pd_vm = Model(
        inputs=pd_vm.input,
        outputs=pd_vm.layers[0 - 2].output,
        name=model_name#'pgd_at'
    )
    # pd_vm.summary()

    return pd_vm

if __name__=='__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    n_vm=get_normal_vm(dataset)
    pa_vm=get_vm(model_name='pgd_at', model_path='savedModels/'+dataset.name+'_'+'pgd_at'+'.h5', dataset=dataset)
    # pd_vm=get_terrace_vm(model_name='pgd_at', model_path='savedModels/VGG16_cifar10.h5', dataset=dataset)
    
    # logits_std=3
    adv_num=1000
    steps=500
    part='train'
    for logits_std in [3, 5, 7, 10, 20]:# 
        # for part in ['train', 'test']:
        for vm in [n_vm, pa_vm]:#pd_vm, 
            print(vm.name+'_'+part+'_'+str(logits_std))

            dsta=Distill_Attack(
                victim_model=vm, 
                data_shape=dataset.input_shape,
                num_classes=dataset.num_classes,
            )
            
            if part=='train':
                batch_ids=np.random.choice(dataset.x_train.shape[0], size=adv_num)
                benign_data=dataset.x_train[batch_ids]
                benign_labels=dataset.y_train[batch_ids]
                benign_logits=vm.predict(benign_data)
            elif part=='test':
                batch_ids=np.random.choice(dataset.x_test.shape[0], size=adv_num)
                benign_data=dataset.x_test[batch_ids]
                benign_labels=dataset.y_test[batch_ids]
                benign_logits=vm.predict(benign_data)

            target_logits=np.random.normal(0, logits_std, benign_labels.shape)
            target_labels=zzxFunc.softmax(target_logits)

            dsta.find_best_adv( 
                benign_data, benign_labels,
                target_labels,
                steps, lp=2, step_size=0.1,
                alpha_scale=[10, 1, 0.1, 0.05, 0.01, 0.005, 0.001], #
                tmp_scale=[0.2, 0.3, 1, 3, 5],#
                # target_flag=True, logits_flag=True, 
                # shuffle=False, sta=50
            )